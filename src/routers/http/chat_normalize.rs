//! Opt-in normalization of OpenAI-style chat request bodies before they are
//! parsed (VLLM_ROUTER_NORMALIZE_CHAT=1).
//!
//! Aggregators such as OpenRouter forward request shapes that other providers
//! accept but this router's typed parser or vLLM rejects (422/400). A tool
//! request rejected that way counts against the endpoint's tool-call success.
//! The rewrites below are lossless or drop only fields the backend cannot
//! honour anyway:
//! - `tools`: flat Responses-style entries (`{"type":"function","name":..}`)
//!   are nested under `function`; a missing `type` becomes `function`; a
//!   function without `parameters` gets an empty object schema; non-function
//!   tools (web_search, custom, ...) are dropped; an empty list is removed.
//! - `tool_choice`: the flat `{"type":"function","name":..}` form is nested;
//!   `{"type":"auto"|"none"|"required"}` objects and Anthropic-style
//!   `{"type":"any"}` become strings; it is removed when no tools remain (a
//!   forced choice then has nothing to force).
//! - `stop`: lists are cut to the first 4 entries (vLLM's limit).
//! - assistant `tool_calls`: a missing `type` becomes `function` and object
//!   `arguments` are serialized to a JSON string.
use serde_json::{Map, Value};
use tracing::debug;

pub fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("VLLM_ROUTER_NORMALIZE_CHAT")
                .ok()
                .as_deref()
                .map(str::trim),
            Some("1" | "true" | "yes")
        )
    })
}

const MAX_STOP: usize = 4;

/// Normalize a chat request body in place. Returns the number of rewrites.
pub fn normalize_chat_body(body: &mut Value) -> usize {
    let Some(obj) = body.as_object_mut() else {
        return 0;
    };
    let mut n = 0;
    n += normalize_tools(obj);
    n += normalize_tool_choice(obj);
    n += normalize_stop(obj);
    if let Some(Value::Array(messages)) = obj.get_mut("messages") {
        for message in messages.iter_mut() {
            n += normalize_assistant_tool_calls(message);
        }
    }
    if n > 0 {
        debug!(rewrites = n, "normalized chat request body");
    }
    n
}

fn empty_schema() -> Value {
    serde_json::json!({"type": "object", "properties": {}})
}

/// Returns Some(normalized tool) or None to drop it, plus rewrite count.
fn normalize_tool(tool: Value) -> (Option<Value>, usize) {
    let Value::Object(mut t) = tool else {
        return (None, 1);
    };
    let mut n = 0;
    let ty = t.get("type").and_then(Value::as_str).map(str::to_owned);
    match ty.as_deref() {
        None => {
            t.insert("type".into(), Value::from("function"));
            n += 1;
        }
        Some("function") => {}
        Some(_) => return (None, 1),
    }
    if !t.get("function").is_some_and(Value::is_object) {
        // Flat form: name/description/parameters/strict at the tool's top level.
        let Some(name) = t.get("name").cloned() else {
            return (None, n + 1);
        };
        let mut f = Map::new();
        f.insert("name".into(), name);
        for key in ["description", "parameters", "strict"] {
            if let Some(v) = t.remove(key) {
                f.insert(key.into(), v);
            }
        }
        t.remove("name");
        t.insert("function".into(), Value::Object(f));
        n += 1;
    }
    if let Some(Value::Object(f)) = t.get_mut("function") {
        if !f.get("name").is_some_and(Value::is_string) {
            return (None, n + 1);
        }
        match f.get("parameters") {
            Some(Value::Object(_)) => {}
            _ => {
                f.insert("parameters".into(), empty_schema());
                n += 1;
            }
        }
    }
    (Some(Value::Object(t)), n)
}

fn normalize_tools(obj: &mut Map<String, Value>) -> usize {
    let mut n = 0;
    match obj.remove("tools") {
        None => {}
        Some(Value::Array(tools)) => {
            let mut kept = Vec::with_capacity(tools.len());
            for tool in tools {
                let (t, k) = normalize_tool(tool);
                n += k;
                if let Some(t) = t {
                    kept.push(t);
                }
            }
            if kept.is_empty() {
                n += 1;
            } else {
                obj.insert("tools".into(), Value::Array(kept));
            }
        }
        Some(Value::Null) => n += 1,
        Some(other) => {
            // Leave unexpected shapes for the backend to reject.
            obj.insert("tools".into(), other);
        }
    }
    n
}

fn normalize_tool_choice(obj: &mut Map<String, Value>) -> usize {
    let has_tools = obj.get("tools").is_some_and(|t| t.as_array().is_some_and(|a| !a.is_empty()));
    if !obj.contains_key("tool_choice") {
        return 0;
    }
    if !has_tools {
        obj.remove("tool_choice");
        return 1;
    }
    let Some(choice) = obj.get_mut("tool_choice") else {
        return 0;
    };
    let replacement = match &*choice {
        Value::Object(c) => {
            let ty = c.get("type").and_then(Value::as_str);
            let nested = c.get("function").filter(|f| f.is_object()).cloned();
            match (ty, nested) {
                (Some(t @ ("auto" | "none" | "required")), None) => Some(Value::from(t)),
                (Some("any"), _) => Some(Value::from("required")),
                (Some("function"), Some(_)) => None,
                (Some("tool") | None, Some(f)) => {
                    Some(serde_json::json!({"type": "function", "function": f}))
                }
                (Some("function" | "tool") | None, None) => c
                    .get("name")
                    .cloned()
                    .map(|name| serde_json::json!({"type": "function", "function": {"name": name}})),
                _ => None,
            }
        }
        _ => None,
    };
    match replacement {
        Some(v) => {
            *choice = v;
            1
        }
        None => 0,
    }
}

fn normalize_stop(obj: &mut Map<String, Value>) -> usize {
    if let Some(Value::Array(stop)) = obj.get_mut("stop") {
        if stop.len() > MAX_STOP {
            stop.truncate(MAX_STOP);
            return 1;
        }
    }
    0
}

fn normalize_assistant_tool_calls(message: &mut Value) -> usize {
    if message.get("role").and_then(Value::as_str) != Some("assistant") {
        return 0;
    }
    let Some(Value::Array(calls)) = message.get_mut("tool_calls") else {
        return 0;
    };
    let mut n = 0;
    for call in calls.iter_mut() {
        let Value::Object(c) = call else { continue };
        if !c.contains_key("type") {
            c.insert("type".into(), Value::from("function"));
            n += 1;
        }
        if let Some(Value::Object(f)) = c.get_mut("function") {
            match f.get("arguments") {
                Some(Value::String(_)) => {}
                Some(Value::Null) | None => {
                    f.insert("arguments".into(), Value::from("{}"));
                    n += 1;
                }
                Some(v) => {
                    let s = v.to_string();
                    f.insert("arguments".into(), Value::from(s));
                    n += 1;
                }
            }
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn flat_tool_is_nested_and_params_filled() {
        let mut b = json!({"messages": [], "tools": [
            {"type": "function", "name": "a", "description": "d", "parameters": {"type": "object"}},
            {"type": "function", "function": {"name": "b"}},
            {"type": "web_search"},
            {"name": "c"}
        ]});
        assert!(normalize_chat_body(&mut b) > 0);
        assert_eq!(b["tools"].as_array().unwrap().len(), 3);
        assert_eq!(b["tools"][0], json!({"type": "function", "function": {"name": "a", "description": "d", "parameters": {"type": "object"}}}));
        assert_eq!(b["tools"][1]["function"]["parameters"], json!({"type": "object", "properties": {}}));
        assert_eq!(b["tools"][2]["function"]["name"], "c");
    }

    #[test]
    fn empty_tools_and_orphan_choice_removed() {
        let mut b = json!({"messages": [], "tools": [], "tool_choice": "auto"});
        normalize_chat_body(&mut b);
        assert!(b.get("tools").is_none());
        assert!(b.get("tool_choice").is_none());
        let mut b = json!({"messages": [], "tools": [{"type": "custom", "name": "x"}], "tool_choice": "required"});
        normalize_chat_body(&mut b);
        assert!(b.get("tools").is_none());
        assert!(b.get("tool_choice").is_none());
    }

    #[test]
    fn tool_choice_forms() {
        let tools = json!([{"type": "function", "function": {"name": "f", "parameters": {}}}]);
        let mut b = json!({"messages": [], "tools": tools, "tool_choice": {"type": "function", "name": "f"}});
        normalize_chat_body(&mut b);
        assert_eq!(b["tool_choice"], json!({"type": "function", "function": {"name": "f"}}));
        let mut b = json!({"messages": [], "tools": tools, "tool_choice": {"type": "any"}});
        normalize_chat_body(&mut b);
        assert_eq!(b["tool_choice"], "required");
        let mut b = json!({"messages": [], "tools": tools, "tool_choice": {"type": "auto"}});
        normalize_chat_body(&mut b);
        assert_eq!(b["tool_choice"], "auto");
        let mut b = json!({"messages": [], "tools": tools, "tool_choice": {"type": "function", "function": {"name": "f"}}});
        assert_eq!(normalize_chat_body(&mut b), 0);
    }

    #[test]
    fn stop_and_history() {
        let mut b = json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": {"x": 1}}}]}
        ], "stop": ["a", "b", "c", "d", "e"]});
        normalize_chat_body(&mut b);
        assert_eq!(b["stop"].as_array().unwrap().len(), 4);
        let c = &b["messages"][1]["tool_calls"][0];
        assert_eq!(c["type"], "function");
        assert_eq!(c["function"]["arguments"], "{\"x\":1}");
    }

    #[test]
    fn untouched_when_valid() {
        let mut b = json!({"messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            "tool_choice": "auto", "stop": ["x"]});
        let before = b.clone();
        assert_eq!(normalize_chat_body(&mut b), 0);
        assert_eq!(b, before);
    }
}
