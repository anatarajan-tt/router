/*
    Cache-Aware Load Balancing Router

    This router combines two strategies to optimize both cache utilization and request distribution:

    1. Cache-Aware Routing (Approximate Tree)
    2. Load Balancing (Shortest Queue with Balance Thresholds)

    The router dynamically switches between these strategies based on load conditions:
    - Uses load balancing when the system is imbalanced
    - Uses cache-aware routing when the system is balanced

    A system is considered imbalanced if both conditions are met:
    1. (max - min) > abs_threshold
    2. max > rel_threshold * min

    Strategy Details:

    1. Cache-Aware Routing (Approximate Tree)
    -------------------------------------------
    This strategy maintains an approximate radix tree for each worker based on request history,
    eliminating the need for direct cache state queries. The tree stores raw text characters
    instead of token IDs to avoid tokenization overhead.

    Process:
    a. For each request, find the worker with the highest prefix match
    b. If match rate > cache_threshold:
    Route to the worker with highest match (likely has relevant data cached)
    c. If match rate ≤ cache_threshold:
    Route to the worker with smallest tree size (most available cache capacity)
    d. Background maintenance:
    Periodically evict least recently used leaf nodes to prevent memory overflow

    2. Load Balancing (Shortest Queue)
    -------------------------------------------
    This strategy tracks pending request counts per worker and routes new requests
    to the least busy worker when the system is detected to be imbalanced.

    Configuration Parameters:
    ------------------------
    1. cache_threshold: (float, 0.0 to 1.0)
    Minimum prefix match ratio to use highest-match routing.
    Below this threshold, routes to worker with most available cache space.

    2. balance_abs_threshold: (integer)
    Absolute difference threshold for load imbalance detection.
    System is potentially imbalanced if (max_load - min_load) > abs_threshold

    3. balance_rel_threshold: (float)
    Relative ratio threshold for load imbalance detection.
    System is potentially imbalanced if max_load > min_load * rel_threshold
    Used in conjunction with abs_threshold to determine final imbalance state.

    4. eviction_interval_secs: (integer)
    Interval between LRU eviction cycles for the approximate trees.

    5. max_tree_size: (integer)
    Maximum nodes per tree. When exceeded, LRU leaf nodes are evicted
    during the next eviction cycle.

    6. prefill_aware: (optional, off by default)
    Replaces steps 1-2 with a latency score for engines whose prefill blocks
    every decode step of its data-parallel rank (see `PrefillAwareConfig`).
    Each worker's cost is the estimated time this request adds:

        uncached = prompt_tokens * (1 - prefix_match_rate on that worker)
        cost     = decoder_weight * decoders * (uncached / rate + fixed)
                 + pending / rate + uncached / rate

    `pending` comes from a per-worker FIFO of the uncached tokens the router
    sent there, drained at `rate`; entries beyond the worker's in-flight
    count are dropped (those requests already finished). `decoders` is the
    in-flight count minus the FIFO length. A long cached prefix keeps its
    worker unless the expected wait there costs more than recomputing the
    prefix elsewhere.
*/

use super::{get_healthy_worker_indices, CacheAwareConfig, LoadBalancingPolicy, RequestHeaders};
use crate::config::PrefillAwareConfig;
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::policies::normalize_model_key;
use crate::tree::Tree;
use dashmap::DashMap;
use rand::Rng;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// Backend candidate for a side-effect-free cache-affinity query.
#[derive(Debug, Clone, Copy)]
pub struct CacheAwareCandidate<'a> {
    pub target_id: &'a str,
    pub kv_pressure: f64,
}

/// Cache-affinity result committed separately by the caller.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheAwarePlacement {
    pub target_id: String,
    pub match_rate: f32,
}

/// Router-side estimate of one worker's outstanding prefill work.
///
/// The router cannot see an engine's prefill progress, so each routed
/// request is modelled as a FIFO entry that starts when the previous entry
/// finishes and drains its uncached tokens at the configured rate.
#[derive(Debug, Default)]
struct PrefillQueue {
    entries: VecDeque<PrefillEntry>,
}

#[derive(Debug, Clone, Copy)]
struct PrefillEntry {
    finish: Instant,
    tokens: f64,
}

impl PrefillQueue {
    /// Drop finished entries, then the oldest entries beyond `in_flight`
    /// (requests that completed or were cancelled before the model says so).
    fn refresh(&mut self, now: Instant, in_flight: usize) {
        while self.entries.front().is_some_and(|e| e.finish <= now) {
            self.entries.pop_front();
        }
        while self.entries.len() > in_flight {
            self.entries.pop_front();
        }
    }

    /// Estimated prompt tokens still to be prefilled on this worker.
    fn pending_tokens(&self, now: Instant, rate: f64) -> f64 {
        self.entries
            .iter()
            .map(|e| {
                let left = e.finish.saturating_duration_since(now).as_secs_f64() * rate;
                left.min(e.tokens)
            })
            .sum()
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn push(&mut self, now: Instant, tokens: f64, rate: f64, fixed: Duration) {
        let start = self.entries.back().map_or(now, |e| e.finish.max(now));
        let finish = start + fixed + Duration::from_secs_f64(tokens / rate);
        self.entries.push_back(PrefillEntry { finish, tokens });
    }
}

/// Estimated latency a request adds when sent to one worker (seconds).
fn prefill_aware_cost(
    pa: &PrefillAwareConfig,
    uncached_tokens: f64,
    pending_tokens: f64,
    decoders: usize,
) -> f64 {
    let rate = pa.prefill_rate_tokens_per_sec;
    let own = uncached_tokens / rate;
    let fixed = pa.prefill_fixed_cost_ms / 1000.0;
    pa.decoder_weight * decoders as f64 * (own + fixed) + pending_tokens / rate + own
}

/// Cache-aware routing policy
///
/// Routes requests based on cache affinity when load is balanced,
/// switches to shortest-queue routing when load is imbalanced.
/// Maintains separate trees per model for multi-model support.
/// With `prefill_aware` set, scores workers by estimated added latency.
#[derive(Debug)]
pub struct CacheAwarePolicy {
    config: CacheAwareConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>, // model_id -> Arc<Tree>
    eviction_handle: Option<thread::JoinHandle<()>>,
    /// worker url -> estimated outstanding prefill (prefill-aware mode only)
    prefill_queues: Mutex<HashMap<String, PrefillQueue>>,
}

impl CacheAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareConfig::default())
    }

    pub fn with_config(config: CacheAwareConfig) -> Self {
        let trees = Arc::new(DashMap::<String, Arc<Tree>>::new());

        // Start background eviction thread if configured
        let eviction_handle = if config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
            let max_tree_size = config.max_tree_size;
            let interval = config.eviction_interval_secs;

            Some(thread::spawn(move || loop {
                thread::sleep(Duration::from_secs(interval));

                // Evict for all model trees
                for entry in trees_clone.iter() {
                    let model_id = entry.key();
                    let tree = entry.value();
                    tree.evict_tenant_by_size(max_tree_size);
                    debug!(
                        "Cache eviction completed for model {}, max_size: {}",
                        model_id, max_tree_size
                    );
                }
            }))
        } else {
            None
        };

        Self {
            config,
            trees,
            eviction_handle,
            prefill_queues: Mutex::new(HashMap::new()),
        }
    }

    /// Select by the product of text-prefix affinity and current KV headroom.
    pub fn evaluate_initial_placement(
        &self,
        model_id: &str,
        request_text: &str,
        candidates: &[CacheAwareCandidate<'_>],
    ) -> Option<CacheAwarePlacement> {
        let tree = self.trees.get(normalize_model_key(model_id));
        let input_chars = request_text.chars().count();
        candidates
            .iter()
            .map(|candidate| {
                let matched_chars = tree.as_ref().map_or(0, |tree| {
                    tree.prefix_match_tenant_char_count(request_text, candidate.target_id)
                });
                let affinity = if input_chars == 0 {
                    0.0
                } else {
                    matched_chars as f64 / input_chars as f64
                };
                let headroom = 1.0 - candidate.kv_pressure.clamp(0.0, 1.0);
                (candidate, affinity, headroom, (1.0 + affinity) * headroom)
            })
            .max_by(|left, right| {
                left.3
                    .total_cmp(&right.3)
                    .then_with(|| left.2.total_cmp(&right.2))
                    .then_with(|| left.1.total_cmp(&right.1))
                    .then_with(|| right.0.target_id.cmp(left.0.target_id))
            })
            .map(|(candidate, affinity, _, _)| CacheAwarePlacement {
                target_id: candidate.target_id.to_string(),
                match_rate: affinity as f32,
            })
    }

    /// Update the affinity tree after the selected target is committed.
    pub fn commit_placement(&self, model_id: &str, request_text: &str, target_id: &str) {
        if request_text.is_empty() {
            return;
        }
        self.trees
            .entry(normalize_model_key(model_id).to_string())
            .or_insert_with(|| Arc::new(Tree::new()))
            .insert(request_text, target_id);
    }

    /// Remove one unavailable target from a model's affinity tree.
    pub fn remove_placement_target(&self, model_id: &str, target_id: &str) {
        if let Some(tree) = self.trees.get(normalize_model_key(model_id)) {
            tree.remove_tenant(target_id);
        }
    }

    /// Add a single worker to the tree (incremental update)
    pub fn add_worker(&self, worker: &dyn Worker) {
        let tree_key = normalize_model_key(worker.model_id());
        let tree = self
            .trees
            .entry(tree_key.to_string())
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", worker.url());
    }

    /// Add a worker by URL and model (for backward compatibility)
    pub fn add_worker_by_url(&self, url: &str, model_id: &str) {
        let tree = self
            .trees
            .entry(model_id.to_string())
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", url);
    }

    /// Remove a worker from the tree
    pub fn remove_worker(&self, worker: &dyn Worker) {
        let tree_key = normalize_model_key(worker.model_id());
        if let Some(tree) = self.trees.get(tree_key) {
            tree.remove_tenant(worker.url());
        }
        self.forget_prefill_queue(worker.url());
    }

    fn forget_prefill_queue(&self, url: &str) {
        if let Ok(mut queues) = self.prefill_queues.lock() {
            queues.remove(url);
        }
    }

    /// Remove a worker by URL (removes from all model trees for backward compatibility)
    pub fn remove_worker_by_url(&self, url: &str) {
        // Remove from all trees since we don't know which model it belongs to
        for tree_ref in self.trees.iter() {
            tree_ref.value().remove_tenant(url);
        }
        self.forget_prefill_queue(url);
    }

    /// Run cache eviction to prevent unbounded growth
    pub fn evict_cache(&self, max_size: usize) {
        for tree_ref in self.trees.iter() {
            let model_id = tree_ref.key();
            let tree = tree_ref.value();
            tree.evict_tenant_by_size(max_size);
            debug!(
                "Cache eviction for model {}, max_size: {}",
                model_id, max_size
            );
        }
    }

    /// Prefill-aware selection: the healthy worker with the lowest estimated
    /// added latency (see `prefill_aware_cost`), ties broken by load, then index.
    fn select_worker_prefill_aware(
        &self,
        pa: &PrefillAwareConfig,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        healthy_indices: &[usize],
        model_id: &str,
    ) -> Option<usize> {
        let text = request_text.unwrap_or("");
        let tree = self.trees.get(model_id).map(|entry| entry.value().clone());
        let input_chars = text.chars().count();
        let prompt_tokens = input_chars as f64 / pa.chars_per_token;

        let mut pool: Vec<usize> = if pa.max_load > 0 {
            healthy_indices
                .iter()
                .copied()
                .filter(|&idx| workers[idx].load() < pa.max_load)
                .collect()
        } else {
            Vec::new()
        };
        if pool.is_empty() {
            pool = healthy_indices.to_vec();
        }

        // Tree walks happen outside the queue lock.
        let uncached: Vec<f64> = pool
            .iter()
            .map(|&idx| {
                let matched = match (&tree, input_chars) {
                    (Some(tree), n) if n > 0 => {
                        tree.prefix_match_tenant_char_count(text, workers[idx].url())
                    }
                    _ => 0,
                };
                input_chars.saturating_sub(matched) as f64 / pa.chars_per_token
            })
            .collect();

        let rate = pa.prefill_rate_tokens_per_sec;
        let fixed = Duration::from_secs_f64(pa.prefill_fixed_cost_ms / 1000.0);
        let now = Instant::now();
        let selected_idx = {
            // Score and commit under one lock so concurrent arrivals see each other.
            let mut queues = self
                .prefill_queues
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut best: Option<(f64, usize, usize, f64)> = None;
            for (pos, &idx) in pool.iter().enumerate() {
                let load = workers[idx].load();
                let queue = queues.entry(workers[idx].url().to_string()).or_default();
                queue.refresh(now, load);
                let pending = queue.pending_tokens(now, rate);
                let decoders = load.saturating_sub(queue.len());
                let cost = prefill_aware_cost(pa, uncached[pos], pending, decoders);
                debug!(
                    "prefill_aware | worker={} load={} pending_tokens={:.0} decoders={} uncached_tokens={:.0} cost={:.3}s",
                    workers[idx].url(),
                    load,
                    pending,
                    decoders,
                    uncached[pos],
                    cost
                );
                let better = match best {
                    None => true,
                    Some((best_cost, best_load, best_idx, _)) => cost
                        .total_cmp(&best_cost)
                        .then(load.cmp(&best_load))
                        .then(idx.cmp(&best_idx))
                        .is_lt(),
                };
                if better {
                    best = Some((cost, load, idx, uncached[pos]));
                }
            }
            let (_, _, idx, tokens) = best?;
            queues
                .entry(workers[idx].url().to_string())
                .or_default()
                .push(now, tokens, rate, fixed);
            idx
        };

        debug!(
            "prefill_aware selected {} (prompt_tokens~{:.0})",
            workers[selected_idx].url(),
            prompt_tokens
        );
        if let Some(tree) = tree {
            if !text.is_empty() {
                tree.insert(text, workers[selected_idx].url());
            }
        }
        workers[selected_idx].increment_processed();
        RouterMetrics::record_processed_request(workers[selected_idx].url());
        RouterMetrics::record_policy_decision(self.name(), workers[selected_idx].url());
        Some(selected_idx)
    }

    fn select_worker_min_load(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        healthy_indices: &[usize],
        model_id: &str,
        max_load: usize,
        min_load: usize,
    ) -> Option<usize> {
        // Log load balancing trigger (only compute worker loads if debug enabled)
        if tracing::enabled!(tracing::Level::DEBUG) {
            let worker_loads: Vec<(&str, usize)> =
                workers.iter().map(|w| (w.url(), w.load())).collect();
            debug!(
                "Load balancing triggered | max: {} | min: {} | workers: {:?}",
                max_load, min_load, worker_loads
            );
        }

        RouterMetrics::record_load_balancing_event();
        RouterMetrics::set_load_range(max_load, min_load);

        // Use shortest queue when imbalanced
        let min_load_idx = healthy_indices
            .iter()
            .min_by_key(|&&idx| workers[idx].load())
            .copied()?;

        // Even in imbalanced mode, update the tree to maintain cache state
        if let Some(text) = request_text {
            // Get the tree reference without locking the entire HashMap
            // DashMap only locks the specific shard containing this key
            let tree = self.trees.get(model_id).map(|entry| entry.value().clone());

            if let Some(tree) = tree {
                let worker_url = workers[min_load_idx].url();
                tree.insert(text, worker_url);
            } else {
                debug!(
                    "Warning: No tree found for model '{}', skipping cache update",
                    model_id
                );
            }
        }

        // Increment processed counter
        workers[min_load_idx].increment_processed();
        RouterMetrics::record_processed_request(workers[min_load_idx].url());
        RouterMetrics::record_policy_decision(self.name(), workers[min_load_idx].url());

        Some(min_load_idx)
    }
}

impl LoadBalancingPolicy for CacheAwarePolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Determine the model for this set of workers (router pre-filters by model)
        // All workers should be from the same model
        let model_id = normalize_model_key(workers[healthy_indices[0]].model_id());

        // Base the load decision on workers that are eligible for selection.
        // Unhealthy workers must not affect whether healthy workers use cache
        // affinity or load balancing.
        let (min_load, max_load) = healthy_indices
            .iter()
            .map(|&idx| workers[idx].load())
            .fold((usize::MAX, 0usize), |(min, max), load| {
                (min.min(load), max.max(load))
            });
        let min_load = if min_load == usize::MAX { 0 } else { min_load };

        if let Some(pa) = &self.config.prefill_aware {
            return self.select_worker_prefill_aware(
                pa,
                workers,
                request_text,
                &healthy_indices,
                model_id,
            );
        }

        // Check if load is imbalanced
        let is_imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
            && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);

        debug!(
            "Load status for model: max_load={}, min_load={}, is_imbalanced={}",
            max_load, min_load, is_imbalanced
        );

        if is_imbalanced {
            return self.select_worker_min_load(
                workers,
                request_text,
                &healthy_indices,
                model_id,
                max_load,
                min_load,
            );
        }

        // Use cache-aware routing when balanced
        let text = request_text.unwrap_or("");

        // Get the tree reference without locking the entire HashMap
        // DashMap only locks the specific shard containing this key
        let tree = self.trees.get(model_id).map(|entry| entry.value().clone());

        let keys: Vec<_> = self.trees.iter().map(|entry| entry.key().clone()).collect();
        debug!("Available tree keys: {:?}", keys);

        let Some(tree) = tree else {
            // No tree for this model, log warning and use random selection
            debug!(
                "Warning: No tree found for model '{}', using random worker selection",
                model_id
            );
            // Return a random healthy worker
            let mut rng = rand::rng();
            let random_idx = rng.random_range(0..healthy_indices.len());
            let selected_idx = healthy_indices[random_idx];

            workers[selected_idx].increment_processed();
            RouterMetrics::record_processed_request(workers[selected_idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[selected_idx].url());

            return Some(selected_idx);
        };
        debug!("Using cache-aware routing for model '{}'", model_id);
        // Now we work with the tree without holding the HashMap lock
        // Use prefix_match_with_counts to avoid redundant chars().count() calls
        let result = tree.prefix_match_with_counts(text);
        let match_rate = if result.input_char_count == 0 {
            0.0
        } else {
            result.matched_char_count as f32 / result.input_char_count as f32
        };

        debug!(
            "Cache match for model '{}': matched_chars={}, input_chars={}, match_rate={:.2}",
            model_id, result.matched_char_count, result.input_char_count, match_rate
        );
        // Select worker without String allocation
        let selected_idx = if match_rate > self.config.cache_threshold {
            // Cache hit path: find worker by URL (compare &str directly, no allocation)
            let tenant_url: &str = &result.tenant;
            workers
                .iter()
                .position(|w| w.url() == tenant_url)
                .filter(|&idx| workers[idx].is_healthy())
        } else {
            // Low cache match: use worker with minimum load
            healthy_indices
                .iter()
                .min_by_key(|&&idx| workers[idx].load())
                .copied()
        };

        if let Some(idx) = selected_idx {
            // Update the tree with this request (use worker URL directly, no allocation)
            tree.insert(text, workers[idx].url());

            // Increment processed counter
            workers[idx].increment_processed();
            RouterMetrics::record_processed_request(workers[idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[idx].url());

            return Some(idx);
        }

        // Selected worker no longer exists or unhealthy, remove stale tenant from tree
        if match_rate > self.config.cache_threshold {
            let tenant_url: &str = &result.tenant;
            tree.remove_tenant(tenant_url);
            debug!("Removed stale worker {} from cache tree", tenant_url);
        }

        // Fallback to first healthy worker
        if let Some(idx) = healthy_indices.first().copied() {
            workers[idx].increment_processed();
            RouterMetrics::record_processed_request(workers[idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[idx].url());

            Some(idx)
        } else {
            None
        }
    }

    fn name(&self) -> &'static str {
        "cache_aware"
    }

    fn needs_request_text(&self) -> bool {
        true // Cache-aware policy needs request text for cache affinity
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        // Could track success rates per worker for more intelligent routing
        if !success {
            // Optionally reduce affinity for failed requests
            tracing::debug!(
                "Request to {} completed with success={}",
                worker_url,
                success
            );
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn select_worker_pair_with_headers(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<(usize, usize)> {
        // DEPRECATED: This method is no longer used when separate policies are configured.
        // The PD router now uses separate policies for prefill and decode selection.
        // This implementation remains for backward compatibility when a single policy is used.

        // In PD mode with single policy:
        // - Prefill: Use cache-aware routing for better cache utilization
        // - Decode: Use least-load routing for better load distribution

        // Select prefill worker using cache-aware logic
        let prefill_idx =
            self.select_worker_with_headers(prefill_workers, request_text, headers)?;

        // Select decode worker using least-load logic
        let healthy_decode = get_healthy_worker_indices(decode_workers);
        if healthy_decode.is_empty() {
            return None;
        }

        let decode_idx = healthy_decode
            .iter()
            .min_by_key(|&&idx| decode_workers[idx].load())
            .copied()?;

        Some((prefill_idx, decode_idx))
    }

    fn requires_initialization(&self) -> bool {
        true // Cache-aware policy requires init_workers() to set up trees
    }

    fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        // Group workers by model
        info!(
            "Initializing workers for cache-aware policy: {}",
            workers
                .iter()
                .map(|w| w.url())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let mut model_workers: HashMap<String, Vec<&Arc<dyn Worker>>> = HashMap::new();
        for worker in workers {
            let tree_key = normalize_model_key(worker.model_id());
            model_workers
                .entry(tree_key.to_string())
                .or_default()
                .push(worker);
        }

        // Initialize tree for each model
        for (tree_key, model_workers) in model_workers {
            info!(
                "Creating tree for model key: '{}' with {} workers",
                tree_key,
                model_workers.len()
            );
            let tree = self
                .trees
                .entry(tree_key)
                .or_insert_with(|| Arc::new(Tree::new()))
                .clone();
            for worker in model_workers {
                tree.insert("", worker.url());
            }
        }
    }
}

impl Default for CacheAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CacheAwarePolicy {
    fn drop(&mut self) {
        // Note: We can't properly stop the eviction thread since it's in an infinite loop
        // In a production system, we'd use a channel or atomic flag to signal shutdown
        if let Some(handle) = self.eviction_handle.take() {
            // The thread will continue running until the program exits
            // This is acceptable for now since the router typically runs for the lifetime of the program
            drop(handle);
        }
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    #[test]
    fn unhealthy_worker_load_does_not_disable_cache_affinity() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 0,
            balance_rel_threshold: 1.0,
            eviction_interval_secs: 0,
            max_tree_size: 100,
            prefill_aware: None,
        });
        let workers: Vec<Arc<dyn Worker>> = (0..3)
            .map(|index| {
                Arc::new(BasicWorker::new(
                    format!("http://worker-{index}:8000"),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect();
        workers[0].set_healthy(false);
        workers[2].set_healthy(false);
        policy.init_workers(&workers);
        for _ in 0..5 {
            workers[0].increment_load();
            workers[1].increment_load();
        }

        // Establish affinity while worker 1 is the only healthy worker.
        assert_eq!(policy.select_worker(&workers, Some("sticky")), Some(1));
        workers[0].set_healthy(true);

        // Both eligible workers have equal load, so the unhealthy zero-load
        // worker must not switch the policy into load-balancing mode.
        assert_eq!(policy.select_worker(&workers, Some("sticky")), Some(1));
    }

    #[test]
    fn test_cache_aware_with_balanced_load() {
        // Create policy without eviction thread for testing
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        // Initialize the policy with workers
        policy.init_workers(&workers);

        // First request should be distributed
        let idx1 = policy.select_worker(&workers, Some("hello world")).unwrap();

        // Same request should go to same worker (cache hit)
        let idx2 = policy.select_worker(&workers, Some("hello world")).unwrap();
        assert_eq!(idx1, idx2);

        // Similar request should also go to same worker
        let idx3 = policy.select_worker(&workers, Some("hello")).unwrap();
        assert_eq!(idx1, idx3);
    }

    #[test]
    fn literal_token_ids_prefix_keeps_cache_affinity() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];
        policy.init_workers(&workers);
        let key = "token_ids: explain this field";
        let model_id = normalize_model_key(workers[0].model_id());
        policy
            .trees
            .get(model_id)
            .expect("initialized model cache")
            .insert(key, workers[1].url());

        assert_eq!(policy.select_worker(&workers, Some(key)), Some(1));
    }

    #[test]
    fn test_cache_aware_with_imbalanced_load() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0, // Disable eviction thread
            max_tree_size: 10000,
            prefill_aware: None,
        });

        let worker1 = BasicWorker::new("http://w1:8000".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://w2:8000".to_string(), WorkerType::Regular);

        // Create significant load imbalance
        for _ in 0..20 {
            worker1.increment_load();
        }
        // worker2 has load 0

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker1), Arc::new(worker2)];
        policy.init_workers(&workers);

        // Should select worker2 (lower load) despite cache affinity
        for _ in 0..5 {
            let idx = policy.select_worker(&workers, Some("test")).unwrap();
            assert_eq!(idx, 1); // Should always pick worker2
        }
    }

    #[test]
    fn test_cache_aware_worker_removal() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        policy.init_workers(&workers);

        // Route some requests
        policy.select_worker(&workers, Some("test1"));
        policy.select_worker(&workers, Some("test2"));

        // Remove a worker
        policy.remove_worker_by_url("http://w1:8000");
        workers[0].set_healthy(false);

        // All requests should now go to worker2
        let idx = policy.select_worker(&workers, Some("test1")).unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn initial_placement_combines_affinity_and_headroom_without_mutation() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let candidates = [
            CacheAwareCandidate {
                target_id: "rank-0",
                kv_pressure: 0.2,
            },
            CacheAwareCandidate {
                target_id: "rank-1",
                kv_pressure: 0.1,
            },
        ];
        assert_eq!(
            policy
                .evaluate_initial_placement("model", "shared suffix", &candidates)
                .unwrap()
                .target_id,
            "rank-1"
        );
        policy.commit_placement("model", "shared first", "rank-0");
        let equal_pressure = [
            CacheAwareCandidate {
                target_id: "rank-0",
                kv_pressure: 0.1,
            },
            CacheAwareCandidate {
                target_id: "rank-1",
                kv_pressure: 0.1,
            },
        ];
        assert_eq!(
            policy
                .evaluate_initial_placement("model", "shared second", &equal_pressure)
                .unwrap()
                .target_id,
            "rank-0"
        );
    }

    fn prefill_aware_policy(pa: PrefillAwareConfig) -> CacheAwarePolicy {
        CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            prefill_aware: Some(pa),
            ..Default::default()
        })
    }

    /// One token per character keeps the arithmetic readable.
    fn unit_pa() -> PrefillAwareConfig {
        PrefillAwareConfig {
            prefill_rate_tokens_per_sec: 8000.0,
            prefill_fixed_cost_ms: 80.0,
            decoder_weight: 1.0,
            chars_per_token: 1.0,
            max_load: 0,
        }
    }

    fn two_workers() -> Vec<Arc<dyn Worker>> {
        vec![
            Arc::new(BasicWorker::new(
                "http://w0:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
        ]
    }

    /// Route like the HTTP router does: select, then count the request in flight.
    fn route(policy: &CacheAwarePolicy, workers: &[Arc<dyn Worker>], text: &str) -> usize {
        let idx = policy.select_worker(workers, Some(text)).unwrap();
        workers[idx].increment_load();
        idx
    }

    fn set_pending(policy: &CacheAwarePolicy, url: &str, tokens: f64) {
        let mut queues = policy.prefill_queues.lock().unwrap();
        let queue = queues.entry(url.to_string()).or_default();
        queue.entries.clear();
        if tokens > 0.0 {
            queue.push(Instant::now(), tokens, 8000.0, Duration::ZERO);
        }
    }

    #[test]
    fn prefill_queue_is_fifo_and_drains_at_rate() {
        let t0 = Instant::now();
        let mut q = PrefillQueue::default();
        q.push(t0, 8000.0, 8000.0, Duration::from_millis(0));
        q.push(t0, 4000.0, 8000.0, Duration::from_millis(0));
        // second entry starts when the first finishes (t0 + 1 s) and ends at t0 + 1.5 s
        assert!((q.pending_tokens(t0, 8000.0) - 12000.0).abs() < 1.0);
        let t_half = t0 + Duration::from_millis(500);
        assert!((q.pending_tokens(t_half, 8000.0) - 8000.0).abs() < 1.0);
        let t_1_25 = t0 + Duration::from_millis(1250);
        q.refresh(t_1_25, 10);
        assert_eq!(q.len(), 1);
        assert!((q.pending_tokens(t_1_25, 8000.0) - 2000.0).abs() < 1.0);
        q.refresh(t0 + Duration::from_secs(2), 10);
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn prefill_queue_drops_entries_beyond_in_flight() {
        let t0 = Instant::now();
        let mut q = PrefillQueue::default();
        for _ in 0..3 {
            q.push(t0, 8000.0, 8000.0, Duration::ZERO);
        }
        q.refresh(t0, 1);
        assert_eq!(q.len(), 1);
        // the newest entry survives and keeps its FIFO finish time (t0 + 3 s)
        assert!((q.pending_tokens(t0 + Duration::from_secs(2), 8000.0) - 8000.0).abs() < 1.0);
        q.refresh(t0, 0);
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn prefill_aware_cost_formula() {
        let pa = unit_pa();
        // 3 decoders x (1 s own + 0.08 s fixed) + 0.5 s pending + 1 s own
        let cost = prefill_aware_cost(&pa, 8000.0, 4000.0, 3);
        assert!((cost - (3.0 * 1.08 + 0.5 + 1.0)).abs() < 1e-9);
        let no_decoders = PrefillAwareConfig {
            decoder_weight: 0.0,
            ..pa
        };
        assert!((prefill_aware_cost(&no_decoders, 8000.0, 4000.0, 3) - 1.5).abs() < 1e-9);
    }

    #[test]
    fn prefill_aware_avoids_worker_that_is_prefilling() {
        let policy = prefill_aware_policy(unit_pa());
        let workers = two_workers();
        policy.init_workers(&workers);

        let long = "a".repeat(80_000); // ~10 s of prefill at 8000 tok/s
        assert_eq!(route(&policy, &workers, &long), 0);
        // A small request lands on the idle worker, not behind the 80K prefill.
        assert_eq!(route(&policy, &workers, "hello"), 1);
    }

    #[test]
    fn prefill_aware_avoids_stalling_running_decoders() {
        let policy = prefill_aware_policy(unit_pa());
        let workers = two_workers();
        policy.init_workers(&workers);
        // w0 runs three decoding requests (in flight, nothing pending).
        for _ in 0..3 {
            workers[0].increment_load();
        }
        let prompt = "b".repeat(8_000);
        assert_eq!(route(&policy, &workers, &prompt), 1);
    }

    #[test]
    fn prefill_aware_keeps_cache_affinity_until_the_wait_outweighs_it() {
        let policy = prefill_aware_policy(unit_pa());
        let workers = two_workers();
        policy.init_workers(&workers);

        let prefix = "c".repeat(40_000); // 5 s to recompute elsewhere
        assert_eq!(route(&policy, &workers, &prefix), 0);
        workers[0].decrement_load();
        set_pending(&policy, "http://w0:8000", 0.0);

        let turn = format!("{prefix} next turn");
        // Idle and cached: stay.
        assert_eq!(policy.select_worker(&workers, Some(&turn)), Some(0));

        // w0 has 1 s of pending prefill: still cheaper than 5 s of recompute.
        workers[0].increment_load();
        set_pending(&policy, "http://w0:8000", 8_000.0);
        assert_eq!(policy.select_worker(&workers, Some(&turn)), Some(0));

        // w0 has 10 s of pending prefill: recomputing on w1 is faster.
        set_pending(&policy, "http://w0:8000", 80_000.0);
        assert_eq!(policy.select_worker(&workers, Some(&turn)), Some(1));
    }

    #[test]
    fn prefill_aware_max_load_skips_full_workers() {
        let policy = prefill_aware_policy(PrefillAwareConfig {
            max_load: 2,
            ..unit_pa()
        });
        let workers = two_workers();
        policy.init_workers(&workers);
        let prefix = "d".repeat(40_000);
        assert_eq!(route(&policy, &workers, &prefix), 0);
        set_pending(&policy, "http://w0:8000", 0.0);
        workers[0].increment_load(); // w0 now at max_load
        let turn = format!("{prefix} more");
        assert_eq!(policy.select_worker(&workers, Some(&turn)), Some(1));
        // When every worker is full the filter is ignored.
        workers[1].increment_load();
        workers[1].increment_load();
        assert_eq!(policy.select_worker(&workers, Some(&turn)), Some(0));
    }

    #[test]
    fn prefill_aware_ignores_imbalance_thresholds() {
        // Classic cache_aware would switch to least-load here (abs 0, rel 1.0);
        // prefill-aware scores instead and keeps the cached worker.
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 0,
            balance_rel_threshold: 1.0,
            eviction_interval_secs: 0,
            max_tree_size: 100,
            prefill_aware: Some(unit_pa()),
        });
        let workers = two_workers();
        policy.init_workers(&workers);
        let prefix = "e".repeat(40_000);
        assert_eq!(route(&policy, &workers, &prefix), 0);
        workers[0].decrement_load();
        set_pending(&policy, "http://w0:8000", 0.0);
        workers[0].increment_load(); // one decoder on w0, w1 idle
        let turn = format!("{prefix} again");
        assert_eq!(policy.select_worker(&workers, Some(&turn)), Some(0));
    }

    #[test]
    fn removing_a_worker_forgets_its_prefill_queue() {
        let policy = prefill_aware_policy(unit_pa());
        let workers = two_workers();
        policy.init_workers(&workers);
        route(&policy, &workers, &"f".repeat(1000));
        assert!(policy
            .prefill_queues
            .lock()
            .unwrap()
            .contains_key("http://w0:8000"));
        policy.remove_worker(workers[0].as_ref());
        assert!(!policy
            .prefill_queues
            .lock()
            .unwrap()
            .contains_key("http://w0:8000"));
    }
}
