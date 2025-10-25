# Claude Stage Implementation Roadmap

This document consolidates every requested feature, experiment, and deliverable into a single staged execution plan. Each stage corresponds to a major prompt, summarising objectives, required components, integration points, and expected outputs. Use this as the authoritative backlog while implementing the new H-SOKV capabilities.

---

## Stage 1 — Memory Consolidation Module (Prompt 1)
- **Goal:** Transfer high-confidence episodic memories into transformer weights to mimic human sleep consolidation.
- **Deliverables:**
  - New file `hsokv_core/consolidation.py` housing a `ConsolidationModule` with:
    - `identify_candidates()` filtering KV entries where `confidence > 0.85`, `retrieval_count > 10`, `success_rate > 0.7`.
    - `create_consolidation_dataset()` generating synthetic `(context → definition)` pairs from candidate memories.
    - `consolidate()` to fine-tune the model for 50 steps at learning rate `1e-5`, remove consolidated memories, and report `consolidated_count`, `memory_freed`, `avg_loss`.
  - Verbose logging, error handling, and device-aware tensor management.
- **Integration:** Invoke consolidation every 5 iterations inside the swarm training loop (e.g., `hsokv_core/training.py` or `swarm.Supervisor`). Ensure consolidated memories are pruned from KV storage immediately after weight updates.

---

## Stage 2 — Context-Aware Retrieval (Prompt 2)
- **Goal:** Modulate KV retrieval using recency, domain, emotion, and importance signals.
- **Deliverables:**
  - New file `hsokv_core/context_retrieval.py` defining `ContextualRetrievalModule`.
    - `extract_context_signals()` to infer query domain/emotion from transformer hidden states.
    - `compute_context_modulated_similarity()` applying multiplicative boosts with documented factors:
      - Recency: `0.95^( (current_step - created_at) / 100 )`.
      - Domain: `1.5` multiplier when domains match.
      - Emotion: `1 + 0.3 * (1 - |memory_emotion - query_emotion|)`.
      - Importance: `1 + 0.5 * success_rate`.
    - `contextual_retrieve()` returning top-k memories using modulated scores.
  - Metadata extensions for `domain` and `emotion` (default fallbacks when absent).
- **Integration:** Replace the cosine-only retrieval in `KeyValueMemory.retrieve()` (or wrap it) so `TransformerWithKV` calls the contextual version seamlessly.

---

## Stage 3 — Surprise-Based Selective Writing (Prompt 3)
- **Goal:** Write memories only when predictions fail or context is novel, reducing redundancy by ~60%.
- **Deliverables:**
  - New file `hsokv_core/surprise_writing.py` with `SurpriseBasedWriter`.
    - `compute_prediction_error()` per-sample cross-entropy (surprise).
    - `compute_novelty()` measuring `1 - max_cosine_similarity` against existing keys.
    - `should_write()` returning boolean mask using criteria: `(surprise > 0.5) OR (novelty > 0.7)`.
    - `selective_write()` performing filtered writes and tracking `surprise_scores`, `novelty_scores`, `write_count`, `skip_count`.
  - Initial confidence set to `1.0 - surprise`.
- **Integration:** Modify `Agent._update_memory()` (in `hsokv_core/swarm.py`) to use the writer post-optimizer step instead of unconditional writes.

---

## Stage 4 — Automatic Forgetting Mechanism (Prompt 4)
- **Status:** Implemented in current repo snapshot.
- **Goal:** Prune low-utility or interfering memories to keep KV size within 200–300 entries.
- **Deliverables:**
  - New file `hsokv_core/forgetting.py` containing `ForgettingModule`.
    - `compute_memory_utility()` with score `0.3*confidence + 0.3*success_rate + 0.2*recency + 0.2*log_frequency`.
    - `identify_interfering_memories()` flagging pairs where cosine similarity > 0.8 and resolving by confidence.
    - `should_forget()` determining when to trigger (every 10 iterations or when >80% full).
    - `forget()` removing entries, outputting `forgotten_count`, `memory_size_after`.
  - Visualization routine plotting utility distributions before/after forgetting (saved under `results`).
- **Integration:** Hook into training loop so forgetting runs on schedule or under pressure.

---

## Stage 5 — Human Comparison Experiments (Prompt 5)
- **Status:** Implemented in current repo snapshot.
- **Goal:** Benchmark H-SOKV learning curves against human and model baselines.
- **Deliverables:**
  - New script `experiments/human_comparison.py` implementing:
    - `collect_human_baseline()` (simulated or provided data).
    - `run_model_baseline()` covering BERT, GPT-3 few-shot (stub or API), MAML, and H-SOKV across shot counts {1,2,5,10,50,100}.
    - `run_learning_curve_experiment()` executing 5 seeds per condition.
    - `plot_comparison()` (accuracy vs. shots with error bars) saved to `results/human_comparison_curves.png`.
    - `statistical_test()` writing significance outcomes to `results/statistical_tests.txt`.
- **Notes:** Use `RARE_WORD_SPECS` from `hsokv_core/data.py` to build datasets of 20 words; 100-test split per condition.

---

## Stage 6 — Memory Visualization Tools (Prompt 6)
- **Status:** Implemented in current repo snapshot (core plots complete; advanced t-SNE/interactive views deferred).
- **Goal:** Provide interpretability tooling for KV memory contents and lifecycle.
- **Deliverables:**
  - New module `hsokv_core/visualization.py` providing:
    - `plot_consolidation_timeline()` for retention / entropy / regret trends.
    - `plot_memory_statistics()` with histogram + scatter diagnostics.
    - `generate_report()` bundling assets and basic JSON stats in `results/visualizations/`.
- **Notes:** t-SNE / interactive dashboards remain future enhancements.

---

## Stage 7 — Ablation Study Automation (Prompt 7)
- **Status:** Implemented in current repo snapshot.
- **Goal:** Systematically evaluate contribution of each new component.
- **Deliverables:**
  - Script `experiments/comprehensive_ablations.py` covering:
    1. Full H-SOKV.
    2. Minus consolidation.
    3. Minus context-aware retrieval.
    4. Minus surprise-based writing.
    5. Minus forgetting.
    6. Minus swarm optimization.
    7. Minus KV memory.
  - Output helpers for statistics (mean/std) and Markdown/JSON reports, plus delta summaries.
- **Notes:** Visual charts can be added later; default seeds=3 configurable via CLI.

---

## Stage 8 — Scaling Experiments (Prompt 8)
- **Status:** Implemented in current repo snapshot.
- **Goal:** Profile retrieval latency, memory footprint, and accuracy from 1K to 100K memories.
- **Deliverables:**
  - Script `experiments/scaling_study.py` with:
    - `generate_large_dataset()` helper to expand synthetic corpus.
    - `run_scaling_experiment()` for configurable memory sizes (defaults 1K–5K, accepts higher via CLI).
    - Metrics/report writers plus matplotlib curves for latency & footprint.
    - `recommend_optimizations()` emitting scaling guidance.
- **Notes:** Extremely large regimes (50K–100K) may require further optimisation/ANN backends.

---

## Stage 9 — Cross-Domain Continual Learning (Prompt 9)
- **Status:** Implemented in current repo snapshot.
- **Goal:** Assess catastrophic forgetting across medical, legal, finance, technology, and culinary domains.
- **Deliverables:**
  - Script `experiments/continual_learning.py` with:
    - `generate_domain_datasets()` constructing 5×100-word corpora.
    - `run_continual_learning()` sequentially training and logging accuracy after each domain.
    - `measure_transfer()` calculating backward and forward transfer.
    - `plot_accuracy_matrix()` heatmap saved to `results/continual_learning_matrix.png`.
    - `compare_to_baselines()` writing simple delta metrics to `results/transfer_metrics.txt`.

---

## Stage 10 — Integration & Testing Suite (Prompt 10)
- **Status:** Implemented in current repo snapshot.
- **Goal:** Ensure reliability of all modules via pytest-based integration coverage.
- **Deliverables:**
  - Test file `tests/test_integration.py` including fixtures (sample model, dataset, temp checkpoint) and tests:
    - `test_consolidation_forgetting_pipeline()`
    - `test_context_aware_surprise_writing()`
    - `test_full_training_pipeline()`
    - `test_edge_cases()` (empty/full memory, all consolidated, single-shot).
    - `test_device_compatibility()` across CPU/GPU combinations.
    - `test_memory_lifecycle()`
    - `test_performance_assertions()` checking accuracy gains, memory bounds, context boosts.
- **Command:** `pytest tests/test_integration.py -v`

---

## Stage 11 — Paper-Ready Experiment Suite (Prompt 11)
- **Status:** Implemented in current repo snapshot.
- **Goal:** Produce publication-grade figures, tables, and statistical analyses for an academic submission.
- **Deliverables:**
  - Master script `experiments/paper_experiments.py` implementing:
    - `run_all_paper_experiments()` orchestrator.
    - `experiment_1_one_shot_learning()`
    - `experiment_2_continual_learning()`
    - `experiment_3_consolidation_ablation()`
    - `experiment_4_interpretability()`
    - `generate_ablation_table()`, `compute_statistical_significance()`, `generate_latex_tables()`, `generate_publication_figures()`.
  - Output directory `results/paper/` with:
    - Figures (`fig1_learning_curves.pdf`, `fig2_continual_matrix.pdf`, `fig3_memory_growth.pdf`, `fig4_memory_tsne.pdf`).
    - Tables (`table1_one_shot.tex`, `table2_continual.tex`, `table3_ablations.tex`).
    - Statistics (`stats/significance_tests.txt`).
- **Formatting:** Vector graphics (>=300 DPI), NeurIPS/Nature-ready styling.
- **Usage:** `python experiments/paper_experiments.py --output results/paper/`

---

## Stage 12 — Production Deployment Module (Prompt 12)
- **Status:** Pending — not yet implemented.
- **Goal:** Deliver a production-grade inference API with monitoring, management, and deployment support.
- **Deliverables:**
  - Module `hsokv_core/production_api.py` featuring `HSOKV_API` with methods:
    - `from_pretrained()`
    - `learn_concept()`
    - `query()` (returns prediction, confidence, retrieved memories, explanation).
    - `provide_feedback()`
    - `consolidate_memory()` (background task triggered on thresholds).
    - `export_memory()` / `import_memory()`
    - `get_metrics()` (latency, accuracy, memory size, etc.).
    - Rate limiting, input sanitisation, memory caps, caching, batch inference, FAISS integration for large stores.
  - REST interface `api_server.py` (FastAPI) exposing `/learn`, `/query`, `/feedback`, `/metrics`, `/memory/{id}`.
  - `docker-compose.yml` provisioning the service with necessary dependencies.
- **Operational Requirements:** <100 ms/query at 10K memories, auto-consolidation/forgetting when thresholds exceeded, structured logging for monitoring.

---

## Bonus Stage — Provisional Patent Draft (Bonus Prompt)
- **Status:** Draft prepared in current repo snapshot.
- **Goal:** Document the inventive aspects of H-SOKV for legal review.
- **Deliverables:**
  - Markdown draft `patent/provisional_application.md` containing:
    - Title, inventor info, abstract (~150 words), background, summary, detailed description (5–10 pages equivalent), claims (broad + dependent), drawing descriptions, prior art comparison table.
    - Explicit claims covering consolidation, context retrieval, surprise-based writing, forgetting, and overall system composition.
  - Figures under `patent/figures/`:
    - `system_architecture.pdf`
    - `consolidation_flowchart.pdf`
    - `retrieval_algorithm.pdf`
- **Guidance:** Format per USPTO provisional standards; highlight novelty over Memory Networks, Neural Turing Machines, MAML, etc.

---

## Implementation Notes & Dependencies
- **Metadata enhancements:** Ensure KV metadata stores `domain`, `emotion`, `created_at`, frequency counters, utility scores, and lifecycle states (`active`, `consolidated`, `forgotten`).
- **Device handling:** Maintain compatibility across CPU/GPU, including when memory tensors reside on different devices.
- **Logging:** Adopt consistent logging (e.g., `logging` module) for consolidation, retrieval, writing, forgetting, and production API events.
- **Results directory:** Create subfolders as needed (`results/visualizations`, `results/human_comparison`, `results/paper`, etc.) with graceful fallbacks if missing.
- **Testing cadence:** Expand `run_validation_tests()` or CI scripts to invoke the new pytest suite alongside existing checks.
- **Documentation:** Update `README.md` and internal docs (e.g., `docs/`) after each stage with usage instructions and experiment summaries.

---

## Suggested Execution Order
1. **Core Memory Mechanics:** Stages 1–4 to stabilise consolidation, retrieval, writing, and forgetting.
2. **Visualization & Experiments:** Stages 5–9 to validate behaviour and performance.
3. **Testing & Publication Assets:** Stages 10–11 to ensure reliability and produce shareable artifacts.
4. **Deployment & Legal Preparation:** Stage 12 and Bonus Stage for production readiness and intellectual property documentation.

Track progress by updating this file (or a project management tool) as stages are completed. Each stage can be parallelised where dependencies allow, but the suggested order minimises integration risk.
