# H-SOKV: Hierarchical Swarm-KV Research Prototype

H-SOKV couples a lightweight transformer backbone with an adaptive key–value memory and a three-tier swarm optimizer (agents → managers → supervisor). The goal is to demonstrate human-like one-shot vocabulary learning, few-shot generalisation on GLUE, continual retention on Split-CIFAR, and scalable exploration via distributed swarms—all within a reproducible PyTorch codebase that runs end-to-end on modest hardware (Colab T4).

---

## 1. Repository Layout

```
hsokv.py                  # CLI entrypoint and experiment orchestration
hsokv_core/
  ablations.py           # Prompt-1 suite plus automation hooks
  benchmarks.py          # Few-shot GLUE + Split-CIFAR loaders/evaluators
  config.py              # Global CONFIG dict and override helpers
  consolidation.py       # Stage-1 consolidation module
  context_retrieval.py   # Stage-2 context-aware retrieval
  data.py                # Synthetic rare-word dataset + tokenizer utilities
  distributed.py         # Toy CartPole swarm simulator (Ray/mp/simulate)
  forgetting.py          # Stage-4 automatic forgetting utilities
  hf_adapter.py          # Hugging Face-style trainer wrapper (HFSwarmTrainer)
  memory.py              # Normalised KeyValueMemory implementation
  model.py               # TransformerWithKV + BaselineTransformer
  metrics.py             # FLOP estimation and reporting helpers
  surprise_writing.py    # Stage-3 surprise-based selective writer
  swarm.py               # Agent / Manager / Supervisor hierarchy
  training.py            # Core training loops and baselines
  visualization.py       # Stage-6 timeline/statistics reporting
  utils.py               # Device helpers, seeding, diversity metrics
experiments/              # Stage-5–11 experiment pipelines (human, scaling, continual, paper, etc.)
results/                  # Plots, tables, and benchmark logs (created on demand)
docs/                     # Colab walkthrough and supplementary notes
tests/                    # Pytest integration suite (Stage 10)
patent/                   # Provisional draft + placeholder figures (bonus stage)
```

---

## 2. Environment & Setup

### 2.1 Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q matplotlib numpy tqdm scikit-learn
```

> **Tip:** On Google Colab just run the two `pip install` commands above.

### 2.2 Quick Smoke Tests

```bash
# Synthetic training with plots (results/learning_curves.png, retention.png, ...)
python hsokv.py --iterations 10 --visualize

# Validation suite (sanity checks for KV, swarm, benchmarks, distributed sim)
python hsokv.py --mode test

# Pytest integration checks (consolidation, forgetting, surprise writing, e2e)
pytest tests/test_integration.py -k full_training_pipeline
```

---

## Quick Start

### 1. Fast Test (2 minutes)

```bash
python hsokv.py --preset quick_test --visualize
```

### 2. Demo Run (15 minutes) - RECOMMENDED

```bash
python hsokv.py --preset demo --task language_model --visualize
```

### 3. Full Research (30+ minutes)

```bash
python hsokv.py --preset research --visualize
```

### Custom Corpus

```bash
python hsokv.py --task language_model --lm-corpus your_corpus.txt --visualize
```

---

## 3. Running Experiments

### 3.1 Synthetic Rare-Word Task

- **Default:** `python hsokv.py --iterations 10`
- **Disable swarm/KV:** `--no-use-swarm`, `--no-use-kv`
- **Ablation suite:** `python hsokv.py --run-ablations`

### 3.2 Few-Shot GLUE (SST-2 / MNLI)

```bash
python hsokv.py --benchmark glue --glue-task sst2 \
    --allow-download --iterations 5 --agents-per-manager 3 \
    --agent-steps 25 --num-managers 2 --visualize
```

- First run (`--allow-download`) fetches SST-2 via [Hugging Face Datasets](https://huggingface.co/docs/datasets), writing TSVs to `data/glue/sst2/`. Install with `pip install datasets` if missing. Repeat for MNLI by setting `--glue-task mnli`.
- Outputs include `results/glue_*` plots and `results/benchmark_table.md`.
- Use `--save-pretrained outputs/hsokv_glue` to persist checkpoints.

### 3.3 Split-CIFAR Continual Learning

```bash
python hsokv.py --benchmark cifar --allow-download \
    --iterations 4 --agents-per-manager 3 --agent-steps 30 \
    --num-managers 2 --visualize --save-pretrained outputs/hsokv_cifar
```

- Downloads CIFAR-10 to `data/cifar/` if needed.
- Tracks backward transfer via retention loader and reports FLOPs.

### 3.4 Distributed Swarm Simulation

```bash
# Simulated backend (deterministic, no Ray required)
python hsokv.py --iterations 1 --run-distributed --distributed-backend simulate --visualize

# Ray backend (if `pip install ray` and cluster available)
python hsokv.py --iterations 1 --run-distributed --distributed-backend ray --visualize
```

Generates `results/distributed_speedup.png`, `distributed_throughput.png`, and `distributed_reward.png`.

### 3.5 Complementary Experiment Pipelines (Stages 5–11)

- **Human vs model comparison:** `python -m experiments.human_comparison`
- **Comprehensive ablations (Stage 7):** `python -m experiments.comprehensive_ablations --seeds 3`
- **Scaling study (Stage 8):** `python -m experiments.scaling_study --scales 1000 5000 10000`
- **Cross-domain continual learning (Stage 9):** `python -m experiments.continual_learning --seeds 2`
- **Paper-ready bundle (Stage 11):** `python -m experiments.paper_experiments --seeds 3`

Outputs are written under `results/` (e.g., `results/human_comparison/`, `results/ablations/`, `results/paper/`).

### 3.6 Hugging Face Trainer Workflow

#### CLI
```bash
python hsokv.py --hf-train --iterations 2 \
    --agents-per-manager 2 --agent-steps 5 --num-managers 1 \
    --hf-output-dir outputs/hsokv_hf --visualize
```

#### Python API
```python
from hsokv_core import HFSwarmConfig, HFSwarmTrainer

config = HFSwarmConfig(
    meta_iterations=2,
    extras={"agents_per_manager": 2, "agent_steps": 5, "num_managers": 1}
)
trainer = HFSwarmTrainer(config=config)
summary = trainer.train()
print(summary["test_metrics"])
trainer.save_model("outputs/hsokv_hf")
```

### 3.7 Checkpoint Management

- Save: `python hsokv.py --iterations 5 --save-pretrained outputs/hsokv_checkpoint`
- Load/Evaluate: `python hsokv.py --load-pretrained outputs/hsokv_checkpoint --visualize`

Each checkpoint directory contains:
```
config.json
kv_memory.pt
pytorch_model.bin
tokenizer.json
```
which can be reloaded via `TransformerWithKV.from_pretrained(path, tokenizer)`.

---

## 4. Colab Workflow

A step-by-step notebook outline (environment setup, GLUE/CIFAR runs, distributed sim, checkpointing) is available in [`docs/colab_walkthrough.md`](docs/colab_walkthrough.md). It is tuned for a Colab T4 session (<30 minutes end-to-end) with reduced meta-iterations and agent steps.

---

## 5. Research Summary

- **Architecture:** 4-layer transformer (d_model=256) with gated KV retrieval. Memory embeddings are ℓ2-normalised; surprise/novelty gating regulates writes, confidence-weighted updates promote reliable entries, and forgetting utilities keep the store compact.
- **Swarm Optimisation:** Hierarchical loop of agents (optimiser/learning-rate strategies) overseen by a supervisor that monitors entropy, regret, and retention. Optional distributed execution across Ray / multiprocessing / simulated backends.
- **Memory Lifecycle (Stages 1–4):** Consolidation transfers high-confidence memories into model weights, context-aware retrieval boosts relevant domains/emotions, selective writing throttles redundancy, and automatic forgetting prunes low-utility or interfering entries.
- **Benchmarking & Experiments (Stages 5–11):**
  - *Few-shot GLUE / Split-CIFAR:* Baselines (fine-tune, KV-only, in-context) with FLOP accounting.
  - *Human comparison suite:* Learning curves against simulated human scores.
  - *Scaling study:* Retrieval latency and footprint vs. memory size with optimisation tips.
  - *Cross-domain continual learning:* Forward/backward transfer diagnostics (heatmaps, metrics).
  - *Paper-ready pipeline:* Figures, tables, and stats under `results/paper/`.
- **Tooling:** Visualisation helpers, comprehensive ablation automation, pytest integration suite, and a provisional patent draft capturing novelty areas.

Remaining roadmap item: Stage 12 production deployment module (inference API, service packaging).

---

## 6. Roadmap & Status

| Stage | Focus | Status |
|-------|-------|--------|
| 1 | Memory consolidation module | ✅ Completed |
| 2 | Context-aware retrieval | ✅ Completed |
| 3 | Surprise-based selective writing | ✅ Completed |
| 4 | Automatic forgetting mechanism | ✅ Completed |
| 5 | Human comparison experiments | ✅ Completed |
| 6 | Memory visualisation toolkit | ✅ Completed (advanced interactive views pending) |
| 7 | Comprehensive ablation automation | ✅ Completed |
| 8 | Scaling experiments | ✅ Completed (extended ANN backends optional) |
| 9 | Cross-domain continual learning suite | ✅ Completed |
| 10 | Pytest integration & validation | ✅ Completed |
| 11 | Paper-ready experiment suite | ✅ Completed |
| 12 | Production deployment API | ⏳ Pending |
| Bonus | Provisional patent draft | ✅ Draft prepared |

---

## 7. Reproducibility Checklist

- `python hsokv.py --mode test` (sanity suite)
- `pytest tests/test_integration.py` (integration coverage)
- `python hsokv.py --iterations 10 --visualize` (synthetic)
- `python hsokv.py --benchmark glue --allow-download` (real few-shot GLUE)
- `python hsokv.py --benchmark cifar --allow-download` (Split-CIFAR continual)
- `python hsokv.py --run-distributed --distributed-backend simulate` (distributed speedup)
- `python hsokv.py --hf-train --hf-output-dir outputs/hsokv_hf` (HF trainer checkpoint)

Run each command; inspect `results/` and `outputs/` for plots, tables, and checkpoints.

---

## 8. Troubleshooting

| Issue | Resolution |
|-------|------------|
| Dataset download blocked | Upload pre-downloaded GLUE/CIFAR folders into `data/`, or mount Google Drive. |
| Runtime exceeding budget | Reduce `--iterations`, `--agent-steps`, `--agents-per-manager`; lower `d_model` in `config.py`. |
| KV hit rate near zero | Increase `--definition_max_length`, tweak `--kv-confidence-threshold`, or warm-start memory with `--load-pretrained`. |
| Ray errors | Fallback to `--distributed-backend simulate` or install Ray via `pip install ray`. |
| Checkpoint mismatch | Delete checkpoint directory and rerun `--save-pretrained`; ensures config/tokenizer align. |

---

## 9. Contribution & Citation

1. Fork and submit PRs against the `main` branch (avoid editing `hsokv_legacy_dont_touch.py`).
2. Keep modifications ASCII-only unless upstream files already use Unicode.
3. For academic use, cite the project as:



---

Questions or issues? Open a GitHub issue or reach out via the project discussion board. Happy experimenting!
