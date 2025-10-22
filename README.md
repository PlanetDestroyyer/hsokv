# H-SOKV: Hierarchical Swarm-KV Research Prototype

H-SOKV couples a lightweight transformer backbone with an adaptive key–value memory and a three-tier swarm optimizer (agents → managers → supervisor). The goal is to demonstrate human-like one-shot vocabulary learning, few-shot generalisation on GLUE, continual retention on Split-CIFAR, and scalable exploration via distributed swarms—all within a reproducible PyTorch codebase that runs end-to-end on modest hardware (Colab T4).

---

## 1. Repository Layout

```
hsokv.py                  # CLI entrypoint and experiment orchestration
hsokv_core/
  ablations.py           # Ablation suite (full, KV-only, swarm-only, neither)
  benchmarks.py          # Few-shot GLUE + Split-CIFAR loaders/evaluators
  config.py              # Global CONFIG dict and override helpers
  data.py                # Synthetic rare-word dataset + tokenizer utilities
  distributed.py         # Toy CartPole swarm simulator (Ray/mp/simulate)
  hf_adapter.py          # Hugging Face-style trainer wrapper (HFSwarmTrainer)
  memory.py              # Normalised KeyValueMemory implementation
  model.py               # TransformerWithKV + BaselineTransformer
  metrics.py             # FLOP estimation and reporting helpers
  swarm.py               # Agent / Manager / Supervisor hierarchy
  training.py            # Core training loops and baselines
  utils.py               # Device helpers, seeding, diversity metrics
results/                  # Plots, tables, and benchmark logs (created on demand)
docs/                     # Colab walkthrough and supplementary notes
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

- First run downloads SST-2 into `data/glue/sst2/`. Repeat for MNLI by setting `--glue-task mnli`.
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

### 3.5 Hugging Face Trainer Workflow

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

### 3.6 Checkpoint Management

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

- **Architecture:** 4-layer transformer (d_model=256) with gated KV retrieval. Memory embeddings are ℓ2-normalised; confidence-weighted updates encourage reliable entries.
- **Swarm Optimisation:** Hierarchical loop of agents (optimiser/learning-rate search) managed by pooling strategies, overseen by a supervisor tracking entropy/regret. Supports diverse strategies and distributed execution.
- **Benchmarks:**
  - *Few-shot GLUE:* 16-shot SST-2/MNLI with FLOP-aligned baselines (fine-tune, KV-only, in-context)
  - *Split-CIFAR-10:* 5-task continual learning evaluating retention and backward transfer
- **Distributed Swarm:** Toy CartPole-like environment to demonstrate scaling behaviour (Ray, multiprocessing, or deterministic simulator).
- **HF/OSS Hooks:** Pretrained checkpoints, trainer wrapper, and documented workflow for reproducibility.

Pending work before publication: execute real-data sweeps (`--allow-download`), tune hyperparameters for <30 minute runs, and curate plots/tables for Stage 6 packaging.

---

## 6. Roadmap & Status

| Stage | Focus | Status |
|-------|-------|--------|
| 1 | Core hardening (KV normalisation, budgets) | ✅ Completed |
| 2 | Ablations & instrumentation | ✅ Completed |
| 3 | GLUE & Split-CIFAR benchmarks | ✅ Completed (real-data run outstanding) |
| 4 | Distributed swarm simulation | ✅ Completed |
| 5 | Hugging Face & OSS hooks | 🔄 In progress (trainer + docs done; real-data Colab verification pending) |
| 6 | Final validation & packaging | ⏳ Pending |

---

## 7. Reproducibility Checklist

- `python hsokv.py --mode test` (sanity suite)
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
