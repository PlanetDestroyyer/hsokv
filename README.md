# H-SOKV Research Prototype

Hierarchical Swarm-KV (H-SOKV) is a research-grade prototype that pairs a transformer backbone with a hierarchical swarm optimizer and adaptive key-value memory. This repo contains the entire training stack, synthetic vocabulary tasks, few-shot GLUE and Split-CIFAR benchmarks, and distributed swarm simulations.

## Quick Start

```bash
# Synthetic experiment with plots
python hsokv.py --iterations 10 --visualize

# Run staged validation suite
python hsokv.py --mode test
```

Key CLI flags:

- `--benchmark glue|cifar` – run the corresponding benchmark pipeline (uses synthetic fallback if datasets missing).
- `--allow-download` – permit benchmark loaders to download SST-2/MNLI or CIFAR-10 into `data/`.
- `--run-distributed --distributed-backend simulate` – generate distributed swarm speedup/throughput plots.
- `--save-pretrained PATH` / `--load-pretrained PATH` – persist or reload an H-SOKV checkpoint (model, KV memory, tokenizer).
- `--hf-train --hf-output-dir OUT` – train via the Hugging Face-style trainer wrapper and optionally save the checkpoint.

## Hugging Face Integration

You can drive H-SOKV through the provided `HFSwarmTrainer` API:

```python
from hsokv_core import HFSwarmConfig, HFSwarmTrainer

config = HFSwarmConfig(
    meta_iterations=2,
    extras={"agents_per_manager": 2, "agent_steps": 5, "num_managers": 1},
)
trainer = HFSwarmTrainer(config=config)
summary = trainer.train()
print(summary["test_metrics"])
trainer.save_model("hf_hsokv_checkpoint")
```

The CLI mirrors this behaviour:

```bash
python hsokv.py --hf-train --hf-output-dir hf_hsokv_checkpoint --iterations 2 \
  --agents-per-manager 2 --agent-steps 5 --num-managers 1 --visualize
```

Any checkpoint saved with `save_pretrained` stores `config.json`, `pytorch_model.bin`, `kv_memory.pt`, and `tokenizer.json`, making it easy to reload via `TransformerWithKV.from_pretrained(...)`.

## Real Benchmarks

1. Download data (or run once with `--allow-download`).
   ```bash
   python hsokv.py --benchmark glue --allow-download --iterations 5 --visualize
   python hsokv.py --benchmark cifar --allow-download --iterations 5 --visualize
   ```
2. Collect plots (`results/`), Markdown/LaTeX tables, and optional checkpoints via `--save-pretrained`.

For <30 minute Colab runs, reduce meta iterations (`--iterations`), agent steps (`--agent-steps`), or managers (`--num-managers`). A step-by-step Colab notebook outline lives in [`docs/colab_walkthrough.md`](docs/colab_walkthrough.md).

## Distributed Swarm Simulation

```bash
python hsokv.py --iterations 1 --run-distributed --distributed-backend simulate --visualize
```

This produces `distributed_speedup.png`, `distributed_throughput.png`, and `distributed_reward.png` in `results/`.  Switch to `--distributed-backend ray` if Ray is available.

## Roadmap

- Stage 5 (HF & OSS) – finalize Hugging Face Trainer wrapper docs, Colab demo, and real benchmark configurations.
- Stage 6 (Packaging) – lock benchmark metrics, publish plots/tables, and prepare release notes for arXiv/OSS.
