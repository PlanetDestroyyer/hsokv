# H-SOKV Colab Walkthrough

This guide outlines a reproducible Colab workflow for the research-grade H-SOKV prototype. It assumes access to a T4/V100 GPU instance with `torch>=2.0` and the ability to download SST-2/MNLI and CIFAR-10 datasets.

## 1. Environment Setup

```python
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q matplotlib numpy tqdm scikit-learn
!git clone https://your-fork/hsokv.git
%cd hsokv
```

Set device and enable downloads via CLI flags (`--allow-download`).

## 2. Synthetic Sanity Check

```python
!python hsokv.py --iterations 5 --visualize
```

Verify that plots land in `results/` and loss decreases.

## 3. Few-Shot GLUE (SST-2 / MNLI)

```python
!python hsokv.py --benchmark glue --iterations 5 --allow-download \
    --glue-task sst2 --visualize --hf-train --hf-output-dir outputs/hsokv_glue
```

Notes:
- Downloads SST-2 into `data/glue/sst2/` on first run.
- Adjust `--iterations`, `--agents-per-manager`, and `--agent-steps` to keep runtime <30 minutes.
- Resulting plots: `results/glue_benchmark_scores.png`, `results/glue_benchmark_flops.png`.

## 4. Split-CIFAR Continual Learning

```python
!python hsokv.py --benchmark cifar --iterations 4 --allow-download \
    --agents-per-manager 3 --agent-steps 30 --num-managers 2 \
    --hf-train --hf-output-dir outputs/hsokv_cifar
```

Tips:
- CIFAR-10 downloads to `data/cifar` automatically when `--allow-download` is set.
- Monitor runtime and reduce `--meta-iterations`, `--agent-steps`, or batch size if needed.
- Inspect `results/distributed_*` plots if you pair with `--run-distributed`.

## 5. Saving & Reloading Checkpoints

```python
!python hsokv.py --iterations 2 --save-pretrained outputs/hsokv_synthetic
!python hsokv.py --load-pretrained outputs/hsokv_synthetic --visualize
```

`outputs/hsokv_synthetic/` will contain:
- `config.json`
- `pytorch_model.bin`
- `kv_memory.pt`
- `tokenizer.json`

## 6. Distributed Swarm Simulation (Optional)

```python
!python hsokv.py --iterations 1 --run-distributed --distributed-backend simulate --visualize
```

Use `--distributed-backend ray` if the Ray runtime is installed (`pip install ray`).

## 7. Metrics & Reporting

- Final results tables print to stdout and `results/benchmark_table.md`.
- LaTeX-ready tables land under `results/` (e.g., `ablation_table.tex`).
- Upload `results/` artifacts with the manuscript or share via Colab.

## 8. Troubleshooting

- **Dataset downloads blocked**: Upload pre-downloaded GLUE/CIFAR folders into `data/` or mount Google Drive.
- **Runtime too long**: Reduce `--meta-iterations`, `--agents-per-manager`, and `--agent-steps`; shrink `--d_model` in `config.py` if necessary.
- **KV hit rates low**: Increase `--definition_max_length` or adjust memory thresholds (`--kv-confidence-threshold`).

For deeper experimentation, edit `CONFIG` in `hsokv_core/config.py` or pass overrides via CLI (`--meta-iterations`, `--agent-steps`, etc.).

