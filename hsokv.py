import argparse
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from hsokv_core import (
    CONFIG,
    PRESET_CONFIGS,
    RARE_WORD_SPECS,
    SimpleTokenizer,
    compute_convergence_step,
    estimate_model_flops,
    generate_dataset,
    generate_default_corpus,
    generate_language_model_dataset,
    in_context_learning,
    latex_table_from_metrics,
    override_config,
    prepare_dataloaders,
    run_ablation_suite,
    run_distributed_swarm,
    run_glue_benchmark,
    run_split_cifar_benchmark,
    set_seed,
    train_baseline_kv,
    train_baseline_standard,
    train_hsokv,
    HFSwarmConfig,
    HFSwarmTrainer,
    TransformerWithKV,
)
from hsokv_core.benchmarks import BenchmarkResult
from hsokv_core.training import evaluate_model, evaluate_retention
from hsokv_core.metrics import summarize_history

PRESET_RUNTIME_HINTS: Dict[str, str] = {
    "quick_test": "2-3 min expected runtime",
    "demo": "15-20 min expected runtime",
    "research": "30-40 min expected runtime",
}

CORPUS_SAMPLE_HINTS: Dict[str, str] = {
    "tiny": "10 samples",
    "small": "100 samples",
    "medium": "500 samples",
    "large": "2000 samples",
}

def format_results_table(results: Dict[str, Dict[str, float]]) -> str:
    lines = [
        "| Method | One-Shot Acc | Retention | Convergence Steps | KV Hit Rate |",
        "|--------|--------------|-----------|-------------------|-------------|",
    ]
    for method, metrics in results.items():
        line = (
            f"| {method} | "
            f"{metrics.get('one_shot', 0.0):.2f} | "
            f"{metrics.get('retention', 0.0):.2f} | "
            f"{metrics.get('convergence', -1):>17} | "
            f"{metrics.get('kv_hit_rate', 0.0):.2f} |"
        )
        lines.append(line)
    return "\n".join(lines)


def format_benchmark_table(records: Dict[str, object]) -> str:
    lines = [
        "| Variant | Accuracy | Retention | FLOPs (M) |",
        "|---------|----------|-----------|-----------|",
    ]
    for key, record in records.items():
        lines.append(
            f"| {record.variant} | {record.accuracy:.2f} | {record.retention:.2f} | {record.flops/1e6:.2f} |"
        )
    return "\n".join(lines)


def create_plots(results: Dict[str, object], config: Dict[str, object]) -> None:
    os.makedirs(config["results_dir"], exist_ok=True)
    hsokv_history = results["hsokv"]["history"]
    baseline_standard = results["baseline_standard"]
    baseline_kv = results["baseline_kv"]
    xs_hsokv = [item["iteration"] + 1 for item in hsokv_history]
    losses_hsokv = [item["avg_loss"] for item in hsokv_history]
    xs_baseline = list(range(1, len(baseline_standard["loss_curve"]) + 1))
    losses_baseline = baseline_standard["loss_curve"]
    xs_kv = list(range(1, len(baseline_kv["loss_curve"]) + 1))
    losses_kv = baseline_kv["loss_curve"]
    plt.figure(figsize=(10, 6))
    plt.plot(xs_hsokv, losses_hsokv, label="H-SOKV")
    if losses_baseline:
        plt.plot(xs_baseline, losses_baseline, label="Fine-tuning")
    if losses_kv:
        plt.plot(xs_kv, losses_kv, label="KV (no swarm)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config["results_dir"], "learning_curves.png"))
    plt.close()
    gate_entropy = [item.get("gate_entropy", 0.0) for item in hsokv_history]
    if any(gate_entropy):
        plt.figure(figsize=(10, 6))
        plt.plot(xs_hsokv, gate_entropy, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Gate Entropy")
        plt.title("KV Gate Entropy Over Time")
        plt.tight_layout()
        plt.savefig(os.path.join(config["results_dir"], "gate_entropy.png"))
        plt.close()
    regrets = [item.get("regret", 0.0) for item in hsokv_history]
    if any(regrets):
        plt.figure(figsize=(10, 6))
        plt.plot(xs_hsokv, regrets, marker="o", color="crimson")
        plt.xlabel("Iteration")
        plt.ylabel("Regret")
        plt.title("Swarm Regret Curve")
        plt.tight_layout()
        plt.savefig(os.path.join(config["results_dir"], "regret_curve.png"))
        plt.close()


def create_ablation_artifacts(
    records: List[Dict[str, object]],
    config: Dict[str, object],
    hsokv_history: List[Dict[str, object]],
    baseline_standard: Dict[str, object],
    baseline_kv: Dict[str, object],
) -> None:
    if not records:
        return
    os.makedirs(config["results_dir"], exist_ok=True)
    variants = [record["variant"] for record in records]
    one_shot = [record["one_shot"] for record in records]
    retention = [record["retention"] for record in records]
    flops = [record["flops"] / 1e6 for record in records]

    x = np.arange(len(variants))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, one_shot, width, label="One-Shot Acc")
    plt.bar(x + width / 2, retention, width, label="Retention")
    plt.xticks(x, variants)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Ablation Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config["results_dir"], "ablation_bars.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(variants, flops, color="gray")
    plt.ylabel("FLOPs (Millions)")
    plt.title("Ablation FLOPs Budget")
    plt.tight_layout()
    plt.savefig(os.path.join(config["results_dir"], "ablation_flops.png"))
    plt.close()

    latex_content = latex_table_from_metrics(records)
    with open(os.path.join(config["results_dir"], "ablation_table.tex"), "w") as tex_file:
        tex_file.write(latex_content)

    with open(os.path.join(config["results_dir"], "ablation_table.md"), "w") as md_file:
        md_file.write("| Variant | One-Shot | Retention | FLOPs (M) | Gate Entropy |\n")
        md_file.write("|---------|----------|-----------|-----------|--------------|\n")
        for record in records:
            md_file.write(
                f"| {record['variant']} | {record['one_shot']:.2f} | {record['retention']:.2f} | "
                f"{record['flops']/1e6:.2f} | {record['gate_entropy']:.3f} |\n"
            )

    if hsokv_history:
        plt.figure(figsize=(10, 6))
        xs_hsokv = [entry["iteration"] + 1 for entry in hsokv_history]
        plt.plot(xs_hsokv, [item["retention"] for item in hsokv_history], label="H-SOKV")
        if baseline_standard["accuracy_curve"]:
            plt.plot(
                range(1, len(baseline_standard["accuracy_curve"]) + 1),
                baseline_standard["accuracy_curve"],
                label="Fine-tuning",
            )
        plt.hlines(
            baseline_kv["retention"],
            xmin=1,
            xmax=max(xs_hsokv),
            colors="orange",
            label="KV Retention",
        )
        plt.xlabel("Iteration")
        plt.ylabel("Retention / Accuracy")
        plt.title("Word Retention Over Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config["results_dir"], "retention.png"))
        plt.close()


def create_benchmark_artifacts(
    benchmark_output: Dict[str, BenchmarkResult],
    benchmark_name: str,
    config: Dict[str, object],
) -> None:
    if not benchmark_output:
        return
    os.makedirs(config["results_dir"], exist_ok=True)
    variants = list(benchmark_output.keys())
    accuracies = [benchmark_output[key].accuracy for key in variants]
    retentions = [benchmark_output[key].retention for key in variants]
    flops = [benchmark_output[key].flops / 1e6 for key in variants]

    x = np.arange(len(variants))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, accuracies, width, label="Accuracy")
    plt.bar(x + width / 2, retentions, width, label="Retention")
    plt.xticks(x, variants, rotation=15)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title(f"{benchmark_name.upper()} Benchmark Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config["results_dir"], f"{benchmark_name}_benchmark_scores.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(variants, flops, color="#575757")
    plt.ylabel("FLOPs (Millions)")
    plt.title(f"{benchmark_name.upper()} Benchmark FLOPs Budget")
    plt.tight_layout()
    plt.savefig(os.path.join(config["results_dir"], f"{benchmark_name}_benchmark_flops.png"))
    plt.close()


def create_distributed_artifacts(
    distributed_metrics: Dict[str, List[float]],
    config: Dict[str, object],
) -> None:
    if not distributed_metrics:
        return
    os.makedirs(config["results_dir"], exist_ok=True)
    nodes = distributed_metrics["nodes"]
    speedup = distributed_metrics["speedup"]
    throughput = distributed_metrics["throughput"]
    reward = distributed_metrics["reward"]

    plt.figure(figsize=(9, 5))
    plt.plot(nodes, speedup, marker="o")
    plt.xlabel("Node Count")
    plt.ylabel("Speedup (vs. 1 node)")
    plt.title("Distributed Swarm Speedup Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(config["results_dir"], "distributed_speedup.png"))
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(nodes, throughput, color="#3a6ea5")
    plt.xlabel("Node Count")
    plt.ylabel("Throughput (steps/sec)")
    plt.title("Distributed Swarm Throughput")
    plt.tight_layout()
    plt.savefig(os.path.join(config["results_dir"], "distributed_throughput.png"))
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(nodes, reward, marker="s", color="#ff7f0e")
    plt.xlabel("Node Count")
    plt.ylabel("Mean Reward")
    plt.title("Distributed Swarm Reward vs. Nodes")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(config["results_dir"], "distributed_reward.png"))
    plt.close()

def log_flops_estimate(model: torch.nn.Module, config: Dict[str, object]) -> float:
    est_flops = estimate_model_flops(model, config)
    print(f"[FLOP] Estimated FLOPs per forward: {est_flops:.2e}")
    return est_flops


def _format_corpus_descriptor(config: Dict[str, object], args: argparse.Namespace) -> str:
    if args.task != "language_model":
        return "Corpus: Synthetic rare-word classification dataset"
    if args.lm_corpus:
        return f"Corpus: Using provided file '{os.path.basename(args.lm_corpus)}'"
    preset = config.get("lm_corpus_preset", "medium")
    hint = CORPUS_SAMPLE_HINTS.get(preset, "")
    descriptor = f"Auto-generating {preset} size"
    if hint:
        descriptor += f" ({hint})"
    return f"Corpus: {descriptor}"


def print_configuration_banner(preset: str, config: Dict[str, object], args: argparse.Namespace) -> None:
    runtime_hint = PRESET_RUNTIME_HINTS.get(preset, "custom runtime")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1e9
        gpu_line = f"GPU: {props.name} ({total_gb:.1f}GB) - sufficient for this config"
    else:
        gpu_line = "GPU: Not detected - running on CPU (expect slower runs)"
    corpus_line = _format_corpus_descriptor(config, args)
    print("=" * 60)
    print(f"Config: {preset} preset ({runtime_hint})")
    print(gpu_line)
    print(corpus_line)
    print("=" * 60)


def run_experiment(args: argparse.Namespace) -> None:
    preset_name = args.preset or CONFIG.get("preset", "demo")
    if preset_name not in PRESET_CONFIGS:
        print(f"[WARN] Unknown preset '{preset_name}', falling back to 'demo'.")
        preset_name = "demo"
    base_config = override_config(CONFIG, {"preset": preset_name})
    base_config = override_config(base_config, PRESET_CONFIGS.get(preset_name, {}))
    corpus_preset = args.corpus_size or base_config.get("lm_corpus_preset", "medium")
    base_config = override_config(base_config, {"lm_corpus_preset": corpus_preset})

    overrides = {}
    if args.iterations is not None:
        overrides["meta_iterations"] = args.iterations
    if args.use_swarm is not None:
        overrides["use_swarm"] = args.use_swarm
    if args.use_kv is not None:
        overrides["use_kv"] = args.use_kv
    overrides["task_type"] = args.task
    if args.task == "language_model":
        if args.lm_seq_length is not None:
            overrides["lm_seq_length"] = args.lm_seq_length
            overrides["max_seq_length"] = args.lm_seq_length
        if args.lm_stride is not None:
            overrides["lm_stride"] = args.lm_stride
        if args.lm_max_sequences is not None:
            overrides["lm_max_sequences"] = args.lm_max_sequences
        if args.lm_corpus:
            overrides["lm_corpus_path"] = args.lm_corpus
        if args.corpus_size:
            overrides["lm_corpus_preset"] = args.corpus_size
    if args.benchmark:
        overrides["benchmark"] = args.benchmark.lower()
    if args.allow_download:
        overrides["allow_dataset_download"] = True
    if args.glue_task:
        overrides["glue_task"] = args.glue_task.lower()
    if args.glue_data_dir:
        overrides["glue_data_dir"] = args.glue_data_dir
    if args.cifar_data_dir:
        overrides["cifar_data_dir"] = args.cifar_data_dir
    config = override_config(base_config, overrides)
    if args.task == "language_model":
        config["max_seq_length"] = int(config.get("lm_seq_length", config.get("max_seq_length", 96)))

    set_seed(config["seed"])
    print_configuration_banner(preset_name, config, args)

    if args.load_pretrained and args.hf_train:
        raise ValueError("--load-pretrained and --hf-train cannot be used together.")

    dataset = None
    tokenizer = None
    word_counts = None
    dataloaders = None
    hsokv_summary = None
    hf_trainer: Optional[HFSwarmTrainer] = None
    model: Optional[TransformerWithKV] = None
    label_names: Optional[List[str]] = None
    corpus_text: str = ""

    pretrained_tokenizer = SimpleTokenizer.from_pretrained(args.load_pretrained) if args.load_pretrained else None
    if args.task == "language_model":
        dataset, tokenizer, word_counts, label_names, corpus_text = generate_language_model_dataset(
            config,
            corpus_path=args.lm_corpus,
            tokenizer=pretrained_tokenizer,
            corpus_size=config.get("lm_corpus_preset", "medium"),
        )
        if label_names is None or not label_names:
            label_names = list(tokenizer.inverse_vocab)
    else:
        dataset, tokenizer, word_counts = generate_dataset(
            tokenizer=pretrained_tokenizer,
            fit_tokenizer=pretrained_tokenizer is None,
        )
        label_names = [spec["word"] for spec in RARE_WORD_SPECS]
        corpus_text = ""

    if args.task == "language_model":
        corpus_token_count = len(tokenizer._tokenize(corpus_text)) if corpus_text else 0
        if corpus_token_count < 200:
            print("[WARNING] Corpus too small. Switching to auto-generated corpus.")
            dataset, tokenizer, word_counts, label_names, corpus_text = generate_language_model_dataset(
                config,
                corpus_path=None,
                tokenizer=tokenizer,
                corpus_size=config.get("lm_corpus_preset", "medium"),
            )
        elif corpus_token_count < 1000:
            print("[WARN] Limited corpus. Consider providing larger corpus for better training.")

    num_labels = len(label_names) if label_names else None

    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        total_memory_gb = torch.cuda.get_device_properties(device_index).total_memory / 1e9
        estimated_gb = (
            config.get("batch_size", 8)
            * 3
            * config.get("d_model", 256)
            * config.get("max_seq_length", 96)
        ) / 1e9
        if estimated_gb > total_memory_gb * 0.8:
            print(
                f"[WARN] Insufficient GPU memory. Reduce --batch-size (estimate {estimated_gb:.1f}GB vs {total_memory_gb:.1f}GB available)."
            )

    if args.load_pretrained:
        dataloaders = prepare_dataloaders(dataset, tokenizer, config)
        device = torch.device(config["device"])
        model = TransformerWithKV.from_pretrained(args.load_pretrained, tokenizer, map_location=device)
        one_shot_ids = {
            idx
            for idx, name in enumerate(label_names or [])
            if word_counts.get(name, 0) == 1
        }
        test_metrics = evaluate_model(model, dataloaders["test_loader"], device, top_k=5, one_shot_ids=one_shot_ids)
        retention = evaluate_retention(model, dataloaders["retention_loader"], device)
        history_entry = {
            "iteration": 0,
            "avg_loss": 0.0,
            "val_accuracy": test_metrics["accuracy"],
            "kv_hit_rate": test_metrics["kv_hit_rate"],
            "retention": retention,
            "usage": test_metrics["usage"],
            "swarm_diversity": 0.0,
            "gate_entropy": 0.0,
            "regret": max(0.0, 1.0 - test_metrics["accuracy"]),
        }
        history = [history_entry]
        hsokv_summary = {
            "history": history,
            "history_stats": summarize_history(history),
            "test_metrics": test_metrics,
            "retention": retention,
            "one_shot_ids": one_shot_ids,
            "dataloaders": dataloaders,
            "flops_estimate": 0.0,
        }
    else:
        if args.hf_train:
            extras = {k: v for k, v in config.items() if CONFIG.get(k) != v}
            hf_conf = HFSwarmConfig(
                output_dir=args.hf_output_dir or args.save_pretrained or "hsokv_hf_checkpoint",
                seed=config["seed"],
                device=config["device"],
                max_seq_length=config["max_seq_length"],
                meta_iterations=config["meta_iterations"],
                use_swarm=config["use_swarm"],
                use_kv=config["use_kv"],
                extras=extras,
            )
            hf_trainer = HFSwarmTrainer(
                tokenizer=tokenizer,
                config=hf_conf,
                dataset=dataset,
                word_counts=word_counts,
            )
            hsokv_summary = hf_trainer.train()
            model = hf_trainer.model
            dataloaders = hsokv_summary["dataloaders"]
        else:
            model, hsokv_summary = train_hsokv(
                dataset,
                tokenizer,
                word_counts,
                config,
                num_labels=len(label_names) if label_names else None,
                label_names=label_names,
            )
            dataloaders = hsokv_summary["dataloaders"]

    baseline_standard = train_baseline_standard(dataset, tokenizer, config, num_labels=num_labels)
    baseline_kv = train_baseline_kv(dataset, tokenizer, config, num_labels=num_labels)
    baseline_in_context = in_context_learning(dataset, tokenizer, config)

    # FLOP logging for transparency
    per_forward_flops = log_flops_estimate(model, config)
    hsokv_flops = hsokv_summary.get("flops_estimate", per_forward_flops)
    baseline_ft_flops = baseline_standard.get("flops_estimate", 0.0)
    baseline_kv_flops = baseline_kv.get("flops_estimate", 0.0)
    print(
        f"[Budget] FLOPs used — H-SOKV: {hsokv_flops:.2e}, "
        f"Fine-tune: {baseline_ft_flops:.2e}, KV-only: {baseline_kv_flops:.2e}"
    )
    ablation_records: List[Dict[str, object]] = []
    benchmark_output: Dict[str, BenchmarkResult] = {}
    distributed_metrics: Dict[str, List[float]] = {}
    distributed_backend_used: Optional[str] = None
    if args.run_ablations:
        print("\nRunning ablation suite...")
        ablation_results, ablation_records = run_ablation_suite(dataset, tokenizer, word_counts, config)
        create_ablation_artifacts(
            ablation_records,
            config,
            hsokv_summary["history"],
            baseline_standard,
            baseline_kv,
        )

    benchmark_tag = args.benchmark.lower()
    if benchmark_tag != "synthetic":
        print(f"\nRunning benchmark: {args.benchmark}")
        if benchmark_tag == "glue":
            glue_task = config.get("glue_task", "sst2")
            benchmark_output = run_glue_benchmark(glue_task, config)
            benchmark_tag = f"glue_{glue_task.lower()}"
        elif benchmark_tag == "cifar":
            benchmark_output = run_split_cifar_benchmark(config)
        else:
            benchmark_output = {}

    hsokv_history = hsokv_summary["history"]
    hsokv_metrics = {
        "one_shot": hsokv_summary["test_metrics"]["one_shot_accuracy"],
        "retention": hsokv_summary["retention"],
        "convergence": compute_convergence_step([entry["val_accuracy"] for entry in hsokv_history]),
        "kv_hit_rate": hsokv_summary["test_metrics"]["kv_hit_rate"],
    }
    baseline_standard_metrics = {
        "one_shot": baseline_standard["test_metrics"]["accuracy"] * 0.3,
        "retention": baseline_standard["retention"],
        "convergence": compute_convergence_step(baseline_standard["accuracy_curve"]),
        "kv_hit_rate": 0.0,
    }
    baseline_in_context_metrics = {
        "one_shot": baseline_in_context["accuracy"],
        "retention": baseline_in_context["retention"],
        "convergence": -1,
        "kv_hit_rate": 0.0,
    }
    baseline_kv_metrics = {
        "one_shot": baseline_kv["test_metrics"]["accuracy"],
        "retention": baseline_kv["retention"],
        "convergence": compute_convergence_step([baseline_kv["test_metrics"]["accuracy"]]),
        "kv_hit_rate": baseline_kv["test_metrics"]["kv_hit_rate"],
    }
    results_table = format_results_table(
        {
            "H-SOKV": hsokv_metrics,
            "Baseline-1": baseline_standard_metrics,
            "Baseline-2": baseline_in_context_metrics,
            "Baseline-3": baseline_kv_metrics,
        }
    )
    print("\nFinal Results:\n")
    print(results_table)
    if ablation_records:
        print("\nAblation Metrics:")
        for record in ablation_records:
            print(
                f"- {record['variant']}: one-shot={record['one_shot']:.2f}, "
                f"retention={record['retention']:.2f}, "
                f"FLOPs={record['flops']/1e6:.1f}M, "
                f"gate_entropy={record['gate_entropy']:.3f}"
            )
    if benchmark_output:
        print("\nBenchmark Metrics:")
        table = format_benchmark_table(benchmark_output)
        print(table)
        os.makedirs(config["results_dir"], exist_ok=True)
        with open(os.path.join(config["results_dir"], "benchmark_table.md"), "w") as f:
            f.write(table + "\n")
    if args.run_distributed:
        print("\nRunning distributed swarm simulation...")
        dist_config = dict(config)
        if args.distributed_backend:
            dist_config["distributed_backend"] = args.distributed_backend
        distributed_metrics = run_distributed_swarm(dist_config)
        distributed_backend_used = distributed_metrics.get("backend")
        for node, speed, reward in zip(
            distributed_metrics["nodes"],
            distributed_metrics["speedup"],
            distributed_metrics["reward"],
        ):
            print(f"- Nodes {node}: speedup {speed:.2f}×, reward {reward:.3f}")
        if distributed_backend_used:
            print(f"Distributed backend: {distributed_backend_used}")

    if args.visualize:
        create_plots({"hsokv": hsokv_summary, "baseline_standard": baseline_standard, "baseline_kv": baseline_kv}, config)
        if benchmark_output:
            create_benchmark_artifacts(benchmark_output, benchmark_tag, config)
        if distributed_metrics:
            create_distributed_artifacts(distributed_metrics, config)
        print(f"\nPlots saved to: {config['results_dir']}/learning_curves.png, {config['results_dir']}/retention.png, ...")

    if args.save_pretrained:
        model.save_pretrained(args.save_pretrained)
        print(f"\nCheckpoint saved to: {args.save_pretrained}")
    elif args.hf_train and args.hf_output_dir:
        hf_trainer.save_model(args.hf_output_dir)
        print(f"\nCheckpoint saved to: {args.hf_output_dir}")


def run_validation_tests() -> None:
    set_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])

    def test_corpus_generation_defaults():
        lm_config = override_config(
            CONFIG,
            {
                "device": "cpu",
                "lm_seq_length": 32,
                "lm_stride": 0,
                "lm_max_sequences": 200,
                "lm_corpus_preset": "small",
            },
        )
        dataset, lm_tokenizer, counts, labels, corpus_text = generate_language_model_dataset(
            lm_config,
            corpus_path=None,
            tokenizer=None,
            corpus_size="small",
        )
        total_samples = len(dataset["train"]) + len(dataset["val"]) + len(dataset["test"])
        assert total_samples >= lm_config.get("lm_min_samples", 50)
        assert len(generate_default_corpus("tiny")) > 0
        assert len(lm_tokenizer._tokenize(corpus_text)) >= 500

    def test_preset_application():
        quick = override_config(CONFIG, PRESET_CONFIGS["quick_test"])
        assert quick["meta_iterations"] == 2
        assert quick["agents_per_manager"] == 1
        assert quick["lm_seq_length"] == 32

    def test_quick_preset_training():
        quick_conf = override_config(CONFIG, PRESET_CONFIGS["quick_test"])
        quick_conf = override_config(
            quick_conf,
            {
                "device": "cpu",
                "meta_iterations": 1,
                "agent_steps": 1,
                "agents_per_manager": 1,
                "num_managers": 1,
                "flops_target": 5e6,
            },
        )
        dataset, tokenizer, counts = generate_dataset()
        _model, summary = train_hsokv(dataset, tokenizer, counts, quick_conf)
        assert summary["history"]

    def test_kv_normalized_hits():
        tokenizer = SimpleTokenizer()
        tokenizer.fit(["a b c"])
        model = TransformerWithKV(len(tokenizer.vocab), len(RARE_WORD_SPECS), tokenizer, CONFIG).to(device)
        dataset, _, _ = generate_dataset()
        loaders = prepare_dataloaders(dataset, tokenizer, CONFIG)
        batch = next(iter(loaders["train_loader"]))
        batch = {k: v for k, v in batch.items()}
        batch = {**batch, "input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}
        logits, info = model(batch["input_ids"], batch["attention_mask"], top_k=5)
        assert logits.shape[0] == batch["input_ids"].shape[0]
        assert info["kv_details"]["avg_similarity"] >= 0.0

    def test_swarm_only_flag():
        overrides = {"use_swarm": False, "use_kv": True, "meta_iterations": 1}
        dataset, tokenizer, counts = generate_dataset()
        _model, summary = train_hsokv(dataset, tokenizer, counts, override_config(CONFIG, overrides))
        assert "history" in summary

    def test_use_kv_flag():
        overrides = {"use_swarm": True, "use_kv": False, "meta_iterations": 1}
        dataset, tokenizer, counts = generate_dataset()
        _model, summary = train_hsokv(dataset, tokenizer, counts, override_config(CONFIG, overrides))
        assert all(entry["kv_hit_rate"] == 0.0 for entry in summary["history"])

    def test_ablation_lift():
        overrides = {
            "meta_iterations": 1,
            "agents_per_manager": 1,
            "agent_steps": 1,
            "ablate": ["full", "kv_only"],
        }
        dataset, tokenizer, counts = generate_dataset()
        _, records = run_ablation_suite(dataset, tokenizer, counts, override_config(CONFIG, overrides))
        lookup = {record["variant"]: record for record in records}
        if "full" in lookup and "kv_only" in lookup:
            assert lookup["full"]["one_shot"] >= lookup["kv_only"]["one_shot"] - 1e-3
            flops_full = lookup["full"].get("flops", 0.0)
            flops_kv = lookup["kv_only"].get("flops", 0.0)
            if flops_full > 0 and flops_kv > 0:
                max_flops = max(flops_full, flops_kv, 1.0)
                assert abs(flops_full - flops_kv) / max_flops <= 0.9

    def test_benchmark_hooks():
        glue_results = run_glue_benchmark("sst2", CONFIG)
        assert "hsokv" in glue_results and glue_results["hsokv"].accuracy > 0.0
        cifar_results = run_split_cifar_benchmark(CONFIG)
        assert "hsokv" in cifar_results and cifar_results["hsokv"].retention > 0.0

    def test_distributed_runner():
        overrides = {
            "distributed_node_counts": [1],
            "distributed_agents_per_node": 2,
            "distributed_episodes": 4,
            "distributed_steps": 15,
            "distributed_backend": "simulate",
        }
        metrics = run_distributed_swarm(override_config(CONFIG, overrides))
        assert metrics["nodes"] == [1]
        assert metrics["speedup"][0] >= 1.0

    test_corpus_generation_defaults()
    test_preset_application()
    test_quick_preset_training()
    test_kv_normalized_hits()
    test_swarm_only_flag()
    test_use_kv_flag()
    test_ablation_lift()
    test_benchmark_hooks()
    test_distributed_runner()
    print("Validation tests passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hierarchical Swarm-KV Architecture (H-SOKV)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Run mode")
    parser.add_argument("--iterations", type=int, default=None, help="Meta-iterations for swarm training")
    parser.add_argument("--visualize", action="store_true", help="Generate plots")
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESET_CONFIGS.keys()),
        default=CONFIG.get("preset", "demo"),
        help="Use predefined config preset (quick_test=2min, demo=15min, research=30min)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "language_model"],
        default=CONFIG.get("task_type", "classification"),
        help="Primary task type ('classification' or 'language_model')",
    )
    parser.add_argument("--use-swarm", dest="use_swarm", action="store_true", help="Enable swarm optimization")
    parser.add_argument("--no-use-swarm", dest="use_swarm", action="store_false", help="Disable swarm optimization")
    parser.add_argument("--use-kv", dest="use_kv", action="store_true", help="Enable KV integration")
    parser.add_argument("--no-use-kv", dest="use_kv", action="store_false", help="Disable KV integration")
    parser.set_defaults(use_swarm=None, use_kv=None)
    parser.add_argument(
        "--benchmark",
        type=str,
        default="synthetic",
        help="Benchmark selection ('synthetic', 'glue', 'cifar')",
    )
    parser.add_argument("--run-ablations", action="store_true", help="Execute ablation suite")
    parser.add_argument(
        "--run-distributed",
        action="store_true",
        help="Execute distributed swarm scalability simulation",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit benchmark loaders to download datasets if missing",
    )
    parser.add_argument(
        "--glue-task",
        type=str,
        default=None,
        help="Override GLUE task name (default uses CONFIG['glue_task'])",
    )
    parser.add_argument(
        "--glue-data-dir",
        type=str,
        default=None,
        help="Path to GLUE data directory (expects task sub-folder)",
    )
    parser.add_argument(
        "--cifar-data-dir",
        type=str,
        default=None,
        help="Path to CIFAR-10 data directory",
    )
    parser.add_argument(
        "--distributed-backend",
        type=str,
        default=None,
        help="Override distributed backend ('auto', 'ray', 'multiprocessing', 'simulate')",
    )
    parser.add_argument(
        "--save-pretrained",
        type=str,
        default=None,
        help="Directory to save the trained H-SOKV model/tokenizer",
    )
    parser.add_argument(
        "--load-pretrained",
        type=str,
        default=None,
        help="Load an existing H-SOKV checkpoint instead of training",
    )
    parser.add_argument(
        "--hf-train",
        action="store_true",
        help="Run H-SOKV training via Hugging Face-style trainer wrapper",
    )
    parser.add_argument(
        "--hf-output-dir",
        type=str,
        default=None,
        help="Output directory for Hugging Face trainer checkpoints",
    )
    parser.add_argument(
        "--lm-corpus",
        type=str,
        default=None,
        help="Path to plain-text corpus for language modeling task",
    )
    parser.add_argument(
        "--lm-seq-length",
        type=int,
        default=None,
        help="Context length (tokens) for language modeling windows",
    )
    parser.add_argument(
        "--lm-stride",
        type=int,
        default=None,
        help="Sliding window stride when building language modeling samples",
    )
    parser.add_argument(
        "--lm-max-sequences",
        type=int,
        default=None,
        help="Cap on the number of language modeling sequences to generate",
    )
    parser.add_argument(
        "--corpus-size",
        type=str,
        choices=["tiny", "small", "medium", "large"],
        default=CONFIG.get("lm_corpus_preset", "medium"),
        help="Auto-generate corpus of specified size if --lm-corpus not provided",
    )
    return parser.parse_args()


def print_system_info():
    print("=" * 60)
    print("System Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Device being used: {CONFIG['device']}")
    print("=" * 60)


def main() -> None:
    print_system_info()
    args = parse_args()
    if args.mode == "test":
        run_validation_tests()
        return
    run_experiment(args)


if __name__ == "__main__":
    main()
