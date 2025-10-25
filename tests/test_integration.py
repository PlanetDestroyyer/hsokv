"""Integration tests for the H-SOKV pipeline."""

from __future__ import annotations

import math

import torch
import pytest

from hsokv_core.config import CONFIG, override_config
from hsokv_core.consolidation import ConsolidationModule
from hsokv_core.context_retrieval import ContextualRetrievalModule
from hsokv_core.data import generate_dataset
from hsokv_core.forgetting import ForgettingModule
from hsokv_core.memory import KeyValueMemory
from hsokv_core.model import TransformerWithKV
from hsokv_core.surprise_writing import SurpriseBasedWriter
from hsokv_core.training import train_hsokv
from hsokv_core.utils import set_seed


@pytest.fixture(scope="module")
def tiny_config():
    return override_config(
        CONFIG,
        {
            "device": "cpu",
            "meta_iterations": 1,
            "agents_per_manager": 1,
            "agent_steps": 4,
            "batch_size": 4,
            "flops_target": 5e5,
            "max_memory_entries": 64,
            "forgetting_interval": 2,
            "use_consolidation": True,
            "use_forgetting": True,
        },
    )


def _build_model(tokenizer, config):
    return TransformerWithKV(len(tokenizer.vocab), len(tokenizer.vocab), tokenizer, config)


def test_consolidation_forgetting_pipeline(tiny_config):
    set_seed(0)
    dataset, tokenizer, word_counts = generate_dataset()
    model = _build_model(tokenizer, tiny_config)
    memory = model.kv_memory
    pooled = torch.randn(tiny_config["d_model"])
    value_vec = torch.randn(tiny_config["d_model"])
    for idx in range(20):
        memory.write(
            pooled + 0.01 * idx,
            {"word": f"word_{idx}", "definition": "def", "usage": "usage", "value_vector": value_vec},
            {"confidence": 0.9, "retrieval_count": 15, "success_rate": 0.8, "created_at": float(idx)},
        )
    consolidator = ConsolidationModule(model, tiny_config, tokenizer)
    metrics = consolidator.consolidate()
    assert metrics.consolidated_count >= 0
    forgetter = ForgettingModule(
        memory,
        memory_cap=tiny_config["max_memory_entries"],
        confidence_threshold=tiny_config["kv_confidence_threshold"],
        trigger_interval=1,
    )
    if forgetter.should_forget(1):
        report = forgetter.forget(1, current_step=1.0)
        assert report.memory_size_after >= 0


def test_context_aware_surprise_writing(tiny_config):
    set_seed(1)
    dataset, tokenizer, _ = generate_dataset()
    model = _build_model(tokenizer, tiny_config)
    writer = SurpriseBasedWriter(tiny_config)
    memory = model.kv_memory
    pooled = torch.randn(1, tiny_config["d_model"])
    logits = torch.randn(1, len(tokenizer.vocab))
    cache = {}
    stats = writer.selective_write(
        model=model,
        memory=memory,
        batch={
            "labels": torch.zeros(1, dtype=torch.long),
            "rare_words": ["example"],
            "definitions": ["definition"],
            "usages": ["usage"],
        },
        pooled=pooled,
        logits=logits,
        cache=cache,
    )
    assert stats["writes"] >= 0
    context_module = ContextualRetrievalModule(tiny_config)
    signals = context_module.extract_context_signals(torch.randn(1, 2, tiny_config["d_model"]), torch.randn(1, tiny_config["d_model"]), 1)
    assert signals and isinstance(signals, list)


def test_full_training_pipeline(tiny_config):
    set_seed(2)
    dataset, tokenizer, word_counts = generate_dataset()
    _, summary = train_hsokv(dataset, tokenizer, word_counts, tiny_config)
    assert "test_metrics" in summary
    accuracy = summary["test_metrics"]["accuracy"]
    assert 0.0 <= accuracy <= 1.0


def test_edge_cases(tiny_config):
    memory = KeyValueMemory(tiny_config["d_model"], torch.device("cpu"))
    forgetter = ForgettingModule(
        memory,
        memory_cap=tiny_config["max_memory_entries"],
        confidence_threshold=tiny_config["kv_confidence_threshold"],
    )
    assert not forgetter.should_forget(0)
    report = forgetter.forget(0, current_step=0.0)
    assert report.forgotten_count == 0


def test_device_compatibility(tiny_config):
    dataset, tokenizer, word_counts = generate_dataset()
    config = override_config(tiny_config, {"device": "cpu"})
    _, summary = train_hsokv(dataset, tokenizer, word_counts, config)
    assert summary["test_metrics"]["accuracy"] >= 0.0


def test_memory_lifecycle(tiny_config):
    dataset, tokenizer, word_counts = generate_dataset()
    _, summary = train_hsokv(dataset, tokenizer, word_counts, tiny_config)
    history = summary["history"]
    assert history
    last_entry = history[-1]
    assert "retention" in last_entry


def test_performance_assertions(tiny_config):
    dataset, tokenizer, word_counts = generate_dataset()
    _, summary = train_hsokv(dataset, tokenizer, word_counts, tiny_config)
    acc = summary["test_metrics"]["accuracy"]
    retention = summary["retention"]
    assert retention >= 0.0
    assert math.isfinite(acc)
