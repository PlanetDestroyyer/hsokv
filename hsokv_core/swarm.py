"""Hierarchical swarm components."""

import itertools
import random
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .config import CONFIG
from .memory import KeyValueMemory
from .model import TransformerWithKV
from .utils import compute_swarm_diversity, compute_usage_correctness, move_batch_to_device


@dataclass
class AgentConfig:
    strategy: str
    learning_rate: float
    kv_retrieval_k: int


class Agent:
    def __init__(
        self,
        agent_id: int,
        tokenizer,
        config: Dict[str, object],
        model_factory,
        device: torch.device,
        strategy: Optional[str] = None,
    ) -> None:
        self.agent_id = agent_id
        self.tokenizer = tokenizer
        self.config = config
        self.model_factory = model_factory
        self.device = device
        strategies = ["sgd", "adam", "rmsprop", "random_search"]
        self.strategy = strategy or random.choice(strategies)
        self.learning_rate = random.uniform(*config["learning_rate_range"])
        self.kv_retrieval_k = random.randint(*config["kv_top_k_range"])
        self.steps = config["agent_steps"]
        self.encoded_cache: Dict[str, torch.Tensor] = {}

    def adjust_strategy(self, strategy: str, learning_rate: Optional[float] = None, kv_k: Optional[int] = None) -> None:
        self.strategy = strategy
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if kv_k is not None:
            self.kv_retrieval_k = kv_k

    def train_episode(self, task_data: Dict[str, object]) -> Dict[str, object]:
        model: TransformerWithKV = self.model_factory()
        model.load_state_dict(task_data["base_state"])
        if task_data.get("kv_state"):
            model.kv_memory.load_state(task_data["kv_state"])
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = self._build_optimizer(model)
        train_loader = task_data["train_loader"]
        iterator = itertools.cycle(train_loader)
        loss_curve: List[float] = []
        kv_hits: List[float] = []
        usage_curve: List[float] = []
        gate_entropy_curve: List[float] = []
        gate_mean_curve: List[float] = []
        start_time = time.time()
        for _ in range(self.steps):
            batch = next(iterator)
            batch = move_batch_to_device(batch, self.device)
            model.train()
            logits, info = model(batch["input_ids"], batch["attention_mask"], top_k=self.kv_retrieval_k)
            loss = criterion(logits, batch["labels"])
            gate_values = info["gate_values"]
            if gate_values.numel() > 0:
                gv = gate_values.clamp(1e-6, 1 - 1e-6)
                entropy = -(gv * torch.log2(gv) + (1 - gv) * torch.log2(1 - gv))
                gate_entropy_curve.append(float(entropy.mean().item()))
                gate_mean_curve.append(float(gate_values.mean().item()))
            if self.strategy != "random_search":
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            else:
                with torch.no_grad():
                    noise_scale = 0.01
                    for param in model.parameters():
                        param.add_(noise_scale * torch.randn_like(param))
            loss_curve.append(loss.item())
            kv_hits.append(info["kv_details"]["avg_similarity"])
            usage_curve.append(compute_usage_correctness(logits.argmax(dim=-1), batch["labels"], info["gate_values"]))
            if self.config.get("use_kv", True):
                self._update_memory(model, batch, info["pooled"])
            if len(model.kv_memory) > self.config["max_memory_entries"]:
                model.kv_memory.prune(self.config["kv_confidence_threshold"])
        val_metrics = task_data["evaluate_model"](model, task_data["val_loader"], self.device, self.kv_retrieval_k, task_data["one_shot_ids"])
        retention_score = task_data["evaluate_retention"](model, task_data["retention_loader"], self.device)
        elapsed = time.time() - start_time
        return {
            "agent_id": self.agent_id,
            "strategy": self.strategy,
            "learning_rate": self.learning_rate,
            "kv_k": self.kv_retrieval_k,
            "loss_curve": loss_curve,
            "usage_curve": usage_curve,
            "kv_hits": kv_hits,
            "gate_entropy": float(np.mean(gate_entropy_curve)) if gate_entropy_curve else 0.0,
            "avg_gate": float(np.mean(gate_mean_curve)) if gate_mean_curve else 0.0,
            "val_accuracy": val_metrics["accuracy"],
            "one_shot_accuracy": val_metrics["one_shot_accuracy"],
            "kv_hit_rate": val_metrics["kv_hit_rate"],
            "usage": val_metrics["usage"],
            "retention": retention_score,
            "convergence": task_data["compute_convergence_step"]([val_metrics["accuracy"]]),
            "model_state": deepcopy(model.state_dict()),
            "kv_state": model.kv_memory.get_state(),
            "elapsed": elapsed,
        }

    def _build_optimizer(self, model: TransformerWithKV):
        params = [p for p in model.parameters() if p.requires_grad]
        if self.strategy == "sgd":
            return torch.optim.SGD(params, lr=self.learning_rate, momentum=0.9)
        if self.strategy == "adam":
            return torch.optim.Adam(params, lr=self.learning_rate, betas=(0.9, 0.99))
        if self.strategy == "rmsprop":
            return torch.optim.RMSprop(params, lr=self.learning_rate, momentum=0.9)
        return torch.optim.Adam(params, lr=self.learning_rate)

    def _update_memory(self, model: TransformerWithKV, batch: Dict[str, object], pooled: torch.Tensor) -> None:
        with torch.no_grad():
            seen_in_batch = set()
            for vector, definition, usage, rare_word in zip(
                pooled,
                batch["definitions"],
                batch["usages"],
                batch["rare_words"],
            ):
                if not rare_word or rare_word in seen_in_batch:
                    continue
                seen_in_batch.add(rare_word)
                story_hash = hash((rare_word, definition, usage))
                if any(meta["story_hash"] == story_hash for meta in model.kv_memory.metadata):
                    continue
                cache_key = rare_word
                if cache_key not in self.encoded_cache:
                    self.encoded_cache[cache_key] = model.encode_text(
                        definition + " " + usage, self.config["definition_max_length"]
                    ).detach()
                value_vector = self.encoded_cache[cache_key]
                model.kv_memory.write(
                    key_embedding=vector,
                    value_dict={"word": rare_word, "definition": definition, "usage": usage, "value_vector": value_vector},
                    metadata={"confidence": 0.25, "retrieval_count": 0, "success_rate": 0.0, "story_hash": story_hash},
                )


class Manager:
    def __init__(self, manager_id: int, agents: List[Agent]) -> None:
        self.manager_id = manager_id
        self.agents = agents
        self.meta_memory: Dict[str, AgentConfig] = {}
        self.performance_history: List[Dict[str, object]] = []

    def coordinate_agents(self, task_data: Dict[str, object], task_context: str) -> Dict[str, object]:
        results = []
        for agent in self.agents:
            outcome = agent.train_episode(task_data)
            results.append(outcome)
        best = max(results, key=lambda res: res["val_accuracy"])
        self.meta_memory[task_context] = AgentConfig(best["strategy"], best["learning_rate"], best["kv_k"])
        threshold = max(0.0, best["val_accuracy"] * 0.6)
        for agent, res in zip(self.agents, results):
            if res["val_accuracy"] < threshold:
                suggested = self.meta_memory.get(task_context)
                if suggested:
                    agent.adjust_strategy(
                        strategy=suggested.strategy,
                        learning_rate=suggested.learning_rate * random.uniform(0.8, 1.2),
                        kv_k=min(max(int(suggested.kv_retrieval_k + random.randint(-1, 1)), 1), 10),
                    )
                else:
                    agent.adjust_strategy(random.choice(["sgd", "adam", "rmsprop"]))
        self.performance_history.append({"context": task_context, "results": results})
        return {"manager_id": self.manager_id, "results": results, "best": best}


class Supervisor:
    def __init__(
        self,
        model_factory,
        tokenizer,
        config: Dict[str, object],
        device: torch.device,
        base_state: Optional[Dict[str, torch.Tensor]] = None,
        kv_state: Optional[Dict[str, object]] = None,
    ) -> None:
        self.model_factory = model_factory
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.managers: List[Manager] = []
        self.global_memory: Dict[str, object] = {"history": [], "strategy_counts": []}
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_kv_state: Optional[Dict[str, object]] = None
        self.best_score = -float("inf")
        strategies = ["sgd", "adam", "rmsprop", "random_search"]
        agent_count = 0
        for manager_id in range(config["num_managers"]):
            agents: List[Agent] = []
            for idx in range(config["agents_per_manager"]):
                strategy = strategies[agent_count % len(strategies)]
                agent = Agent(
                    agent_id=manager_id * config["agents_per_manager"] + idx,
                    tokenizer=tokenizer,
                    config=config,
                    model_factory=model_factory,
                    device=device,
                    strategy=strategy,
                )
                agents.append(agent)
                agent_count += 1
            self.managers.append(Manager(manager_id, agents))
        base_model = self.model_factory()
        if base_state:
            base_model.load_state_dict(base_state)
        if kv_state:
            base_model.kv_memory.load_state(kv_state)
        self.base_state = deepcopy(base_model.state_dict())
        self.best_state = deepcopy(self.base_state)
        self.best_kv_state = deepcopy(base_model.kv_memory.get_state()) if len(base_model.kv_memory) else None
        if kv_state:
            self.best_kv_state = deepcopy(kv_state)
        del base_model

    def run_meta_iteration(self, task_data: Dict[str, object], iteration: int) -> Dict[str, object]:
        task_context = "low_support" if iteration % 2 == 0 else "mixed_support"
        task_data = dict(task_data)
        task_data["base_state"] = deepcopy(self.base_state)
        task_data["kv_state"] = deepcopy(self.best_kv_state) if self.best_kv_state else None
        manager_outputs = []
        for manager in self.managers:
            output = manager.coordinate_agents(task_data, task_context)
            manager_outputs.append(output)
        best_overall = max((output["best"] for output in manager_outputs), key=lambda res: res["val_accuracy"])
        if best_overall["val_accuracy"] > self.best_score:
            self.best_score = best_overall["val_accuracy"]
            self.best_state = deepcopy(best_overall["model_state"])
            self.best_kv_state = deepcopy(best_overall["kv_state"])
            self.base_state = deepcopy(best_overall["model_state"])
        strategy_counter = Counter(res["strategy"] for output in manager_outputs for res in output["results"])
        avg_loss = float(np.mean(best_overall["loss_curve"])) if best_overall["loss_curve"] else 0.0
        diversity = compute_swarm_diversity(strategy_counter)
        if iteration == 0 and diversity < 0.8:
            print(f"[Warning] Swarm diversity below target: {diversity:.3f}")
        gate_entropy = best_overall.get("gate_entropy", 0.0)
        regret = max(0.0, 1.0 - best_overall["val_accuracy"])
        self.global_memory["history"].append(
            {
                "iteration": iteration,
                "avg_loss": avg_loss,
                "val_accuracy": best_overall["val_accuracy"],
                "kv_hit_rate": best_overall["kv_hit_rate"],
                "retention": best_overall["retention"],
                "usage": best_overall["usage"],
                "swarm_diversity": diversity,
                "gate_entropy": gate_entropy,
                "regret": regret,
            }
        )
        self.global_memory["strategy_counts"].append(strategy_counter)
        return {
            "iteration": iteration,
            "manager_outputs": manager_outputs,
            "best": best_overall,
            "strategy_counts": strategy_counter,
        }

    def get_best_model(self) -> TransformerWithKV:
        model = self.model_factory()
        if self.best_state:
            model.load_state_dict(self.best_state)
        if self.best_kv_state:
            model.kv_memory.load_state(self.best_kv_state)
        model.to(self.device)
        return model
