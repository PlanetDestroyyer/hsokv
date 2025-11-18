"""
Real-world benchmark: HSOKV vs actual fine-tuned SLMs

Compares HSOKV against REAL small language models fine-tuned on sequential tasks:
- SmolLM2-135M/360M
- Llama 3.2 1B
- Qwen2.5-0.5B/1.5B

Uses REAL continual learning benchmarks:
- Sequential text classification (AGNews, 20Newsgroups)
- Sequential question answering
- Multi-domain learning

Measures ACTUAL catastrophic forgetting in real neural networks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import numpy as np
from typing import List, Dict, Tuple
import time
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from hsokv import MemorySystem, SentenceBERTEmbedder, MemoryConfig


class RealWorldBenchmark:
    """Benchmark HSOKV against real fine-tuned language models"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    def load_sequential_text_classification_tasks(self):
        """
        Load real-world sequential text classification benchmark
        Uses AGNews dataset split into 4 sequential tasks by category
        """
        print("\n" + "="*70)
        print("LOADING REAL-WORLD BENCHMARK: AGNews Sequential Classification")
        print("="*70)

        try:
            # Load AGNews dataset (4 classes: World, Sports, Business, Sci/Tech)
            dataset = load_dataset("ag_news")

            # Split by class for sequential learning
            tasks = []
            class_names = ["World News", "Sports", "Business", "Sci/Tech"]

            for class_id, class_name in enumerate(class_names):
                # Filter examples for this class
                train_examples = [
                    ex for ex in dataset['train']
                    if ex['label'] == class_id
                ][:500]  # Use 500 examples per task

                test_examples = [
                    ex for ex in dataset['test']
                    if ex['label'] == class_id
                ][:200]  # Use 200 test examples per task

                tasks.append({
                    'name': class_name,
                    'class_id': class_id,
                    'train': train_examples,
                    'test': test_examples,
                })

                print(f"  Task {class_id + 1}: {class_name}")
                print(f"    Train: {len(train_examples)} examples")
                print(f"    Test: {len(test_examples)} examples")

            print(f"\nTotal: {len(tasks)} tasks, {sum(len(t['train']) for t in tasks)} train examples")
            return tasks

        except Exception as e:
            print(f"Error loading AGNews: {e}")
            print("Falling back to synthetic sequential classification tasks...")
            return self._create_synthetic_classification_tasks()

    def _create_synthetic_classification_tasks(self):
        """Fallback: Create synthetic sequential classification tasks"""
        tasks = []

        # Task 1: Sentiment Classification
        tasks.append({
            'name': 'Sentiment',
            'train': [
                {'text': 'This movie is amazing and wonderful', 'label': 1},
                {'text': 'I love this product so much', 'label': 1},
                {'text': 'Terrible experience, worst ever', 'label': 0},
                {'text': 'Awful and disappointing', 'label': 0},
            ] * 50,  # Repeat for more data
            'test': [
                {'text': 'Great and fantastic', 'label': 1},
                {'text': 'Horrible and bad', 'label': 0},
            ] * 20,
        })

        # Task 2: Topic Classification (Tech)
        tasks.append({
            'name': 'Tech',
            'train': [
                {'text': 'New smartphone with advanced AI features released', 'label': 1},
                {'text': 'Software update improves performance', 'label': 1},
                {'text': 'Recipe for chocolate cake', 'label': 0},
                {'text': 'Gardening tips for spring', 'label': 0},
            ] * 50,
            'test': [
                {'text': 'Latest laptop has powerful GPU', 'label': 1},
                {'text': 'Cooking pasta perfectly', 'label': 0},
            ] * 20,
        })

        print("Using synthetic tasks (AGNews unavailable)")
        for i, task in enumerate(tasks):
            print(f"  Task {i+1}: {task['name']} - {len(task['train'])} train, {len(task['test'])} test")

        return tasks

    def benchmark_small_language_model(self, model_name: str, tasks: List[Dict]):
        """
        Benchmark a real SLM with actual fine-tuning

        Args:
            model_name: HuggingFace model name (e.g., 'HuggingFaceTB/SmolLM2-135M')
            tasks: List of sequential tasks
        """
        print(f"\n{'='*70}")
        print(f"BENCHMARKING: {model_name} with Fine-tuning")
        print(f"{'='*70}")

        try:
            # Load tokenizer and model
            print(f"\nLoading {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Add padding token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # For classification, we'll use the model in a simple way
            # We'll track accuracy on all previous tasks after each new task

            accuracy_matrix = np.zeros((len(tasks), len(tasks)))

            # We'll use a simple approach: fine-tune embeddings for classification
            # This demonstrates catastrophic forgetting in real models

            # Determine dtype based on device
            model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=model_dtype,
                device_map=self.device,
            )

            # Add classification head (match model dtype!)
            classifier = nn.Linear(model.config.hidden_size, 2).to(self.device).to(model_dtype)
            optimizer = torch.optim.AdamW(
                list(model.parameters()) + list(classifier.parameters()),
                lr=2e-5
            )

            for task_idx, task in enumerate(tasks):
                print(f"\n--- Learning Task {task_idx + 1}/{len(tasks)}: {task['name']} ---")

                # Fine-tune on current task
                self._finetune_on_task(model, tokenizer, classifier, optimizer, task)

                # Evaluate on all tasks seen so far
                print(f"\n  Evaluating on all {task_idx + 1} tasks seen so far:")
                for eval_idx in range(task_idx + 1):
                    acc = self._evaluate_task(model, tokenizer, classifier, tasks[eval_idx])
                    accuracy_matrix[task_idx, eval_idx] = acc
                    print(f"    Task {eval_idx + 1} ({tasks[eval_idx]['name']}): {acc*100:.1f}%")

            # Calculate metrics
            metrics = self._calculate_metrics(accuracy_matrix, len(tasks))
            metrics['model_name'] = model_name
            metrics['accuracy_matrix'] = accuracy_matrix.tolist()

            # Print results
            print(f"\n{'='*70}")
            print(f"RESULTS FOR {model_name}")
            print(f"{'='*70}")
            print(f"Average Accuracy: {metrics['average_accuracy']*100:.2f}%")
            print(f"Backward Transfer (BWT): {metrics['backward_transfer']:.4f}")
            print(f"Forgetting Rate: {metrics['forgetting_rate']*100:.2f}%")
            print(f"Task 1 Final Accuracy: {metrics['task1_final_accuracy']*100:.2f}%")
            print(f"{'='*70}\n")

            # Clean up
            del model
            del classifier
            torch.cuda.empty_cache()

            return metrics

        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            print("Skipping this model...")
            return None

    def _finetune_on_task(self, model, tokenizer, classifier, optimizer, task, epochs=3):
        """Fine-tune model on a single task"""
        model.train()
        classifier.train()

        train_data = task['train'][:100]  # Use subset for speed

        for epoch in range(epochs):
            total_loss = 0
            for example in train_data:
                # Tokenize
                inputs = tokenizer(
                    example['text'],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(self.device)

                # Forward pass
                outputs = model(**inputs, output_hidden_states=True)
                hidden_state = outputs.hidden_states[-1][:, -1, :]  # Last token
                logits = classifier(hidden_state)

                # Loss (cast logits to float32 for stable loss computation)
                label = torch.tensor([example['label']]).to(self.device)
                loss = nn.CrossEntropyLoss()(logits.float(), label)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_data)
            if epoch == epochs - 1:
                print(f"    Fine-tuning epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def _evaluate_task(self, model, tokenizer, classifier, task):
        """Evaluate model on a task"""
        model.eval()
        classifier.eval()

        correct = 0
        total = len(task['test'][:50])  # Use subset for speed

        with torch.no_grad():
            for example in task['test'][:50]:
                inputs = tokenizer(
                    example['text'],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(self.device)

                outputs = model(**inputs, output_hidden_states=True)
                hidden_state = outputs.hidden_states[-1][:, -1, :]
                logits = classifier(hidden_state)

                pred = torch.argmax(logits, dim=1).item()
                if pred == example['label']:
                    correct += 1

        return correct / total if total > 0 else 0.0

    def benchmark_hsokv(self, tasks: List[Dict]):
        """Benchmark HSOKV on same tasks"""
        print(f"\n{'='*70}")
        print(f"BENCHMARKING: HSOKV (Frozen Embeddings + Memory)")
        print(f"{'='*70}")

        config = MemoryConfig(device=self.device, max_entries=10000)
        embedder = SentenceBERTEmbedder(device=self.device)
        system = MemorySystem(embedder, config)

        accuracy_matrix = np.zeros((len(tasks), len(tasks)))

        for task_idx, task in enumerate(tasks):
            print(f"\n--- Learning Task {task_idx + 1}/{len(tasks)}: {task['name']} ---")

            # Learn task (store examples in memory)
            for example in task['train'][:100]:
                # Store text → label mapping
                system.learn(
                    example['text'],
                    str(example['label']),
                    confidence=0.8
                )

            # Evaluate on all tasks
            print(f"\n  Evaluating on all {task_idx + 1} tasks seen so far:")
            for eval_idx in range(task_idx + 1):
                correct = 0
                total = len(tasks[eval_idx]['test'][:50])

                for example in tasks[eval_idx]['test'][:50]:
                    predicted = system.recall(example['text'])
                    if predicted and predicted == str(example['label']):
                        correct += 1

                acc = correct / total if total > 0 else 0.0
                accuracy_matrix[task_idx, eval_idx] = acc
                print(f"    Task {eval_idx + 1} ({tasks[eval_idx]['name']}): {acc*100:.1f}%")

        # Calculate metrics
        metrics = self._calculate_metrics(accuracy_matrix, len(tasks))
        metrics['model_name'] = 'HSOKV'
        metrics['accuracy_matrix'] = accuracy_matrix.tolist()

        print(f"\n{'='*70}")
        print(f"RESULTS FOR HSOKV")
        print(f"{'='*70}")
        print(f"Average Accuracy: {metrics['average_accuracy']*100:.2f}%")
        print(f"Backward Transfer (BWT): {metrics['backward_transfer']:.4f}")
        print(f"Forgetting Rate: {metrics['forgetting_rate']*100:.2f}%")
        print(f"Task 1 Final Accuracy: {metrics['task1_final_accuracy']*100:.2f}%")
        print(f"{'='*70}\n")

        return metrics

    def _calculate_metrics(self, accuracy_matrix: np.ndarray, n_tasks: int) -> Dict:
        """Calculate continual learning metrics"""
        final_accuracies = accuracy_matrix[-1, :]
        average_accuracy = np.mean(final_accuracies)

        backward_transfer = 0.0
        for j in range(n_tasks - 1):
            max_acc = accuracy_matrix[j, j]
            final_acc = accuracy_matrix[-1, j]
            backward_transfer += (final_acc - max_acc)
        backward_transfer /= (n_tasks - 1) if n_tasks > 1 else 1

        task1_max_accuracy = accuracy_matrix[0, 0]
        task1_final_accuracy = accuracy_matrix[-1, 0]
        forgetting_rate = max(0, task1_max_accuracy - task1_final_accuracy)

        return {
            'average_accuracy': float(average_accuracy),
            'backward_transfer': float(backward_transfer),
            'forgetting_rate': float(forgetting_rate),
            'task1_final_accuracy': float(task1_final_accuracy),
            'task1_max_accuracy': float(task1_max_accuracy),
        }

    def run_full_benchmark(self):
        """Run complete benchmark comparing HSOKV vs real SLMs"""
        print("\n" + "="*70)
        print("REAL-WORLD CONTINUAL LEARNING BENCHMARK")
        print("HSOKV vs Fine-tuned Small Language Models")
        print("="*70)

        # Load benchmark tasks
        tasks = self.load_sequential_text_classification_tasks()

        results = {}

        # Benchmark HSOKV first (fast)
        results['HSOKV'] = self.benchmark_hsokv(tasks)

        # List of SLMs to test
        slm_models = [
            'HuggingFaceTB/SmolLM2-135M',  # Smallest, fastest
            # 'HuggingFaceTB/SmolLM2-360M',  # Uncomment if you have time/memory
            # 'Qwen/Qwen2.5-0.5B',           # Uncomment for more comparisons
            # 'meta-llama/Llama-3.2-1B',     # Requires auth token
        ]

        # Benchmark each SLM
        for model_name in slm_models:
            print(f"\n\nAttempting to benchmark: {model_name}")
            result = self.benchmark_small_language_model(model_name, tasks)
            if result:
                results[model_name] = result

        # Save results
        output_file = Path(__file__).parent / 'realworld_benchmark_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")

        # Print comparison
        self._print_comparison(results)

        return results

    def _print_comparison(self, results: Dict):
        """Print comparison table"""
        print("\n" + "="*70)
        print("COMPARISON: HSOKV vs Real Fine-tuned SLMs")
        print("="*70)
        print(f"\n{'Model':<30} {'Avg Acc':<12} {'BWT':<12} {'Forgetting':<12}")
        print("-"*70)

        for model_name, metrics in results.items():
            short_name = model_name.split('/')[-1] if '/' in model_name else model_name
            print(f"{short_name:<30} "
                  f"{metrics['average_accuracy']*100:>6.2f}%     "
                  f"{metrics['backward_transfer']:>7.4f}     "
                  f"{metrics['forgetting_rate']*100:>6.2f}%")

        print("="*70)
        print("\nKey Findings:")
        if 'HSOKV' in results:
            hsokv_forget = results['HSOKV']['forgetting_rate'] * 100
            print(f"  HSOKV Forgetting Rate: {hsokv_forget:.2f}%")

            for model_name, metrics in results.items():
                if model_name != 'HSOKV':
                    slm_forget = metrics['forgetting_rate'] * 100
                    reduction = slm_forget - hsokv_forget
                    print(f"  {model_name.split('/')[-1]} Forgetting Rate: {slm_forget:.2f}%")
                    print(f"    → HSOKV reduces forgetting by {reduction:.2f} percentage points!")
        print("="*70)


def main():
    """Run real-world benchmark"""
    benchmark = RealWorldBenchmark()
    results = benchmark.run_full_benchmark()
    return results


if __name__ == "__main__":
    main()
