"""
Visualize benchmark results comparing HSOKV vs traditional fine-tuning

Creates graphs showing:
1. Catastrophic forgetting over time
2. Task-by-task accuracy comparison
3. Backward transfer heatmap
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_results(results_file='benchmark_results.json'):
    """Load benchmark results from JSON"""
    results_path = Path(__file__).parent / results_file

    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}\n"
            f"Run benchmark_catastrophic_forgetting.py first!"
        )

    with open(results_path, 'r') as f:
        return json.load(f)


def plot_catastrophic_forgetting(results, save_path='forgetting_comparison.png'):
    """
    Plot how Task 1 accuracy changes as new tasks are learned
    This clearly shows catastrophic forgetting
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ft_matrix = np.array(results['finetuning']['accuracy_matrix'])
    hsokv_matrix = np.array(results['hsokv']['accuracy_matrix'])
    task_names = results['tasks']

    # Extract Task 1 accuracy over time
    ft_task1_over_time = ft_matrix[:, 0] * 100  # First column = Task 1
    hsokv_task1_over_time = hsokv_matrix[:, 0] * 100

    x = range(1, len(task_names) + 1)

    # Plot lines
    ax.plot(x, ft_task1_over_time, marker='o', linewidth=2.5,
            label='Traditional Fine-tuning', color='#e74c3c', markersize=8)
    ax.plot(x, hsokv_task1_over_time, marker='s', linewidth=2.5,
            label='HSOKV (Ours)', color='#2ecc71', markersize=8)

    # Styling
    ax.set_xlabel('Tasks Learned Sequentially', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Accuracy on Task 1 ({task_names[0]}) %', fontsize=12, fontweight='bold')
    ax.set_title('Catastrophic Forgetting: Task 1 Performance Over Time',
                fontsize=14, fontweight='bold', pad=20)

    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels([f'After\nTask {i}' for i in x])

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)

    # Legend
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)

    # Annotations
    final_ft = ft_task1_over_time[-1]
    final_hsokv = hsokv_task1_over_time[-1]
    forgetting_ft = ft_task1_over_time[0] - final_ft
    forgetting_hsokv = hsokv_task1_over_time[0] - final_hsokv

    # Add text box with key metrics
    textstr = f'Final Accuracy on Task 1:\n' \
              f'Fine-tuning: {final_ft:.1f}%\n' \
              f'HSOKV: {final_hsokv:.1f}%\n\n' \
              f'Forgetting:\n' \
              f'Fine-tuning: {forgetting_ft:.1f}%\n' \
              f'HSOKV: {forgetting_hsokv:.1f}%'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.35, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    output_path = Path(__file__).parent / save_path
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_all_tasks_comparison(results, save_path='all_tasks_comparison.png'):
    """
    Bar chart comparing final accuracy on all tasks
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ft_matrix = np.array(results['finetuning']['accuracy_matrix'])
    hsokv_matrix = np.array(results['hsokv']['accuracy_matrix'])
    task_names = results['tasks']

    # Final accuracy on each task
    ft_final = ft_matrix[-1, :] * 100
    hsokv_final = hsokv_matrix[-1, :] * 100

    x = np.arange(len(task_names))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, ft_final, width, label='Traditional Fine-tuning',
                   color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, hsokv_final, width, label='HSOKV (Ours)',
                   color='#2ecc71', alpha=0.8)

    # Styling
    ax.set_xlabel('Tasks', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Final Accuracy on All Tasks After Sequential Learning',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)

    add_value_labels(bars1)
    add_value_labels(bars2)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    output_path = Path(__file__).parent / save_path
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_accuracy_heatmap(results, save_path='accuracy_heatmap.png'):
    """
    Heatmap showing accuracy matrix over time
    Rows = after learning task i
    Cols = accuracy on task j
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ft_matrix = np.array(results['finetuning']['accuracy_matrix']) * 100
    hsokv_matrix = np.array(results['hsokv']['accuracy_matrix']) * 100
    task_names = results['tasks']

    # Fine-tuning heatmap
    sns.heatmap(ft_matrix, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=task_names, yticklabels=[f'After Task {i+1}' for i in range(len(task_names))],
                cbar_kws={'label': 'Accuracy (%)'}, ax=ax1, vmin=0, vmax=100)
    ax1.set_title('Traditional Fine-tuning', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Task Evaluated', fontsize=11)
    ax1.set_ylabel('Training Stage', fontsize=11)

    # HSOKV heatmap
    sns.heatmap(hsokv_matrix, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=task_names, yticklabels=[f'After Task {i+1}' for i in range(len(task_names))],
                cbar_kws={'label': 'Accuracy (%)'}, ax=ax2, vmin=0, vmax=100)
    ax2.set_title('HSOKV (Ours)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Task Evaluated', fontsize=11)
    ax2.set_ylabel('Training Stage', fontsize=11)

    plt.suptitle('Accuracy Matrix: Rows=Training Stage, Cols=Task Evaluated',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = Path(__file__).parent / save_path
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_metrics_comparison(results, save_path='metrics_comparison.png'):
    """
    Bar chart comparing key metrics
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Average\nAccuracy', 'Task 1 Final\nAccuracy', 'Forgetting\nRate']
    ft_values = [
        results['finetuning']['average_accuracy'] * 100,
        results['finetuning']['task1_final_accuracy'] * 100,
        results['finetuning']['forgetting_rate'] * 100,
    ]
    hsokv_values = [
        results['hsokv']['average_accuracy'] * 100,
        results['hsokv']['task1_final_accuracy'] * 100,
        results['hsokv']['forgetting_rate'] * 100,
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, ft_values, width, label='Traditional Fine-tuning',
                   color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, hsokv_values, width, label='HSOKV (Ours)',
                   color='#2ecc71', alpha=0.8)

    # Styling
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Key Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)

    # Add value labels
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    add_value_labels(bars1)
    add_value_labels(bars2)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Note about forgetting rate
    ax.text(0.5, -0.15, 'Note: Lower forgetting rate is better',
           transform=ax.transAxes, ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    output_path = Path(__file__).parent / save_path
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def generate_all_plots(results_file='benchmark_results.json'):
    """Generate all visualization plots"""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")

    try:
        results = load_results(results_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Creating plots...\n")

    plot_catastrophic_forgetting(results)
    plot_all_tasks_comparison(results)
    plot_accuracy_heatmap(results)
    plot_metrics_comparison(results)

    print("\n" + "="*70)
    print("✓ All visualizations generated successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  - forgetting_comparison.png (main result)")
    print("  - all_tasks_comparison.png")
    print("  - accuracy_heatmap.png")
    print("  - metrics_comparison.png")
    print("\nThese graphs clearly demonstrate HSOKV's advantage over")
    print("traditional fine-tuning in preventing catastrophic forgetting.")
    print("="*70 + "\n")


if __name__ == "__main__":
    generate_all_plots()
