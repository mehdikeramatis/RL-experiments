#!/usr/bin/env python3
"""Re-plot results from saved CSV files with improved visualization."""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_csv(filepath):
    """Load CSV file and return data as list of lists (excluding header)."""
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            rows.append(row)
    return rows

def plot_mean_std_improved(x, datas, labels, filepath, title, xlabel='Episode', ylabel='Return'):
    """Plot mean and std with proper axis scaling and improved visibility."""
    plt.figure(figsize=(12, 7))
    
    # Collect all values to determine global min/max for consistent scaling
    all_values = []
    plot_data = []
    
    for label, series in zip(labels, datas):
        # Convert series to list of floats
        if isinstance(series, np.ndarray):
            values = series.flatten().astype(np.float32)
        else:
            values = np.array([float(s) for s in series], dtype=np.float32)
        
        all_values.extend(values.tolist())
        plot_data.append((label, values))
    
    # Calculate global min/max with some padding
    all_values = np.array(all_values)
    global_min = np.nanmin(all_values)
    global_max = np.nanmax(all_values)
    value_range = global_max - global_min
    padding = value_range * 0.1  # 10% padding
    y_min = global_min - padding
    y_max = global_max + padding
    
    # Plot each series
    for label, values in plot_data:
        if len(values) > 1:
            # Plot single line with all points
            plt.plot(x[:len(values)], values, label=label, linewidth=2, alpha=0.8)
        else:
            plt.plot(x[:len(values)], values, 'o', label=label, markersize=8)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

def plot_mean_std_with_seeds(x, datas, labels, filepath, title, xlabel='Episode', ylabel='Return'):
    """Plot individual seed curves plus mean with std."""
    plt.figure(figsize=(12, 7))
    
    all_values = []
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    
    for idx, (label, series_list) in enumerate(zip(labels, datas)):
        # series_list is a list of arrays (one per seed)
        arrays = []
        for seed_idx, s in enumerate(series_list):
            if isinstance(s, (list, tuple)):
                arr = np.array([float(val) for val in s], dtype=np.float32)
            else:
                arr = np.asarray(s, dtype=np.float32).flatten()
            arrays.append(arr)
            all_values.extend(arr.tolist())
            
            # Plot each seed as a thin line
            plt.plot(x[:len(arr)], arr, color=colors[idx], alpha=0.3, linewidth=1)
        
        # Compute and plot mean
        if arrays:
            stacked = np.stack(arrays, axis=0)
            mean = np.mean(stacked, axis=0)
            std = np.std(stacked, axis=0)
            
            # Plot mean line with error band
            plt.plot(x[:len(mean)], mean, label=label, color=colors[idx], linewidth=2.5, 
                    marker='o', markersize=5, markevery=max(1, len(mean)//15))
            plt.fill_between(x[:len(mean)], mean - std, mean + std, alpha=0.15, color=colors[idx])
    
    # Set y-axis limits
    if all_values:
        all_values = np.array(all_values)
        global_min = np.nanmin(all_values)
        global_max = np.nanmax(all_values)
        value_range = global_max - global_min
        padding = value_range * 0.1
        y_min = global_min - padding
        y_max = global_max + padding
        plt.ylim(y_min, y_max)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

def plot_bar(values, errors, labels, filepath, title, ylabel='Return'):
    """Plot bar chart with error bars."""
    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    bars = plt.bar(x, values, yerr=errors, capsize=8, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    plt.xticks(x, labels, rotation=15, ha='right', fontsize=11)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

def main():
    raw_dir = 'results/raw'
    plot_dir = 'results/plots'
    
    methods = ['discrete_value', 'reinforce', 'actor_critic', 'ppo', 'sac']
    seeds = [0, 1, 2]
    episodes = 120
    
    aggregated_returns = defaultdict(list)
    final_returns = []
    final_labels = []
    
    print("\n" + "="*80)
    print("Re-plotting results from saved CSV files...")
    print("="*80 + "\n")
    
    # Reload and plot individual method results
    for method in methods:
        print(f"\nProcessing {method.upper()}...")
        all_returns = []
        final_seed_returns = []
        
        for seed in seeds:
            csv_path = os.path.join(raw_dir, f'{method}_seed{seed}.csv')
            rows = load_csv(csv_path)
            returns = [float(row[1]) for row in rows]  # Column 1 is return
            all_returns.append(returns)
            final_seed_returns.append(np.mean(returns[-10:]))
        
        aggregated_returns[method] = all_returns
        mean_final = float(np.mean(final_seed_returns))
        final_returns.append(mean_final)
        final_labels.append(method)
        
        # Plot individual method curves with all seeds
        plot_mean_std_with_seeds(
            x=list(range(episodes)),
            datas=[all_returns],  # Wrap in list to match label structure
            labels=[method],  # Single label for this method
            filepath=os.path.join(plot_dir, f'{method}_return_curves.png'),
            title=f'{method.upper()} - Return Learning Curves (3 seeds)',
            xlabel='Episode',
            ylabel='Episode Return'
        )
    
    # Comparison plot
    print(f"\nCreating comparison plot...")
    plot_mean_std_with_seeds(
        x=list(range(episodes)),
        datas=[aggregated_returns[method] for method in final_labels],
        labels=final_labels,
        filepath=os.path.join(plot_dir, 'comparison_return_curves.png'),
        title='Comparison of RL Methods - Return Learning Curves',
        xlabel='Episode',
        ylabel='Episode Return'
    )
    
    # Final performance plot
    print(f"Creating final performance plot...")
    plot_bar(
        final_returns, 
        [0.0] * len(final_returns), 
        final_labels, 
        os.path.join(plot_dir, 'final_performance.png'), 
        'Final Performance - Mean Return (Last 10 Episodes)'
    )
    
    print("\n" + "="*80)
    print("All plots have been re-generated successfully!")
    print(f"Plots saved to: {plot_dir}")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
