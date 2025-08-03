#!/usr/bin/env python3
"""
Training Log Analyzer for Hierarchical DRO
This script analyzes training logs and TensorBoard data to generate comprehensive training plots.
"""

import argparse
import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
import torch
from collections import defaultdict, OrderedDict

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze training logs and generate plots')
    
    # Input configuration
    parser.add_argument('--log-dir', type=str, default='runs', 
                       help='Directory containing TensorBoard logs')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Specific log file to analyze')
    parser.add_argument('--training-log', type=str, default=None,
                       help='Text training log file to parse')
    
    # Output configuration
    parser.add_argument('--output-dir', default='training_analysis_results', 
                       help='Output directory for analysis results')
    parser.add_argument('--run-name', default='', help='Custom run name')
    
    # Analysis configuration
    parser.add_argument('--metrics', nargs='+', 
                       default=['loss', 'accuracy', 'lr', 'grad_norm'],
                       help='Specific metrics to analyze')
    parser.add_argument('--plot-types', nargs='+',
                       default=['loss_curves', 'accuracy_curves', 'lr_schedule', 'grad_norms'],
                       help='Types of plots to generate')
    
    return parser.parse_args()

def extract_tensorboard_data(log_dir, run_name=None):
    """Extract data from TensorBoard event files."""
    print(f"🔍 Extracting TensorBoard data from {log_dir}...")
    
    # Find all event files
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        print("❌ No TensorBoard event files found!")
        return {}
    
    print(f"📊 Found {len(event_files)} event files")
    
    all_data = {}
    
    for event_file in event_files:
        try:
            # Extract run name from path
            run_path = os.path.dirname(event_file)
            run_name = os.path.basename(run_path) if not run_name else run_name
            
            print(f"  📈 Processing: {run_name}")
            
            # Load event data
            ea = EventAccumulator(event_file)
            ea.Reload()
            
            # Extract scalar data
            run_data = {}
            for tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                run_data[tag] = {
                    'steps': [event.step for event in events],
                    'values': [event.value for event in events],
                    'wall_times': [event.wall_time for event in events]
                }
            
            all_data[run_name] = run_data
            
        except Exception as e:
            print(f"  ⚠️ Failed to process {event_file}: {str(e)}")
            continue
    
    return all_data

def parse_training_log(log_file):
    """Parse text-based training log files."""
    print(f"📝 Parsing training log: {log_file}")
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return {}
    
    # Common patterns in training logs
    patterns = {
        'epoch': r'Epoch (\d+)',
        'loss': r'loss[:\s]*([\d\.]+)',
        'accuracy': r'acc[:\s]*([\d\.]+)',
        'lr': r'lr[:\s]*([\d\.e+-]+)',
        'time': r'(\d{2}:\d{2}:\d{2})',
        'step': r'step[:\s]*(\d+)',
        'batch': r'batch[:\s]*(\d+)',
        'val_loss': r'val_loss[:\s]*([\d\.]+)',
        'val_acc': r'val_acc[:\s]*([\d\.]+)'
    }
    
    log_data = defaultdict(list)
    
    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f):
            for metric, pattern in patterns.items():
                matches = re.findall(pattern, line, re.IGNORECASE)
                if matches:
                    for match in matches:
                        try:
                            if metric in ['loss', 'accuracy', 'lr', 'val_loss', 'val_acc']:
                                value = float(match)
                            elif metric in ['epoch', 'step', 'batch']:
                                value = int(match)
                            else:
                                value = match
                            
                            log_data[metric].append({
                                'line': line_num,
                                'value': value,
                                'timestamp': line.strip()[:50]  # First 50 chars for context
                            })
                        except ValueError:
                            continue
    
    return dict(log_data)

def create_training_plots(tensorboard_data, log_data, output_dir, plot_types):
    """Create comprehensive training plots."""
    print(f"🎨 Generating training plots...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plots_created = []
    
    # 1. Loss Curves
    if 'loss_curves' in plot_types:
        plots_created.extend(plot_loss_curves(tensorboard_data, log_data, output_path))
    
    # 2. Accuracy Curves
    if 'accuracy_curves' in plot_types:
        plots_created.extend(plot_accuracy_curves(tensorboard_data, log_data, output_path))
    
    # 3. Learning Rate Schedule
    if 'lr_schedule' in plot_types:
        plots_created.extend(plot_lr_schedule(tensorboard_data, log_data, output_path))
    
    # 4. Gradient Norms
    if 'grad_norms' in plot_types:
        plots_created.extend(plot_gradient_norms(tensorboard_data, log_data, output_path))
    
    # 5. Training Summary Dashboard
    if 'dashboard' in plot_types:
        plots_created.extend(create_training_dashboard(tensorboard_data, log_data, output_path))
    
    # 6. Loss Components Analysis (for Hierarchical DRO)
    if 'loss_components' in plot_types:
        plots_created.extend(plot_loss_components(tensorboard_data, log_data, output_path))
    
    return plots_created

def plot_loss_curves(tensorboard_data, log_data, output_path):
    """Plot training and validation loss curves."""
    plots = []
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Loss Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: TensorBoard Loss Data
    ax1 = axes[0, 0]
    for run_name, run_data in tensorboard_data.items():
        if 'loss' in run_data:
            for loss_type, loss_data in run_data.items():
                if 'loss' in loss_type.lower():
                    ax1.plot(loss_data['steps'], loss_data['values'], 
                            label=f'{run_name}_{loss_type}', alpha=0.8)
    
    ax1.set_title('TensorBoard Loss Curves')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log File Loss Data
    ax2 = axes[0, 1]
    if 'loss' in log_data:
        loss_values = [entry['value'] for entry in log_data['loss']]
        line_numbers = [entry['line'] for entry in log_data['loss']]
        ax2.plot(line_numbers, loss_values, 'b-', label='Training Loss', linewidth=2)
    
    if 'val_loss' in log_data:
        val_loss_values = [entry['value'] for entry in log_data['val_loss']]
        val_line_numbers = [entry['line'] for entry in log_data['val_loss']]
        ax2.plot(val_line_numbers, val_loss_values, 'r--', label='Validation Loss', linewidth=2)
    
    ax2.set_title('Log File Loss Curves')
    ax2.set_xlabel('Log Line Number')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss Distribution
    ax3 = axes[1, 0]
    all_losses = []
    for run_data in tensorboard_data.values():
        for loss_data in run_data.values():
            if 'loss' in loss_data and 'values' in loss_data:
                all_losses.extend(loss_data['values'])
    
    if all_losses:
        ax3.hist(all_losses, bins=30, alpha=0.7, edgecolor='black')
        ax3.set_title('Loss Distribution')
        ax3.set_xlabel('Loss Value')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Loss Convergence
    ax4 = axes[1, 1]
    for run_name, run_data in tensorboard_data.items():
        if 'loss' in run_data:
            for loss_type, loss_data in run_data.items():
                if 'loss' in loss_type.lower() and len(loss_data['values']) > 10:
                    # Plot last 20% of training for convergence analysis
                    start_idx = int(len(loss_data['values']) * 0.8)
                    steps = loss_data['steps'][start_idx:]
                    values = loss_data['values'][start_idx:]
                    ax4.plot(steps, values, label=f'{run_name}_{loss_type}', alpha=0.8)
    
    ax4.set_title('Loss Convergence (Last 20% of Training)')
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / 'loss_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plots.append(str(plot_path))
    
    return plots

def plot_accuracy_curves(tensorboard_data, log_data, output_path):
    """Plot training and validation accuracy curves."""
    plots = []
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Accuracy Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: TensorBoard Accuracy Data
    ax1 = axes[0]
    for run_name, run_data in tensorboard_data.items():
        if 'accuracy' in run_data or 'acc' in run_data:
            for acc_type, acc_data in run_data.items():
                if 'acc' in acc_type.lower():
                    ax1.plot(acc_data['steps'], acc_data['values'], 
                            label=f'{run_name}_{acc_type}', alpha=0.8)
    
    ax1.set_title('TensorBoard Accuracy Curves')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log File Accuracy Data
    ax2 = axes[1]
    if 'accuracy' in log_data:
        acc_values = [entry['value'] for entry in log_data['accuracy']]
        line_numbers = [entry['line'] for entry in log_data['accuracy']]
        ax2.plot(line_numbers, acc_values, 'g-', label='Training Accuracy', linewidth=2)
    
    if 'val_acc' in log_data:
        val_acc_values = [entry['value'] for entry in log_data['val_acc']]
        val_line_numbers = [entry['line'] for entry in log_data['val_acc']]
        ax2.plot(val_line_numbers, val_acc_values, 'm--', label='Validation Accuracy', linewidth=2)
    
    ax2.set_title('Log File Accuracy Curves')
    ax2.set_xlabel('Log Line Number')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / 'accuracy_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plots.append(str(plot_path))
    
    return plots

def plot_lr_schedule(tensorboard_data, log_data, output_path):
    """Plot learning rate schedule."""
    plots = []
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Learning Rate Schedule Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: TensorBoard LR Data
    ax1 = axes[0]
    for run_name, run_data in tensorboard_data.items():
        if 'lr' in run_data or 'learning_rate' in run_data:
            for lr_type, lr_data in run_data.items():
                if 'lr' in lr_type.lower():
                    ax1.plot(lr_data['steps'], lr_data['values'], 
                            label=f'{run_name}_{lr_type}', alpha=0.8)
    
    ax1.set_title('TensorBoard Learning Rate Schedule')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Learning Rate')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log File LR Data
    ax2 = axes[1]
    if 'lr' in log_data:
        lr_values = [entry['value'] for entry in log_data['lr']]
        line_numbers = [entry['line'] for entry in log_data['lr']]
        ax2.plot(line_numbers, lr_values, 'orange', label='Learning Rate', linewidth=2)
    
    ax2.set_title('Log File Learning Rate Schedule')
    ax2.set_xlabel('Log Line Number')
    ax2.set_ylabel('Learning Rate')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / 'lr_schedule.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plots.append(str(plot_path))
    
    return plots

def plot_gradient_norms(tensorboard_data, log_data, output_path):
    """Plot gradient norms."""
    plots = []
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Gradient Norm Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: TensorBoard Gradient Data
    ax1 = axes[0]
    for run_name, run_data in tensorboard_data.items():
        if 'grad' in run_data or 'gradient' in run_data:
            for grad_type, grad_data in run_data.items():
                if 'grad' in grad_type.lower():
                    ax1.plot(grad_data['steps'], grad_data['values'], 
                            label=f'{run_name}_{grad_type}', alpha=0.8)
    
    ax1.set_title('TensorBoard Gradient Norms')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Gradient Norm')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gradient Norm Distribution
    ax2 = axes[1]
    all_grads = []
    for run_data in tensorboard_data.values():
        for grad_data in run_data.values():
            if 'grad' in grad_data and 'values' in grad_data:
                all_grads.extend(grad_data['values'])
    
    if all_grads:
        ax2.hist(all_grads, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_title('Gradient Norm Distribution')
        ax2.set_xlabel('Gradient Norm')
        ax2.set_ylabel('Frequency')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / 'gradient_norms.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plots.append(str(plot_path))
    
    return plots

def plot_loss_components(tensorboard_data, log_data, output_path):
    """Plot Hierarchical DRO loss components."""
    plots = []
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hierarchical DRO Loss Components Analysis', fontsize=16, fontweight='bold')
    
    # Find loss component data
    loss_components = {}
    for run_name, run_data in tensorboard_data.items():
        for metric_name, metric_data in run_data.items():
            if any(comp in metric_name.lower() for comp in ['pixel', 'feature', 'cross', 'virtual', 'energy']):
                loss_components[metric_name] = metric_data
    
    if loss_components:
        # Plot 1: Loss Components Over Time
        ax1 = axes[0, 0]
        for comp_name, comp_data in loss_components.items():
            ax1.plot(comp_data['steps'], comp_data['values'], 
                    label=comp_name, alpha=0.8)
        
        ax1.set_title('Loss Components Over Time')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss Component Ratios
        ax2 = axes[0, 1]
        if len(loss_components) >= 2:
            # Calculate ratios between components
            comp_names = list(loss_components.keys())
            comp1_data = loss_components[comp_names[0]]
            comp2_data = loss_components[comp_names[1]]
            
            # Align steps
            min_steps = min(len(comp1_data['values']), len(comp2_data['values']))
            ratios = [comp1_data['values'][i] / (comp2_data['values'][i] + 1e-8) 
                     for i in range(min_steps)]
            steps = comp1_data['steps'][:min_steps]
            
            ax2.plot(steps, ratios, label=f'{comp_names[0]}/{comp_names[1]}', alpha=0.8)
            ax2.set_title('Loss Component Ratios')
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Ratio')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Loss Component Distribution
        ax3 = axes[1, 0]
        for comp_name, comp_data in loss_components.items():
            ax3.hist(comp_data['values'], bins=20, alpha=0.6, label=comp_name)
        
        ax3.set_title('Loss Component Distributions')
        ax3.set_xlabel('Loss Value')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Loss Component Correlation
        ax4 = axes[1, 1]
        if len(loss_components) >= 2:
            comp_names = list(loss_components.keys())
            comp1_values = loss_components[comp_names[0]]['values']
            comp2_values = loss_components[comp_names[1]]['values']
            
            min_len = min(len(comp1_values), len(comp2_values))
            ax4.scatter(comp1_values[:min_len], comp2_values[:min_len], alpha=0.6)
            ax4.set_xlabel(comp_names[0])
            ax4.set_ylabel(comp_names[1])
            ax4.set_title('Loss Component Correlation')
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / 'loss_components.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plots.append(str(plot_path))
    
    return plots

def create_training_dashboard(tensorboard_data, log_data, output_path):
    """Create a comprehensive training dashboard."""
    plots = []
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Dashboard - Hierarchical DRO Analysis', fontsize=16, fontweight='bold')
    
    # Dashboard 1: Overall Loss Trend
    ax1 = axes[0, 0]
    for run_name, run_data in tensorboard_data.items():
        if 'loss' in run_data:
            for loss_type, loss_data in run_data.items():
                if 'loss' in loss_type.lower():
                    ax1.plot(loss_data['steps'], loss_data['values'], 
                            label=f'{run_name}_{loss_type}', alpha=0.8)
    
    ax1.set_title('Overall Loss Trend')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Dashboard 2: Learning Rate Schedule
    ax2 = axes[0, 1]
    for run_name, run_data in tensorboard_data.items():
        if 'lr' in run_data:
            for lr_type, lr_data in run_data.items():
                if 'lr' in lr_type.lower():
                    ax2.plot(lr_data['steps'], lr_data['values'], 
                            label=f'{run_name}_{lr_type}', alpha=0.8)
    
    ax2.set_title('Learning Rate Schedule')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Learning Rate')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Dashboard 3: Accuracy Progress
    ax3 = axes[0, 2]
    for run_name, run_data in tensorboard_data.items():
        if 'accuracy' in run_data:
            for acc_type, acc_data in run_data.items():
                if 'acc' in acc_type.lower():
                    ax3.plot(acc_data['steps'], acc_data['values'], 
                            label=f'{run_name}_{acc_type}', alpha=0.8)
    
    ax3.set_title('Accuracy Progress')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Dashboard 4: Training Statistics
    ax4 = axes[1, 0]
    stats_data = []
    for run_name, run_data in tensorboard_data.items():
        for metric_name, metric_data in run_data.items():
            if len(metric_data['values']) > 0:
                stats_data.append({
                    'metric': metric_name,
                    'run': run_name,
                    'final_value': metric_data['values'][-1],
                    'min_value': min(metric_data['values']),
                    'max_value': max(metric_data['values'])
                })
    
    if stats_data:
        df_stats = pd.DataFrame(stats_data)
        df_stats.plot(kind='bar', ax=ax4, x='metric', y='final_value')
        ax4.set_title('Final Metric Values')
        ax4.set_xlabel('Metric')
        ax4.set_ylabel('Value')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
    
    # Dashboard 5: Training Convergence
    ax5 = axes[1, 1]
    for run_name, run_data in tensorboard_data.items():
        if 'loss' in run_data:
            for loss_type, loss_data in run_data.items():
                if 'loss' in loss_type.lower() and len(loss_data['values']) > 10:
                    # Plot moving average for convergence
                    window = min(50, len(loss_data['values']) // 10)
                    if window > 1:
                        moving_avg = pd.Series(loss_data['values']).rolling(window=window).mean()
                        ax5.plot(loss_data['steps'], moving_avg, 
                                label=f'{run_name}_{loss_type}_avg', alpha=0.8)
    
    ax5.set_title('Loss Convergence (Moving Average)')
    ax5.set_xlabel('Training Steps')
    ax5.set_ylabel('Loss (Moving Avg)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Dashboard 6: Training Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create summary text
    summary_text = "Training Summary:\n\n"
    for run_name, run_data in tensorboard_data.items():
        summary_text += f"Run: {run_name}\n"
        for metric_name, metric_data in run_data.items():
            if len(metric_data['values']) > 0:
                final_val = metric_data['values'][-1]
                min_val = min(metric_data['values'])
                max_val = max(metric_data['values'])
                summary_text += f"  {metric_name}: {final_val:.4f} (min: {min_val:.4f}, max: {max_val:.4f})\n"
        summary_text += "\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plot_path = output_path / 'training_dashboard.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plots.append(str(plot_path))
    
    return plots

def generate_training_report(tensorboard_data, log_data, output_path):
    """Generate a comprehensive training report."""
    print(f"📊 Generating training report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {},
        'runs': {},
        'recommendations': []
    }
    
    # Analyze each run
    for run_name, run_data in tensorboard_data.items():
        run_summary = {
            'metrics': {},
            'convergence': {},
            'performance': {}
        }
        
        for metric_name, metric_data in run_data.items():
            if len(metric_data['values']) > 0:
                values = metric_data['values']
                run_summary['metrics'][metric_name] = {
                    'final_value': values[-1],
                    'min_value': min(values),
                    'max_value': max(values),
                    'mean_value': np.mean(values),
                    'std_value': np.std(values),
                    'convergence_rate': (values[0] - values[-1]) / values[0] if values[0] != 0 else 0
                }
        
        report['runs'][run_name] = run_summary
    
    # Generate recommendations
    for run_name, run_summary in report['runs'].items():
        if 'loss' in run_summary['metrics']:
            loss_conv = run_summary['metrics']['loss']['convergence_rate']
            if loss_conv < 0.1:
                report['recommendations'].append(f"{run_name}: Loss convergence is slow, consider adjusting learning rate")
            elif loss_conv > 0.9:
                report['recommendations'].append(f"{run_name}: Excellent loss convergence")
    
    # Save report
    report_path = output_path / 'training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4, default=str)
    
    print(f"✅ Training report saved to: {report_path}")
    return str(report_path)

def main():
    args = parse_args()
    
    print("🔬 Training Log Analyzer for Hierarchical DRO")
    print("=" * 60)
    
    # Extract data
    tensorboard_data = {}
    log_data = {}
    
    if args.log_dir:
        tensorboard_data = extract_tensorboard_data(args.log_dir, args.run_name)
    
    if args.training_log:
        log_data = parse_training_log(args.training_log)
    
    if not tensorboard_data and not log_data:
        print("❌ No data found to analyze!")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}/training_analysis_{timestamp}"
    
    print(f"📁 Output directory: {output_dir}")
    
    # Generate plots
    plots_created = create_training_plots(tensorboard_data, log_data, output_dir, args.plot_types)
    
    # Generate report
    report_path = generate_training_report(tensorboard_data, log_data, Path(output_dir))
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING ANALYSIS COMPLETED!")
    print("=" * 60)
    print(f"📊 Plots created: {len(plots_created)}")
    for plot in plots_created:
        print(f"   • {plot}")
    print(f"📋 Report: {report_path}")
    print(f"📁 All results saved to: {output_dir}")
    
    if tensorboard_data:
        print(f"\n📈 TensorBoard runs analyzed: {len(tensorboard_data)}")
        for run_name in tensorboard_data.keys():
            print(f"   • {run_name}")
    
    if log_data:
        print(f"\n📝 Log metrics extracted: {len(log_data)}")
        for metric in log_data.keys():
            print(f"   • {metric}")

if __name__ == "__main__":
    main() 