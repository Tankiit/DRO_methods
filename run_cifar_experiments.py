#!/usr/bin/env python3
"""
Batch experiment runner for CIFAR ResNet experiments.
This script runs multiple experiments with different configurations.
"""

import subprocess
import itertools
import sys
import os
from pathlib import Path
import time
import json
from datetime import datetime

class CIFARBatchExperiments:
    """Batch runner for CIFAR ResNet experiments"""
    
    def __init__(self, base_output_dir='./cifar_batch_results'):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.results_log = []
        
    def run_experiment_grid(self, 
                           datasets=['cifar10', 'cifar100'],
                           models=['resnet18', 'resnet34', 'resnet50'],
                           epochs=[100],
                           learning_rates=[0.1],
                           batch_sizes=[128],
                           use_hierarchical_dro=[True, False],
                           use_pretrained=[False, True],
                           data_dir='./data'):
        """Run a grid of experiments with different configurations"""
        
        print("Starting CIFAR ResNet Batch Experiments")
        print(f"Base output directory: {self.base_output_dir}")
        print("="*60)
        
        # Create all combinations
        configs = list(itertools.product(
            datasets, models, epochs, learning_rates, 
            batch_sizes, use_hierarchical_dro, use_pretrained
        ))
        
        print(f"Total experiments to run: {len(configs)}")
        
        start_time = time.time()
        successful_runs = 0
        failed_runs = 0
        
        for i, (dataset, model, epoch, lr, batch_size, hierarchical_dro, pretrained) in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] Running experiment:")
            print(f"  Dataset: {dataset}")
            print(f"  Model: {model}")
            print(f"  Epochs: {epoch}")
            print(f"  Learning Rate: {lr}")
            print(f"  Batch Size: {batch_size}")
            print(f"  Hierarchical DRO: {hierarchical_dro}")
            print(f"  Pretrained: {pretrained}")
            
            # Create unique output directory for this experiment
            exp_name = f"{dataset}_{model}_ep{epoch}_lr{lr}_bs{batch_size}_dro{hierarchical_dro}_pre{pretrained}"
            exp_output_dir = self.base_output_dir / exp_name
            
            # Build command
            cmd = [
                sys.executable, 'cifar_resnet_experiments.py',
                '--dataset', dataset,
                '--model', model,
                '--epochs', str(epoch),
                '--lr', str(lr),
                '--batch-size', str(batch_size),
                '--data-dir', data_dir,
                '--output-dir', str(exp_output_dir),
                '--device', 'cuda'
            ]
            
            if not hierarchical_dro:
                cmd.append('--no-hierarchical-dro')
            
            if pretrained:
                cmd.append('--pretrained')
            
            # Run experiment
            exp_start_time = time.time()
            try:
                print(f"  Command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600*4)  # 4 hour timeout
                
                if result.returncode == 0:
                    exp_duration = time.time() - exp_start_time
                    print(f"  ✅ SUCCESS ({exp_duration:.1f}s)")
                    successful_runs += 1
                    
                    # Log success
                    self.results_log.append({
                        'experiment': exp_name,
                        'status': 'success',
                        'duration': exp_duration,
                        'config': {
                            'dataset': dataset,
                            'model': model,
                            'epochs': epoch,
                            'lr': lr,
                            'batch_size': batch_size,
                            'hierarchical_dro': hierarchical_dro,
                            'pretrained': pretrained
                        },
                        'output_dir': str(exp_output_dir),
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    print(f"  ❌ FAILED (return code: {result.returncode})")
                    print(f"  Error: {result.stderr[-500:]}")  # Last 500 chars of error
                    failed_runs += 1
                    
                    # Log failure
                    self.results_log.append({
                        'experiment': exp_name,
                        'status': 'failed',
                        'duration': time.time() - exp_start_time,
                        'config': {
                            'dataset': dataset,
                            'model': model,
                            'epochs': epoch,
                            'lr': lr,
                            'batch_size': batch_size,
                            'hierarchical_dro': hierarchical_dro,
                            'pretrained': pretrained
                        },
                        'error': result.stderr[-500:],
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except subprocess.TimeoutExpired:
                print(f"  ⏰ TIMEOUT (exceeded 4 hours)")
                failed_runs += 1
                
                self.results_log.append({
                    'experiment': exp_name,
                    'status': 'timeout',
                    'duration': time.time() - exp_start_time,
                    'config': {
                        'dataset': dataset,
                        'model': model,
                        'epochs': epoch,
                        'lr': lr,
                        'batch_size': batch_size,
                        'hierarchical_dro': hierarchical_dro,
                        'pretrained': pretrained
                    },
                    'error': 'Experiment exceeded 4 hour timeout',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"  💥 EXCEPTION: {str(e)}")
                failed_runs += 1
                
                self.results_log.append({
                    'experiment': exp_name,
                    'status': 'exception',
                    'duration': time.time() - exp_start_time,
                    'config': {
                        'dataset': dataset,
                        'model': model,
                        'epochs': epoch,
                        'lr': lr,
                        'batch_size': batch_size,
                        'hierarchical_dro': hierarchical_dro,
                        'pretrained': pretrained
                    },
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("BATCH EXPERIMENTS COMPLETED")
        print(f"{'='*60}")
        print(f"Total experiments: {len(configs)}")
        print(f"Successful: {successful_runs}")
        print(f"Failed: {failed_runs}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Average time per experiment: {total_time/len(configs):.1f} seconds")
        
        # Save results log
        self.save_batch_results()
        
        return self.results_log
    
    def run_quick_comparison(self, data_dir='./data'):
        """Run a quick comparison with key configurations"""
        print("Running Quick CIFAR Comparison...")
        
        return self.run_experiment_grid(
            datasets=['cifar10', 'cifar100'],
            models=['resnet18', 'resnet50'],
            epochs=[50],  # Shorter for quick comparison
            learning_rates=[0.1],
            batch_sizes=[128],
            use_hierarchical_dro=[True, False],
            use_pretrained=[False],
            data_dir=data_dir
        )
    
    def run_full_comparison(self, data_dir='./data'):
        """Run comprehensive comparison"""
        print("Running Full CIFAR Comparison...")
        
        return self.run_experiment_grid(
            datasets=['cifar10', 'cifar100'],
            models=['resnet18', 'resnet34', 'resnet50'],
            epochs=[100],
            learning_rates=[0.1, 0.01],
            batch_sizes=[128],
            use_hierarchical_dro=[True, False],
            use_pretrained=[False, True],
            data_dir=data_dir
        )
    
    def run_ablation_study(self, data_dir='./data'):
        """Run ablation study focusing on hierarchical DRO"""
        print("Running Hierarchical DRO Ablation Study...")
        
        return self.run_experiment_grid(
            datasets=['cifar10'],
            models=['resnet18', 'resnet50'],
            epochs=[100],
            learning_rates=[0.1],
            batch_sizes=[128],
            use_hierarchical_dro=[True, False],
            use_pretrained=[False, True],
            data_dir=data_dir
        )
    
    def save_batch_results(self):
        """Save batch experiment results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as JSON
        json_path = self.base_output_dir / f'batch_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(self.results_log, f, indent=4)
        
        # Create summary CSV
        import pandas as pd
        
        # Extract key info for CSV
        summary_data = []
        for result in self.results_log:
            row = {
                'experiment': result['experiment'],
                'status': result['status'],
                'duration_hours': result['duration'] / 3600,
                **result['config']
            }
            if result['status'] == 'success':
                row['output_dir'] = result['output_dir']
            else:
                row['error'] = result.get('error', 'Unknown error')[:100]  # Truncate error
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        csv_path = self.base_output_dir / f'batch_summary_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"Batch results saved:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
    
    def analyze_results(self):
        """Analyze and visualize batch results"""
        if not self.results_log:
            print("No results to analyze")
            return
        
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Create DataFrame from results
        summary_data = []
        for result in self.results_log:
            if result['status'] == 'success':
                row = {
                    'experiment': result['experiment'],
                    'duration_hours': result['duration'] / 3600,
                    **result['config']
                }
                summary_data.append(row)
        
        if not summary_data:
            print("No successful experiments to analyze")
            return
        
        df = pd.DataFrame(summary_data)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Duration by model
        df.groupby('model')['duration_hours'].mean().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Average Training Time by Model')
        axes[0,0].set_ylabel('Hours')
        
        # Duration by dataset
        df.groupby('dataset')['duration_hours'].mean().plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Average Training Time by Dataset')
        axes[0,1].set_ylabel('Hours')
        
        # Duration by hierarchical DRO
        df.groupby('hierarchical_dro')['duration_hours'].mean().plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Training Time: Standard vs Hierarchical DRO')
        axes[1,0].set_ylabel('Hours')
        axes[1,0].set_xticklabels(['Standard', 'Hierarchical DRO'])
        
        # Success rate
        status_counts = pd.Series([r['status'] for r in self.results_log]).value_counts()
        status_counts.plot(kind='pie', ax=axes[1,1], autopct='%1.1f%%')
        axes[1,1].set_title('Experiment Success Rate')
        
        plt.tight_layout()
        
        plots_path = self.base_output_dir / 'batch_analysis.png'
        plt.savefig(plots_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Analysis plot saved to {plots_path}")
        
        # Print summary statistics
        print("\nBatch Experiment Analysis:")
        print(f"Successful experiments: {len(df)}")
        print(f"Average training time: {df['duration_hours'].mean():.2f} hours")
        print(f"Fastest experiment: {df['duration_hours'].min():.2f} hours")
        print(f"Slowest experiment: {df['duration_hours'].max():.2f} hours")
        
        # Model comparison
        print("\nTraining time by model:")
        model_times = df.groupby('model')['duration_hours'].agg(['mean', 'std'])
        print(model_times.round(2))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch CIFAR ResNet Experiments')
    parser.add_argument('--mode', choices=['quick', 'full', 'ablation', 'custom'], 
                       default='quick', help='Experiment mode to run')
    parser.add_argument('--data-dir', default='./data', help='Data directory')
    parser.add_argument('--output-dir', default='./cifar_batch_results', help='Output directory')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze existing results')
    
    # Custom mode options
    parser.add_argument('--datasets', nargs='+', choices=['cifar10', 'cifar100'],
                       default=['cifar10', 'cifar100'], help='Datasets to use')
    parser.add_argument('--models', nargs='+', 
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                       default=['resnet18', 'resnet50'], help='Models to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    
    args = parser.parse_args()
    
    # Create batch runner
    batch_runner = CIFARBatchExperiments(base_output_dir=args.output_dir)
    
    if args.analyze_only:
        batch_runner.analyze_results()
        return
    
    # Run experiments based on mode
    if args.mode == 'quick':
        results = batch_runner.run_quick_comparison(data_dir=args.data_dir)
    elif args.mode == 'full':
        results = batch_runner.run_full_comparison(data_dir=args.data_dir)
    elif args.mode == 'ablation':
        results = batch_runner.run_ablation_study(data_dir=args.data_dir)
    elif args.mode == 'custom':
        results = batch_runner.run_experiment_grid(
            datasets=args.datasets,
            models=args.models,
            epochs=[args.epochs],
            data_dir=args.data_dir
        )
    
    # Analyze results
    batch_runner.analyze_results()


if __name__ == "__main__":
    main() 