"""
Evaluation Infrastructure for Length Generalization Experiments

This module provides comprehensive evaluation metrics for:
1. Perplexity at different sequence lengths
2. Length penalty computation
3. Pattern accuracy (for synthetic data)
4. Attention analysis
5. Position encoding extrapolation quality

Metrics:
- PPL(L): Perplexity at length L
- Length Penalty: PPL(L_long) / PPL(L_short)
- Graceful Degradation: Slope of PPL vs log(L)
- Pattern Recall: Accuracy on long-range patterns
"""

import math
import json
import os
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class LengthEvaluationResult:
    """Container for evaluation results at a specific length."""
    length_range: Tuple[int, int]
    num_samples: int
    total_tokens: int
    avg_loss: float
    perplexity: float
    std_loss: float
    min_loss: float
    max_loss: float


@dataclass
class LengthGeneralizationMetrics:
    """Comprehensive metrics for length generalization."""
    # Per-length results
    results_by_length: Dict[str, LengthEvaluationResult]

    # Aggregate metrics
    length_penalty_2k_8k: float  # PPL(8K) / PPL(2K)
    length_penalty_2k_16k: float  # PPL(16K) / PPL(2K)
    length_penalty_2k_32k: float  # PPL(32K) / PPL(2K)

    # Degradation analysis
    degradation_slope: float  # Slope of PPL vs log(L)
    degradation_intercept: float

    # Extrapolation quality
    extrapolation_ratio: float  # Performance on unseen lengths vs seen lengths

    # Summary
    avg_perplexity: float
    worst_case_perplexity: float
    best_case_perplexity: float


class LengthGeneralizationEvaluator:
    """
    Evaluator for length generalization performance.

    Computes perplexity at multiple sequence lengths and analyzes
    how performance degrades with length.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        length_ranges: List[Tuple[int, int]] = None,
        device: str = "cuda",
        max_eval_samples: int = 1000,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_eval_samples = max_eval_samples

        # Default length ranges for evaluation
        if length_ranges is None:
            length_ranges = [
                (256, 512),
                (512, 1024),
                (1024, 2048),
                (2048, 4096),
                (4096, 8192),
                (8192, 16384),
                (16384, 32768),
            ]
        self.length_ranges = length_ranges

    @torch.no_grad()
    def evaluate_perplexity(
        self,
        dataloader,
        length_range: Optional[Tuple[int, int]] = None,
    ) -> LengthEvaluationResult:
        """
        Evaluate perplexity on a dataloader.

        Args:
            dataloader: DataLoader with evaluation data
            length_range: Optional length range to filter

        Returns:
            LengthEvaluationResult with metrics
        """
        self.model.eval()

        total_loss = 0
        total_tokens = 0
        losses = []
        num_samples = 0

        for batch in dataloader:
            if num_samples >= self.max_eval_samples:
                break

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Filter by length if specified
            if length_range is not None:
                lengths = batch.get('lengths', attention_mask.sum(dim=1) if attention_mask is not None
                                   else torch.full((input_ids.size(0),), input_ids.size(1)))
                mask = (lengths >= length_range[0]) & (lengths < length_range[1])
                if not mask.any():
                    continue
                input_ids = input_ids[mask]
                if attention_mask is not None:
                    attention_mask = attention_mask[mask]

            # Compute loss
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs['loss']

            # Per-sample loss
            batch_size = input_ids.size(0)
            num_tokens = attention_mask.sum().item() if attention_mask is not None else input_ids.numel()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            losses.append(loss.item())
            num_samples += batch_size

        if total_tokens == 0:
            return LengthEvaluationResult(
                length_range=length_range or (0, 0),
                num_samples=0,
                total_tokens=0,
                avg_loss=float('inf'),
                perplexity=float('inf'),
                std_loss=0,
                min_loss=float('inf'),
                max_loss=float('inf'),
            )

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow

        return LengthEvaluationResult(
            length_range=length_range or (0, max(self.length_ranges, key=lambda x: x[1])[1]),
            num_samples=num_samples,
            total_tokens=total_tokens,
            avg_loss=avg_loss,
            perplexity=perplexity,
            std_loss=float(np.std(losses)) if losses else 0,
            min_loss=min(losses) if losses else float('inf'),
            max_loss=max(losses) if losses else float('inf'),
        )

    def evaluate_all_lengths(
        self,
        dataloaders: Dict[str, 'DataLoader'],
    ) -> LengthGeneralizationMetrics:
        """
        Evaluate across all length ranges.

        Args:
            dataloaders: Dict mapping length range names to dataloaders

        Returns:
            LengthGeneralizationMetrics with comprehensive results
        """
        results_by_length = {}

        for name, dataloader in dataloaders.items():
            # Parse length range from name (e.g., "eval_512_1024")
            parts = name.split('_')
            if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                length_range = (int(parts[1]), int(parts[2]))
            else:
                length_range = None

            result = self.evaluate_perplexity(dataloader, length_range)
            results_by_length[name] = result

        # Compute aggregate metrics
        ppls = {name: r.perplexity for name, r in results_by_length.items() if r.perplexity < float('inf')}

        # Length penalty computation
        def get_ppl_for_length(target_len: int) -> float:
            for name, result in results_by_length.items():
                if result.length_range[0] <= target_len < result.length_range[1]:
                    return result.perplexity
            return float('inf')

        ppl_2k = get_ppl_for_length(2048)
        ppl_8k = get_ppl_for_length(8192)
        ppl_16k = get_ppl_for_length(16384)
        ppl_32k = get_ppl_for_length(32768)

        length_penalty_2k_8k = ppl_8k / ppl_2k if ppl_2k > 0 else float('inf')
        length_penalty_2k_16k = ppl_16k / ppl_2k if ppl_2k > 0 else float('inf')
        length_penalty_2k_32k = ppl_32k / ppl_2k if ppl_2k > 0 else float('inf')

        # Degradation slope (linear regression of log(PPL) vs log(L))
        slope, intercept = self._compute_degradation_slope(results_by_length)

        # Extrapolation ratio (PPL on lengths > training vs <= training)
        training_max_length = 2048  # Assume training on up to 2K
        seen_ppls = [r.perplexity for r in results_by_length.values()
                     if r.length_range[1] <= training_max_length and r.perplexity < float('inf')]
        unseen_ppls = [r.perplexity for r in results_by_length.values()
                       if r.length_range[0] >= training_max_length and r.perplexity < float('inf')]

        avg_seen = sum(seen_ppls) / len(seen_ppls) if seen_ppls else float('inf')
        avg_unseen = sum(unseen_ppls) / len(unseen_ppls) if unseen_ppls else float('inf')
        extrapolation_ratio = avg_unseen / avg_seen if avg_seen > 0 else float('inf')

        return LengthGeneralizationMetrics(
            results_by_length={name: asdict(result) for name, result in results_by_length.items()},
            length_penalty_2k_8k=length_penalty_2k_8k,
            length_penalty_2k_16k=length_penalty_2k_16k,
            length_penalty_2k_32k=length_penalty_2k_32k,
            degradation_slope=slope,
            degradation_intercept=intercept,
            extrapolation_ratio=extrapolation_ratio,
            avg_perplexity=sum(ppls.values()) / len(ppls) if ppls else float('inf'),
            worst_case_perplexity=max(ppls.values()) if ppls else float('inf'),
            best_case_perplexity=min(ppls.values()) if ppls else float('inf'),
        )

    def _compute_degradation_slope(
        self,
        results: Dict[str, LengthEvaluationResult],
    ) -> Tuple[float, float]:
        """Compute slope of log(PPL) vs log(L) using linear regression."""
        points = []
        for result in results.values():
            if result.perplexity < float('inf') and result.length_range[0] > 0:
                avg_length = (result.length_range[0] + result.length_range[1]) / 2
                points.append((math.log(avg_length), math.log(result.perplexity)))

        if len(points) < 2:
            return 0.0, 0.0

        # Linear regression
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])

        n = len(points)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        intercept = (np.sum(y) - slope * np.sum(x)) / n

        return float(slope), float(intercept)


class PatternAccuracyEvaluator:
    """
    Evaluator for pattern recognition accuracy in synthetic data.

    Measures how well the model learns patterns at different scales:
    - Local patterns (within ~50 tokens)
    - Chunk patterns (within ~500 tokens)
    - Global patterns (within ~4000 tokens)
    """

    def __init__(
        self,
        model: nn.Module,
        synthetic_dataset,  # SyntheticLongDocumentDataset
        device: str = "cuda",
    ):
        self.model = model
        self.dataset = synthetic_dataset
        self.device = device

    @torch.no_grad()
    def evaluate_pattern_accuracy(
        self,
        num_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate pattern recognition accuracy.

        Returns:
            Dict with accuracy per pattern type (local, chunk, global)
        """
        self.model.eval()

        accuracies = {'local': [], 'chunk': [], 'global': []}

        for idx in range(min(num_samples, len(self.dataset))):
            sample = self.dataset[idx]
            input_ids = sample['input_ids'].unsqueeze(0).to(self.device)

            # Get model predictions
            outputs = self.model(input_ids)
            predictions = outputs['logits'].argmax(dim=-1).squeeze(0)

            # Evaluate pattern accuracy
            pattern_acc = self.dataset.get_pattern_accuracy(predictions, idx)

            for pattern_type in ['local', 'chunk', 'global']:
                if pattern_acc[pattern_type] > 0:
                    accuracies[pattern_type].append(pattern_acc[pattern_type])

        return {
            pattern_type: np.mean(accs) if accs else 0.0
            for pattern_type, accs in accuracies.items()
        }


class AttentionAnalyzer:
    """
    Analyze attention patterns for length generalization insights.

    Examines:
    - Attention entropy (higher = more distributed)
    - Long-range attention ratio
    - Position bias patterns
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device

    @torch.no_grad()
    def analyze_attention(
        self,
        input_ids: torch.Tensor,
        layer_idx: int = -1,
    ) -> Dict[str, float]:
        """
        Analyze attention patterns.

        Args:
            input_ids: Input token IDs
            layer_idx: Which layer to analyze (-1 for last)

        Returns:
            Dict with attention metrics
        """
        self.model.eval()
        input_ids = input_ids.to(self.device)

        # Get attention weights (requires model modification to output attention)
        # For now, we'll compute proxy metrics

        outputs = self.model(input_ids, output_hidden_states=True)
        hidden_states = outputs.get('hidden_states', [outputs['last_hidden_state']])

        # Use hidden state similarity as proxy for attention patterns
        hidden = hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)

        # Compute pairwise similarities
        hidden_norm = F.normalize(hidden, dim=-1)
        similarities = torch.matmul(hidden_norm, hidden_norm.transpose(-2, -1))  # (batch, seq_len, seq_len)

        seq_len = similarities.size(1)

        # Compute metrics
        # 1. Long-range similarity (positions > seq_len/2 apart)
        long_range_mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=seq_len // 2)
        long_range_sim = (similarities * long_range_mask).sum() / (long_range_mask.sum() + 1e-8)

        # 2. Local similarity (positions < 10 apart)
        local_mask = (torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=0) -
                      torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=10))
        local_sim = (similarities * local_mask).sum() / (local_mask.sum() + 1e-8)

        # 3. Position entropy (how distributed is attention)
        # Higher = more distributed (better for long sequences)
        sim_probs = F.softmax(similarities, dim=-1)
        entropy = -(sim_probs * torch.log(sim_probs + 1e-8)).sum(dim=-1).mean()

        return {
            'long_range_similarity': long_range_sim.item(),
            'local_similarity': local_sim.item(),
            'attention_entropy': entropy.item(),
            'long_range_ratio': (long_range_sim / (local_sim + 1e-8)).item(),
        }


class EvaluationReporter:
    """
    Generate comprehensive evaluation reports.

    Outputs:
    - JSON results
    - Markdown summary
    - Plots (if matplotlib available)
    """

    def __init__(self, output_dir: str = "./results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_report(
        self,
        metrics: LengthGeneralizationMetrics,
        model_name: str,
        experiment_name: str,
    ) -> str:
        """Generate comprehensive evaluation report."""
        # Save JSON results
        json_path = os.path.join(self.output_dir, f"{experiment_name}_results.json")
        with open(json_path, 'w') as f:
            json.dump(asdict(metrics) if hasattr(metrics, '__dataclass_fields__') else metrics.__dict__,
                     f, indent=2, default=str)

        # Generate markdown summary
        md_report = self._generate_markdown(metrics, model_name, experiment_name)
        md_path = os.path.join(self.output_dir, f"{experiment_name}_report.md")
        with open(md_path, 'w') as f:
            f.write(md_report)

        # Generate plots if available
        if HAS_MATPLOTLIB:
            self._generate_plots(metrics, experiment_name)

        return md_report

    def _generate_markdown(
        self,
        metrics: LengthGeneralizationMetrics,
        model_name: str,
        experiment_name: str,
    ) -> str:
        """Generate markdown report."""
        lines = [
            f"# Length Generalization Evaluation Report",
            f"",
            f"**Model**: {model_name}",
            f"**Experiment**: {experiment_name}",
            f"",
            f"## Summary Metrics",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Average Perplexity | {metrics.avg_perplexity:.2f} |",
            f"| Best Perplexity | {metrics.best_case_perplexity:.2f} |",
            f"| Worst Perplexity | {metrics.worst_case_perplexity:.2f} |",
            f"| Length Penalty (2K→8K) | {metrics.length_penalty_2k_8k:.2f}x |",
            f"| Length Penalty (2K→16K) | {metrics.length_penalty_2k_16k:.2f}x |",
            f"| Length Penalty (2K→32K) | {metrics.length_penalty_2k_32k:.2f}x |",
            f"| Degradation Slope | {metrics.degradation_slope:.4f} |",
            f"| Extrapolation Ratio | {metrics.extrapolation_ratio:.2f} |",
            f"",
            f"## Per-Length Results",
            f"",
            f"| Length Range | Samples | PPL | Loss (std) |",
            f"|--------------|---------|-----|------------|",
        ]

        for name, result in metrics.results_by_length.items():
            if isinstance(result, dict):
                lines.append(f"| {result['length_range']} | {result['num_samples']} | "
                           f"{result['perplexity']:.2f} | {result['avg_loss']:.4f} ({result['std_loss']:.4f}) |")

        lines.extend([
            f"",
            f"## Analysis",
            f"",
            f"### Degradation Analysis",
            f"",
            f"The model shows a degradation slope of {metrics.degradation_slope:.4f}, meaning that "
            f"perplexity increases by approximately {math.exp(metrics.degradation_slope):.2f}x for each "
            f"doubling of sequence length.",
            f"",
            f"### Extrapolation Quality",
            f"",
            f"The extrapolation ratio of {metrics.extrapolation_ratio:.2f} indicates that performance "
            f"{'degrades significantly' if metrics.extrapolation_ratio > 2 else 'degrades moderately' if metrics.extrapolation_ratio > 1.5 else 'remains relatively stable'} "
            f"on sequences longer than those seen during training.",
        ])

        return "\n".join(lines)

    def _generate_plots(
        self,
        metrics: LengthGeneralizationMetrics,
        experiment_name: str,
    ):
        """Generate visualization plots."""
        if not HAS_MATPLOTLIB:
            return

        # Plot 1: PPL vs Length
        fig, ax = plt.subplots(figsize=(10, 6))

        lengths = []
        ppls = []
        for name, result in metrics.results_by_length.items():
            if isinstance(result, dict) and result['perplexity'] < float('inf'):
                avg_len = (result['length_range'][0] + result['length_range'][1]) / 2
                lengths.append(avg_len)
                ppls.append(result['perplexity'])

        ax.semilogx(lengths, ppls, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Sequence Length', fontsize=12)
        ax.set_ylabel('Perplexity', fontsize=12)
        ax.set_title('Perplexity vs Sequence Length', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{experiment_name}_ppl_vs_length.png"), dpi=150)
        plt.close()

        # Plot 2: Length penalty comparison
        fig, ax = plt.subplots(figsize=(8, 6))

        penalties = [
            metrics.length_penalty_2k_8k,
            metrics.length_penalty_2k_16k,
            metrics.length_penalty_2k_32k,
        ]
        labels = ['2K→8K', '2K→16K', '2K→32K']

        bars = ax.bar(labels, penalties, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax.axhline(y=1.0, color='gray', linestyle='--', label='No penalty')
        ax.set_ylabel('Length Penalty (PPL ratio)', fontsize=12)
        ax.set_title('Length Penalty Analysis', fontsize=14)

        # Add value labels
        for bar, val in zip(bars, penalties):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{val:.2f}x', ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{experiment_name}_length_penalty.png"), dpi=150)
        plt.close()


def compare_methods(
    results: Dict[str, LengthGeneralizationMetrics],
    output_dir: str = "./results",
) -> str:
    """
    Compare multiple methods and generate comparison report.

    Args:
        results: Dict mapping method names to their metrics
        output_dir: Output directory for report

    Returns:
        Markdown comparison report
    """
    os.makedirs(output_dir, exist_ok=True)

    lines = [
        "# Method Comparison Report",
        "",
        "## Summary Table",
        "",
        "| Method | Avg PPL | Worst PPL | LP (2K→8K) | LP (2K→16K) | LP (2K→32K) | Slope |",
        "|--------|---------|-----------|------------|-------------|-------------|-------|",
    ]

    for method_name, metrics in results.items():
        lines.append(
            f"| {method_name} | {metrics.avg_perplexity:.2f} | {metrics.worst_case_perplexity:.2f} | "
            f"{metrics.length_penalty_2k_8k:.2f}x | {metrics.length_penalty_2k_16k:.2f}x | "
            f"{metrics.length_penalty_2k_32k:.2f}x | {metrics.degradation_slope:.4f} |"
        )

    # Find best method for each metric
    best_avg_ppl = min(results.items(), key=lambda x: x[1].avg_perplexity)
    best_worst_ppl = min(results.items(), key=lambda x: x[1].worst_case_perplexity)
    best_lp = min(results.items(), key=lambda x: x[1].length_penalty_2k_16k)

    lines.extend([
        "",
        "## Best Methods",
        "",
        f"- **Best Average PPL**: {best_avg_ppl[0]} ({best_avg_ppl[1].avg_perplexity:.2f})",
        f"- **Best Worst-Case PPL**: {best_worst_ppl[0]} ({best_worst_ppl[1].worst_case_perplexity:.2f})",
        f"- **Best Length Penalty**: {best_lp[0]} ({best_lp[1].length_penalty_2k_16k:.2f}x)",
    ])

    # Generate comparison plot
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        method_names = list(results.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(method_names)))

        # Plot 1: Average PPL
        avg_ppls = [results[m].avg_perplexity for m in method_names]
        axes[0].bar(method_names, avg_ppls, color=colors)
        axes[0].set_ylabel('Average Perplexity')
        axes[0].set_title('Average Perplexity')
        axes[0].tick_params(axis='x', rotation=45)

        # Plot 2: Length penalties
        x = np.arange(len(method_names))
        width = 0.25
        lp_8k = [results[m].length_penalty_2k_8k for m in method_names]
        lp_16k = [results[m].length_penalty_2k_16k for m in method_names]
        lp_32k = [results[m].length_penalty_2k_32k for m in method_names]

        axes[1].bar(x - width, lp_8k, width, label='2K→8K')
        axes[1].bar(x, lp_16k, width, label='2K→16K')
        axes[1].bar(x + width, lp_32k, width, label='2K→32K')
        axes[1].set_ylabel('Length Penalty')
        axes[1].set_title('Length Penalties')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(method_names, rotation=45)
        axes[1].legend()
        axes[1].axhline(y=1.0, color='gray', linestyle='--')

        # Plot 3: Degradation slope
        slopes = [results[m].degradation_slope for m in method_names]
        axes[2].bar(method_names, slopes, color=colors)
        axes[2].set_ylabel('Degradation Slope')
        axes[2].set_title('Degradation Slope (lower is better)')
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "method_comparison.png"), dpi=150)
        plt.close()

    report = "\n".join(lines)

    with open(os.path.join(output_dir, "method_comparison.md"), 'w') as f:
        f.write(report)

    return report


if __name__ == "__main__":
    # Test the evaluation infrastructure
    print("Testing Evaluation Infrastructure...")

    # Create mock metrics
    mock_results = {
        'eval_256_512': LengthEvaluationResult(
            length_range=(256, 512), num_samples=100, total_tokens=50000,
            avg_loss=2.5, perplexity=12.18, std_loss=0.3, min_loss=2.0, max_loss=3.0
        ),
        'eval_512_1024': LengthEvaluationResult(
            length_range=(512, 1024), num_samples=100, total_tokens=75000,
            avg_loss=2.6, perplexity=13.46, std_loss=0.35, min_loss=2.1, max_loss=3.2
        ),
        'eval_1024_2048': LengthEvaluationResult(
            length_range=(1024, 2048), num_samples=100, total_tokens=150000,
            avg_loss=2.7, perplexity=14.88, std_loss=0.4, min_loss=2.2, max_loss=3.5
        ),
        'eval_2048_4096': LengthEvaluationResult(
            length_range=(2048, 4096), num_samples=80, total_tokens=250000,
            avg_loss=2.9, perplexity=18.17, std_loss=0.5, min_loss=2.3, max_loss=4.0
        ),
        'eval_4096_8192': LengthEvaluationResult(
            length_range=(4096, 8192), num_samples=50, total_tokens=300000,
            avg_loss=3.2, perplexity=24.53, std_loss=0.6, min_loss=2.5, max_loss=4.5
        ),
    }

    mock_metrics = LengthGeneralizationMetrics(
        results_by_length={name: asdict(result) for name, result in mock_results.items()},
        length_penalty_2k_8k=24.53 / 14.88,
        length_penalty_2k_16k=35.0 / 14.88,
        length_penalty_2k_32k=50.0 / 14.88,
        degradation_slope=0.15,
        degradation_intercept=1.5,
        extrapolation_ratio=1.8,
        avg_perplexity=16.64,
        worst_case_perplexity=24.53,
        best_case_perplexity=12.18,
    )

    # Generate report
    reporter = EvaluationReporter("./test_results")
    report = reporter.generate_report(mock_metrics, "TestModel", "test_experiment")
    print(report)

    print("\nEvaluation Infrastructure test complete!")
