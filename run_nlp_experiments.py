#!/usr/bin/env python3
"""
Main Experiment Runner for Hierarchical DRO NLP Length Generalization

This script runs comprehensive experiments comparing:
1. Standard Fine-tuning (baseline)
2. RoPE + ALiBi position encodings
3. Group DRO across length groups
4. Hierarchical DRO (our method)

Usage:
    python run_nlp_experiments.py --experiment all --model_size small
    python run_nlp_experiments.py --experiment hdro --epochs 10
    python run_nlp_experiments.py --experiment baseline --eval_only
"""

import argparse
import json
import os
import sys
import time
import random
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# Local imports
from nlp_datasets import (
    TokenizerWrapper,
    LengthGroupedDataset,
    LengthBucketSampler,
    SyntheticLongDocumentDataset,
    collate_variable_length,
    create_dataloaders,
    _generate_synthetic_wiki_text,
)
from nlp_dataset_loaders import (
    load_dataset_by_name,
    list_available_datasets,
    get_dataset_info,
    collate_fn,
    LongDocumentDataset,
    SyntheticLengthDataset,
    DATASET_REGISTRY,
)
from transformer_lm import (
    TransformerConfig,
    TransformerLanguageModel,
    create_model,
    count_parameters,
)
from hierarchical_perturbations import (
    PerturbationConfig,
    HierarchicalPerturbationManager,
)
from hierarchical_dro_nlp import (
    HierarchicalDROConfig,
    HierarchicalDROLoss,
    SimplifiedHierarchicalDROLoss,
    GroupDROLengthLoss,
    LengthRobustTrainer,
)
from nlp_evaluation import (
    LengthGeneralizationEvaluator,
    LengthGeneralizationMetrics,
    LengthEvaluationResult,
    EvaluationReporter,
    compare_methods,
)


@dataclass
class ExperimentConfig:
    """Configuration for experiment."""
    # Experiment settings
    experiment_name: str = "hdro_length_gen"
    seed: int = 42
    output_dir: str = "./results"

    # Model settings
    model_size: str = "small"  # tiny, small, medium, large
    vocab_size: int = 10000
    max_seq_len: int = 8192
    position_encoding: str = "rope"

    # Training settings
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Data settings
    dataset: str = "synthetic"  # synthetic, wikitext-2, wikitext-103, pg19, arxiv, etc.
    train_length_range: Tuple[int, int] = (512, 2048)
    num_train_samples: int = 10000
    num_eval_samples: int = 1000
    cache_dir: str = "./data"
    streaming: bool = False  # For large datasets like C4

    # DRO settings
    local_epsilon: float = 0.1
    chunk_epsilon: float = 0.2
    global_epsilon: float = 0.5
    consistency_weight: float = 0.1

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_start_ratio: float = 0.3

    # Evaluation
    eval_every: int = 500
    save_every: int = 1000

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Baseline Methods
# =============================================================================

class StandardFineTuning:
    """Standard fine-tuning baseline without DRO."""

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loaders: Dict[str, DataLoader],
    ):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.eval_loaders = eval_loaders

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps = len(train_loader) * config.num_epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

    def train(self) -> Dict:
        """Train with standard fine-tuning."""
        self.model.train()
        global_step = 0
        train_losses = []
        best_eval_loss = float('inf')

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            num_batches = 0

            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.config.device)

                self.optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs['loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                train_losses.append(loss.item())

                if global_step % self.config.eval_every == 0:
                    eval_results = self._evaluate()
                    avg_loss = sum(eval_results.values()) / len(eval_results)
                    if avg_loss < best_eval_loss:
                        best_eval_loss = avg_loss
                    self.model.train()

            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1}/{self.config.num_epochs} | Loss: {avg_epoch_loss:.4f}")

        return {'train_losses': train_losses, 'best_eval_loss': best_eval_loss}

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate on all eval loaders."""
        self.model.eval()
        results = {}

        for name, loader in self.eval_loaders.items():
            total_loss = 0
            total_tokens = 0

            for batch in loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.config.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                num_tokens = attention_mask.sum().item() if attention_mask is not None else input_ids.numel()
                total_loss += outputs['loss'].item() * num_tokens
                total_tokens += num_tokens

            results[name] = total_loss / total_tokens if total_tokens > 0 else float('inf')

        return results


class RoPEALiBiBaseline:
    """
    Baseline combining RoPE and ALiBi for length generalization.

    This represents the state-of-the-art in position encoding methods.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        train_loader: DataLoader,
        eval_loaders: Dict[str, DataLoader],
    ):
        self.config = config
        self.train_loader = train_loader
        self.eval_loaders = eval_loaders

        # Create model with RoPE + scaling
        rope_scaling = {
            'type': 'yarn',  # YaRN for better extrapolation
            'factor': 4.0,    # Support 4x longer sequences
        }

        self.model = create_model(
            config.model_size,
            position_encoding='rope',
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            rope_scaling=rope_scaling,
        ).to(config.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps = len(train_loader) * config.num_epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

    def train(self) -> Dict:
        """Train with RoPE+YaRN scaling."""
        self.model.train()
        train_losses = []
        best_eval_loss = float('inf')
        global_step = 0

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            num_batches = 0

            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.config.device)

                self.optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs['loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                train_losses.append(loss.item())

                if global_step % self.config.eval_every == 0:
                    eval_results = self._evaluate()
                    avg_loss = sum(eval_results.values()) / len(eval_results)
                    if avg_loss < best_eval_loss:
                        best_eval_loss = avg_loss
                    self.model.train()

            print(f"Epoch {epoch + 1}/{self.config.num_epochs} | Loss: {epoch_loss / num_batches:.4f}")

        return {'train_losses': train_losses, 'best_eval_loss': best_eval_loss}

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        self.model.eval()
        results = {}

        for name, loader in self.eval_loaders.items():
            total_loss = 0
            total_tokens = 0

            for batch in loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.config.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                num_tokens = attention_mask.sum().item() if attention_mask is not None else input_ids.numel()
                total_loss += outputs['loss'].item() * num_tokens
                total_tokens += num_tokens

            results[name] = total_loss / total_tokens if total_tokens > 0 else float('inf')

        return results


class GroupDROBaseline:
    """
    Group DRO baseline treating length groups as demographic groups.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loaders: Dict[str, DataLoader],
    ):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.eval_loaders = eval_loaders

        self.group_dro_loss = GroupDROLengthLoss(
            length_groups=[
                (0, 512),
                (512, 1024),
                (1024, 2048),
                (2048, 4096),
                (4096, 8192),
            ],
            adjustment_coefficient=1.0,
        )

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps = len(train_loader) * config.num_epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

    def train(self) -> Dict:
        """Train with Group DRO."""
        self.model.train()
        train_losses = []
        best_eval_loss = float('inf')
        global_step = 0

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            num_batches = 0

            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.config.device)
                lengths = batch.get('lengths')
                if lengths is not None:
                    lengths = lengths.to(self.config.device)

                self.optimizer.zero_grad()

                loss = self.group_dro_loss(
                    self.model, input_ids, attention_mask, lengths=lengths
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                train_losses.append(loss.item())

                if global_step % self.config.eval_every == 0:
                    eval_results = self._evaluate()
                    avg_loss = sum(eval_results.values()) / len(eval_results)
                    if avg_loss < best_eval_loss:
                        best_eval_loss = avg_loss
                    self.model.train()

            print(f"Epoch {epoch + 1}/{self.config.num_epochs} | Loss: {epoch_loss / num_batches:.4f}")

        return {'train_losses': train_losses, 'best_eval_loss': best_eval_loss}

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        self.model.eval()
        results = {}

        for name, loader in self.eval_loaders.items():
            total_loss = 0
            total_tokens = 0

            for batch in loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.config.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                num_tokens = attention_mask.sum().item() if attention_mask is not None else input_ids.numel()
                total_loss += outputs['loss'].item() * num_tokens
                total_tokens += num_tokens

            results[name] = total_loss / total_tokens if total_tokens > 0 else float('inf')

        return results


class HierarchicalDROExperiment:
    """
    Main Hierarchical DRO experiment.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loaders: Dict[str, DataLoader],
    ):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.eval_loaders = eval_loaders

        # Use simplified H-DRO for faster experimentation
        self.hdro_loss = SimplifiedHierarchicalDROLoss(
            local_weight=1.0,
            chunk_weight=1.0,
            global_weight=1.0,
            dropout_prob=config.local_epsilon,
        )

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps = len(train_loader) * config.num_epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

    def train(self) -> Dict:
        """Train with Hierarchical DRO."""
        self.model.train()
        train_losses = []
        best_eval_loss = float('inf')
        global_step = 0

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            num_batches = 0

            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.config.device)

                self.optimizer.zero_grad()

                loss = self.hdro_loss(self.model, input_ids, attention_mask)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                train_losses.append(loss.item())

                if global_step % self.config.eval_every == 0:
                    eval_results = self._evaluate()
                    avg_loss = sum(eval_results.values()) / len(eval_results)
                    if avg_loss < best_eval_loss:
                        best_eval_loss = avg_loss
                    self.model.train()

                # Update curriculum if applicable
                if self.config.use_curriculum:
                    progress = global_step / (len(self.train_loader) * self.config.num_epochs)
                    # Could update train_loader's sampler here

            print(f"Epoch {epoch + 1}/{self.config.num_epochs} | Loss: {epoch_loss / num_batches:.4f}")

        return {'train_losses': train_losses, 'best_eval_loss': best_eval_loss}

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        self.model.eval()
        results = {}

        for name, loader in self.eval_loaders.items():
            total_loss = 0
            total_tokens = 0

            for batch in loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.config.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                num_tokens = attention_mask.sum().item() if attention_mask is not None else input_ids.numel()
                total_loss += outputs['loss'].item() * num_tokens
                total_tokens += num_tokens

            results[name] = total_loss / total_tokens if total_tokens > 0 else float('inf')

        return results


# =============================================================================
# Main Experiment Runner
# =============================================================================

def create_datasets_and_loaders(config: ExperimentConfig) -> Tuple[DataLoader, Dict[str, DataLoader], TokenizerWrapper]:
    """Create datasets and dataloaders for experiments."""
    print("Creating tokenizer and datasets...")
    print(f"Dataset: {config.dataset}")

    # Create tokenizer
    tokenizer = TokenizerWrapper("simple", vocab_size=config.vocab_size)

    # Length buckets for evaluation
    length_buckets = [
        (64, 256),
        (256, 512),
        (512, 1024),
        (1024, 2048),
        (2048, 4096),
        (4096, 8192),
    ]

    # Load dataset based on configuration
    if config.dataset == "synthetic":
        # Generate synthetic text for tokenizer fitting
        synthetic_texts = _generate_synthetic_wiki_text(config.num_train_samples)
        tokenizer.fit(synthetic_texts)

        print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

        # Create training dataset
        train_dataset = LengthGroupedDataset(
            synthetic_texts[:config.num_train_samples],
            tokenizer,
            length_buckets=length_buckets,
            max_length=config.max_seq_len,
        )

        # Create evaluation texts
        eval_texts = _generate_synthetic_wiki_text(config.num_eval_samples)

    elif config.dataset in DATASET_REGISTRY:
        # Use dataset loaders for real datasets
        print(f"Loading {config.dataset} dataset...")
        info = get_dataset_info(config.dataset)
        print(f"  Description: {info.description}")
        print(f"  Avg length: {info.avg_length} tokens")

        # For real datasets, generate synthetic data for tokenizer fitting
        # In practice, you'd use a pre-trained tokenizer (GPT-2, etc.)
        synthetic_for_tokenizer = _generate_synthetic_wiki_text(5000)
        tokenizer.fit(synthetic_for_tokenizer)

        try:
            train_dataset = load_dataset_by_name(
                config.dataset,
                tokenizer,
                split="train",
                max_length=config.max_seq_len,
                max_samples=config.num_train_samples,
                cache_dir=config.cache_dir,
                streaming=config.streaming,
            )
            print(f"Loaded training dataset: {len(train_dataset)} samples")
            if hasattr(train_dataset, 'get_length_distribution'):
                print(f"Length distribution: {train_dataset.get_length_distribution()}")

            # Use same dataset for eval with different split
            eval_texts = None  # Will use dataset directly
        except Exception as e:
            print(f"Warning: Could not load {config.dataset}: {e}")
            print("Falling back to synthetic data...")
            synthetic_texts = _generate_synthetic_wiki_text(config.num_train_samples)
            tokenizer.fit(synthetic_texts)
            train_dataset = LengthGroupedDataset(
                synthetic_texts[:config.num_train_samples],
                tokenizer,
                length_buckets=length_buckets,
                max_length=config.max_seq_len,
            )
            eval_texts = _generate_synthetic_wiki_text(config.num_eval_samples)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}. "
                        f"Available: synthetic, {', '.join(DATASET_REGISTRY.keys())}")

    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    print(f"Training dataset: {len(train_dataset)} samples")

    # Create training dataloader with curriculum
    if hasattr(train_dataset, 'length_to_indices'):
        # LongDocumentDataset from nlp_dataset_loaders
        from torch.utils.data import SubsetRandomSampler

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
            shuffle=True,
            num_workers=0,
        )
    else:
        # LengthGroupedDataset from nlp_datasets
        train_sampler = LengthBucketSampler(
            train_dataset,
            batch_size=config.batch_size,
            curriculum=config.use_curriculum,
            curriculum_progress=config.curriculum_start_ratio if config.use_curriculum else 1.0,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=lambda b: collate_variable_length(b, tokenizer.pad_token_id),
            num_workers=0,
        )

    # Create evaluation dataloaders
    eval_loaders = {}

    if eval_texts is not None:
        # Using synthetic eval texts
        for min_len, max_len in length_buckets:
            eval_dataset = LengthGroupedDataset(
                eval_texts,
                tokenizer,
                length_buckets=[(min_len, max_len)],
                max_length=max_len,
            )

            if len(eval_dataset) > 0:
                eval_loader = DataLoader(
                    eval_dataset,
                    batch_size=config.batch_size,
                    collate_fn=lambda b: collate_variable_length(b, tokenizer.pad_token_id),
                    shuffle=False,
                    num_workers=0,
                )
                eval_loaders[f'eval_{min_len}_{max_len}'] = eval_loader
    else:
        # Create eval loaders from real dataset validation split
        try:
            eval_dataset = load_dataset_by_name(
                config.dataset,
                tokenizer,
                split="validation",
                max_length=config.max_seq_len,
                max_samples=config.num_eval_samples,
                cache_dir=config.cache_dir,
            )

            # Create length-stratified eval loaders
            if hasattr(eval_dataset, 'get_samples_by_length'):
                for min_len, max_len in length_buckets:
                    indices = eval_dataset.get_samples_by_length(min_len, max_len)
                    if len(indices) > 0:
                        subset = torch.utils.data.Subset(eval_dataset, indices)
                        eval_loader = DataLoader(
                            subset,
                            batch_size=config.batch_size,
                            collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
                            shuffle=False,
                            num_workers=0,
                        )
                        eval_loaders[f'eval_{min_len}_{max_len}'] = eval_loader
            else:
                # Single eval loader for all lengths
                eval_loader = DataLoader(
                    eval_dataset,
                    batch_size=config.batch_size,
                    collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
                    shuffle=False,
                    num_workers=0,
                )
                eval_loaders['eval_all'] = eval_loader

        except Exception as e:
            print(f"Warning: Could not load validation split: {e}")
            # Fall back to using portion of train for eval
            print("Using synthetic data for evaluation...")
            eval_texts = _generate_synthetic_wiki_text(config.num_eval_samples)
            for min_len, max_len in length_buckets:
                eval_dataset = LengthGroupedDataset(
                    eval_texts,
                    tokenizer,
                    length_buckets=[(min_len, max_len)],
                    max_length=max_len,
                )
                if len(eval_dataset) > 0:
                    eval_loader = DataLoader(
                        eval_dataset,
                        batch_size=config.batch_size,
                        collate_fn=lambda b: collate_variable_length(b, tokenizer.pad_token_id),
                        shuffle=False,
                        num_workers=0,
                    )
                    eval_loaders[f'eval_{min_len}_{max_len}'] = eval_loader

    print(f"Created {len(eval_loaders)} evaluation dataloaders")

    return train_loader, eval_loaders, tokenizer


def run_experiment(
    method: str,
    config: ExperimentConfig,
    train_loader: DataLoader,
    eval_loaders: Dict[str, DataLoader],
    tokenizer: TokenizerWrapper,
) -> Tuple[nn.Module, Dict, LengthGeneralizationMetrics]:
    """
    Run a single experiment.

    Args:
        method: Method name ('standard', 'rope_alibi', 'group_dro', 'hdro')
        config: Experiment configuration
        train_loader: Training dataloader
        eval_loaders: Evaluation dataloaders
        tokenizer: Tokenizer

    Returns:
        Trained model, training results, evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {method}")
    print(f"{'='*60}")

    # Create model
    if method == 'rope_alibi':
        # RoPE+ALiBi baseline creates its own model with special config
        experiment = RoPEALiBiBaseline(config, train_loader, eval_loaders)
        model = experiment.model
    else:
        model = create_model(
            config.model_size,
            position_encoding=config.position_encoding,
            vocab_size=tokenizer.vocab_size,
            max_seq_len=config.max_seq_len,
        )
        print(f"Model parameters: {count_parameters(model):,}")

        if method == 'standard':
            experiment = StandardFineTuning(config, model, train_loader, eval_loaders)
        elif method == 'group_dro':
            experiment = GroupDROBaseline(config, model, train_loader, eval_loaders)
        elif method == 'hdro':
            experiment = HierarchicalDROExperiment(config, model, train_loader, eval_loaders)
        else:
            raise ValueError(f"Unknown method: {method}")

    # Train
    start_time = time.time()
    train_results = experiment.train()
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f} seconds")

    # Evaluate
    print("\nRunning final evaluation...")
    if method == 'rope_alibi':
        model = experiment.model

    evaluator = LengthGeneralizationEvaluator(
        model,
        tokenizer,
        device=config.device,
    )

    metrics = evaluator.evaluate_all_lengths(eval_loaders)

    print(f"\nResults for {method}:")
    print(f"  Average Perplexity: {metrics.avg_perplexity:.2f}")
    print(f"  Worst Perplexity: {metrics.worst_case_perplexity:.2f}")
    print(f"  Length Penalty (2K→8K): {metrics.length_penalty_2k_8k:.2f}x")
    print(f"  Degradation Slope: {metrics.degradation_slope:.4f}")

    return model, train_results, metrics


def run_all_experiments(config: ExperimentConfig):
    """Run all experiments and generate comparison."""
    set_seed(config.seed)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f"nlp_hdro_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(asdict(config), f, indent=2, default=str)

    # Create datasets
    train_loader, eval_loaders, tokenizer = create_datasets_and_loaders(config)

    # Run all methods
    methods = ['standard', 'rope_alibi', 'group_dro', 'hdro']
    all_results = {}
    all_metrics = {}

    for method in methods:
        try:
            model, results, metrics = run_experiment(
                method, config, train_loader, eval_loaders, tokenizer
            )
            all_results[method] = results
            all_metrics[method] = metrics

            # Save model
            torch.save(model.state_dict(), os.path.join(output_dir, f"{method}_model.pt"))

            # Generate individual report
            reporter = EvaluationReporter(output_dir)
            reporter.generate_report(metrics, f"{method}_model", method)

        except Exception as e:
            print(f"Error running {method}: {e}")
            import traceback
            traceback.print_exc()

    # Generate comparison report
    if len(all_metrics) > 1:
        comparison = compare_methods(all_metrics, output_dir)
        print("\n" + comparison)

    # Save all results
    with open(os.path.join(output_dir, "all_results.json"), 'w') as f:
        json.dump({
            method: {
                'train_losses': results.get('train_losses', [])[-10:],  # Last 10 losses
                'best_eval_loss': results.get('best_eval_loss', float('inf')),
            }
            for method, results in all_results.items()
        }, f, indent=2)

    print(f"\nAll results saved to: {output_dir}")
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Run NLP Length Generalization Experiments")

    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'standard', 'rope_alibi', 'group_dro', 'hdro'],
                       help='Which experiment to run')
    parser.add_argument('--model_size', type=str, default='tiny',
                       choices=['tiny', 'small', 'medium', 'large'],
                       help='Model size')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--vocab_size', type=int, default=5000,
                       help='Vocabulary size')
    parser.add_argument('--max_seq_len', type=int, default=4096,
                       help='Maximum sequence length')
    parser.add_argument('--num_train_samples', type=int, default=1000,
                       help='Number of training samples')
    parser.add_argument('--num_eval_samples', type=int, default=200,
                       help='Number of evaluation samples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    parser.add_argument('--eval_every', type=int, default=200,
                       help='Evaluate every N steps')
    parser.add_argument('--use_curriculum', action='store_true',
                       help='Use curriculum learning')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       help='Dataset to use (synthetic, wikitext-2, wikitext-103, pg19, arxiv, '
                            'govreport, booksum, openwebtext, c4, scrolls-qasper, etc.)')
    parser.add_argument('--cache_dir', type=str, default='./data',
                       help='Cache directory for datasets')
    parser.add_argument('--streaming', action='store_true',
                       help='Use streaming mode for large datasets')
    parser.add_argument('--list_datasets', action='store_true',
                       help='List all available datasets and exit')

    args = parser.parse_args()

    # List datasets if requested
    if args.list_datasets:
        print("\nAvailable Datasets for Length Generalization Experiments:")
        print("=" * 70)
        for name, info in DATASET_REGISTRY.items():
            print(f"\n{name}:")
            print(f"  Description: {info.description}")
            print(f"  Avg Length: {info.avg_length:,} tokens")
            print(f"  Max Length: {info.max_length:,} tokens")
            print(f"  Domains: {', '.join(info.domains)}")
            print(f"  Size: {info.download_size}")
            if info.requires_auth:
                print(f"  ⚠️  Requires authentication")
        print("\n" + "=" * 70)
        print("Usage: python run_nlp_experiments.py --dataset wikitext-103 --experiment hdro")
        return

    # Create config
    config = ExperimentConfig(
        experiment_name=f"nlp_hdro_{args.experiment}_{args.dataset}",
        seed=args.seed,
        output_dir=args.output_dir,
        model_size=args.model_size,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        num_train_samples=args.num_train_samples,
        num_eval_samples=args.num_eval_samples,
        eval_every=args.eval_every,
        use_curriculum=args.use_curriculum,
        device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu'),
        dataset=args.dataset,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
    )

    print("=" * 60)
    print("NLP Length Generalization Experiments")
    print("Hierarchical DRO for Robust Length Extrapolation")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Dataset: {config.dataset}")
    print(f"Model size: {config.model_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Max sequence length: {config.max_seq_len}")
    print("=" * 60)

    if args.experiment == 'all':
        run_all_experiments(config)
    else:
        set_seed(config.seed)
        train_loader, eval_loaders, tokenizer = create_datasets_and_loaders(config)
        model, results, metrics = run_experiment(
            args.experiment, config, train_loader, eval_loaders, tokenizer
        )

        # Save results
        output_dir = os.path.join(config.output_dir, f"nlp_{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(output_dir, exist_ok=True)

        reporter = EvaluationReporter(output_dir)
        reporter.generate_report(metrics, f"{args.experiment}_model", args.experiment)

        torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
        print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
