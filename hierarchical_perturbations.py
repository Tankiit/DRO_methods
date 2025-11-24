"""
Hierarchical Perturbation System for Length Generalization DRO

This module implements three levels of perturbations:
1. Local (Token-Level): Position noise, attention masking, token dropout
2. Chunk (Paragraph-Level): Paragraph shuffling, topic drift, coherence noise
3. Global (Document-Level): Length scaling, structure perturbation, domain shift

Each level defines an uncertainty set for distributionally robust optimization.
"""

import math
import random
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class PerturbationConfig:
    """Configuration for hierarchical perturbations."""
    # Local perturbation parameters
    local_epsilon: float = 0.1  # Uncertainty radius for local
    position_noise_std: float = 0.05
    attention_mask_prob: float = 0.1
    token_dropout_prob: float = 0.05
    token_swap_prob: float = 0.03

    # Chunk perturbation parameters
    chunk_epsilon: float = 0.2  # Uncertainty radius for chunk
    paragraph_shuffle_prob: float = 0.15
    topic_drift_prob: float = 0.1
    coherence_noise_std: float = 0.1
    chunk_size: int = 512  # Tokens per chunk

    # Global perturbation parameters
    global_epsilon: float = 0.5  # Uncertainty radius for global
    length_scale_range: Tuple[float, float] = (0.5, 2.0)
    structure_perturb_prob: float = 0.2
    domain_shift_strength: float = 0.3


class BasePerturbation(ABC):
    """Abstract base class for perturbations."""

    @abstractmethod
    def apply(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply perturbation to input."""
        pass

    @abstractmethod
    def compute_distance(self, x_orig: torch.Tensor, x_pert: torch.Tensor) -> float:
        """Compute distance between original and perturbed samples."""
        pass


# =============================================================================
# Level 1: Local (Token-Level) Perturbations
# =============================================================================

class LocalPerturbation(BasePerturbation):
    """
    Token-level perturbations affecting local attention patterns and positions.

    Perturbation types:
    - Position noise: Add noise to position encodings
    - Attention masking: Randomly mask attention weights
    - Token dropout: Remove or replace tokens
    - Token swap: Swap adjacent tokens
    """

    def __init__(self, config: PerturbationConfig, vocab_size: int = 50000):
        self.config = config
        self.vocab_size = vocab_size

    def apply(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        perturbation_type: str = "all",
    ) -> Dict[str, torch.Tensor]:
        """
        Apply local perturbations.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            position_ids: Position IDs (batch, seq_len)
            hidden_states: Hidden representations (batch, seq_len, hidden_dim)
            perturbation_type: Type of perturbation ('position', 'attention', 'token', 'all')

        Returns:
            Dictionary with perturbed tensors
        """
        result = {
            'input_ids': input_ids.clone(),
            'attention_mask': attention_mask.clone() if attention_mask is not None else None,
            'position_ids': position_ids.clone() if position_ids is not None else None,
            'hidden_states': hidden_states.clone() if hidden_states is not None else None,
        }

        if perturbation_type in ['position', 'all']:
            result = self._apply_position_noise(result)

        if perturbation_type in ['attention', 'all']:
            result = self._apply_attention_masking(result)

        if perturbation_type in ['token', 'all']:
            result = self._apply_token_perturbations(result)

        return result

    def _apply_position_noise(self, data: Dict) -> Dict:
        """Add noise to position representations."""
        if data['hidden_states'] is not None:
            # Add Gaussian noise to hidden states (simulating position noise)
            noise = torch.randn_like(data['hidden_states']) * self.config.position_noise_std
            # Scale noise by position (more noise at later positions for length extrapolation)
            seq_len = data['hidden_states'].size(1)
            position_scale = torch.linspace(1.0, 2.0, seq_len, device=data['hidden_states'].device)
            position_scale = position_scale.view(1, -1, 1)
            data['hidden_states'] = data['hidden_states'] + noise * position_scale

        if data['position_ids'] is not None:
            # Add discrete noise to position IDs
            noise = torch.randint(-2, 3, data['position_ids'].shape, device=data['position_ids'].device)
            data['position_ids'] = torch.clamp(data['position_ids'] + noise, min=0)

        return data

    def _apply_attention_masking(self, data: Dict) -> Dict:
        """Apply random attention masking."""
        if data['attention_mask'] is not None:
            # Randomly set some attention positions to 0
            mask_prob = torch.rand_like(data['attention_mask'].float())
            drop_mask = mask_prob < self.config.attention_mask_prob
            # Don't drop the first token (usually BOS) and last token
            drop_mask[:, 0] = False
            drop_mask[:, -1] = False
            data['attention_mask'] = data['attention_mask'] * (~drop_mask).long()

        return data

    def _apply_token_perturbations(self, data: Dict) -> Dict:
        """Apply token dropout and swapping."""
        input_ids = data['input_ids']
        batch_size, seq_len = input_ids.shape

        # Token dropout (replace with random token)
        dropout_mask = torch.rand(batch_size, seq_len, device=input_ids.device) < self.config.token_dropout_prob
        # Don't drop special tokens (assume first few IDs are special)
        dropout_mask = dropout_mask & (input_ids >= 4)
        random_tokens = torch.randint(4, self.vocab_size, (batch_size, seq_len), device=input_ids.device)
        input_ids = torch.where(dropout_mask, random_tokens, input_ids)

        # Token swap (swap adjacent tokens)
        for b in range(batch_size):
            for i in range(1, seq_len - 2):
                if random.random() < self.config.token_swap_prob:
                    input_ids[b, i], input_ids[b, i + 1] = input_ids[b, i + 1].item(), input_ids[b, i].item()

        data['input_ids'] = input_ids
        return data

    def compute_distance(self, x_orig: torch.Tensor, x_pert: torch.Tensor) -> float:
        """Compute Wasserstein-like distance for local perturbations."""
        # For token sequences, use edit distance normalized by length
        if x_orig.dim() == 1:
            x_orig = x_orig.unsqueeze(0)
            x_pert = x_pert.unsqueeze(0)

        total_dist = 0.0
        for orig, pert in zip(x_orig, x_pert):
            # Count mismatches
            mismatches = (orig != pert).float().sum().item()
            total_dist += mismatches / len(orig)

        return total_dist / len(x_orig)


class PositionEncodingPerturbation(nn.Module):
    """
    Learnable perturbation for position encodings.

    This module learns to generate adversarial position perturbations
    within the local uncertainty set.
    """

    def __init__(
        self,
        max_seq_len: int = 8192,
        hidden_dim: int = 768,
        epsilon: float = 0.1,
    ):
        super().__init__()
        self.epsilon = epsilon

        # Learnable perturbation parameters
        self.perturbation_scale = nn.Parameter(torch.ones(max_seq_len, hidden_dim) * 0.01)
        self.perturbation_direction = nn.Parameter(torch.randn(max_seq_len, hidden_dim))

    def forward(self, position_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable position perturbation.

        Args:
            position_embeddings: Position embeddings (batch, seq_len, hidden_dim)

        Returns:
            Perturbed position embeddings
        """
        seq_len = position_embeddings.size(1)

        # Normalize direction to unit ball
        direction = F.normalize(self.perturbation_direction[:seq_len], dim=-1)

        # Scale to stay within epsilon ball
        scale = torch.tanh(self.perturbation_scale[:seq_len]) * self.epsilon

        # Apply perturbation
        perturbation = direction * scale
        return position_embeddings + perturbation.unsqueeze(0)


# =============================================================================
# Level 2: Chunk (Paragraph-Level) Perturbations
# =============================================================================

class ChunkPerturbation(BasePerturbation):
    """
    Paragraph-level perturbations affecting medium-range coherence.

    Perturbation types:
    - Paragraph shuffling: Reorder paragraphs
    - Topic drift: Inject off-topic content
    - Coherence noise: Perturb sentence transitions
    """

    def __init__(self, config: PerturbationConfig, vocab_size: int = 50000):
        self.config = config
        self.vocab_size = vocab_size

    def apply(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk_boundaries: Optional[List[List[int]]] = None,
        hidden_states: Optional[torch.Tensor] = None,
        perturbation_type: str = "all",
    ) -> Dict[str, torch.Tensor]:
        """
        Apply chunk-level perturbations.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            chunk_boundaries: List of chunk boundary positions per batch item
            hidden_states: Hidden representations
            perturbation_type: Type of perturbation

        Returns:
            Dictionary with perturbed tensors
        """
        result = {
            'input_ids': input_ids.clone(),
            'attention_mask': attention_mask.clone() if attention_mask is not None else None,
            'hidden_states': hidden_states.clone() if hidden_states is not None else None,
        }

        # Infer chunk boundaries if not provided
        if chunk_boundaries is None:
            chunk_boundaries = self._infer_chunk_boundaries(input_ids)

        if perturbation_type in ['shuffle', 'all']:
            result = self._apply_paragraph_shuffle(result, chunk_boundaries)

        if perturbation_type in ['drift', 'all']:
            result = self._apply_topic_drift(result, chunk_boundaries)

        if perturbation_type in ['coherence', 'all']:
            result = self._apply_coherence_noise(result, chunk_boundaries)

        return result

    def _infer_chunk_boundaries(self, input_ids: torch.Tensor) -> List[List[int]]:
        """Infer chunk boundaries based on fixed chunk size."""
        batch_size, seq_len = input_ids.shape
        boundaries = []

        for b in range(batch_size):
            batch_boundaries = list(range(0, seq_len, self.config.chunk_size))
            if batch_boundaries[-1] != seq_len:
                batch_boundaries.append(seq_len)
            boundaries.append(batch_boundaries)

        return boundaries

    def _apply_paragraph_shuffle(
        self,
        data: Dict,
        chunk_boundaries: List[List[int]],
    ) -> Dict:
        """Shuffle paragraphs/chunks within documents."""
        input_ids = data['input_ids']
        batch_size = input_ids.size(0)

        for b in range(batch_size):
            if random.random() < self.config.paragraph_shuffle_prob:
                boundaries = chunk_boundaries[b]
                if len(boundaries) < 3:
                    continue

                # Extract chunks
                chunks = []
                for i in range(len(boundaries) - 1):
                    start, end = boundaries[i], boundaries[i + 1]
                    chunks.append(input_ids[b, start:end].clone())

                # Shuffle (keeping first and last in place)
                if len(chunks) > 2:
                    middle = chunks[1:-1]
                    random.shuffle(middle)
                    chunks = [chunks[0]] + middle + [chunks[-1]]

                # Reconstruct
                pos = 0
                for chunk in chunks:
                    chunk_len = len(chunk)
                    input_ids[b, pos:pos + chunk_len] = chunk
                    pos += chunk_len

        data['input_ids'] = input_ids
        return data

    def _apply_topic_drift(
        self,
        data: Dict,
        chunk_boundaries: List[List[int]],
    ) -> Dict:
        """Inject topic drift by replacing chunk content."""
        input_ids = data['input_ids']
        batch_size = input_ids.size(0)

        for b in range(batch_size):
            boundaries = chunk_boundaries[b]

            for i in range(len(boundaries) - 1):
                if random.random() < self.config.topic_drift_prob:
                    start, end = boundaries[i], boundaries[i + 1]
                    chunk_len = end - start

                    # Replace portion of chunk with random tokens (topic drift)
                    drift_len = int(chunk_len * 0.3)  # 30% of chunk
                    drift_start = start + random.randint(0, chunk_len - drift_len)

                    random_tokens = torch.randint(
                        4, self.vocab_size,
                        (drift_len,),
                        device=input_ids.device
                    )
                    input_ids[b, drift_start:drift_start + drift_len] = random_tokens

        data['input_ids'] = input_ids
        return data

    def _apply_coherence_noise(
        self,
        data: Dict,
        chunk_boundaries: List[List[int]],
    ) -> Dict:
        """Apply noise at chunk boundaries to perturb coherence."""
        if data['hidden_states'] is None:
            return data

        hidden_states = data['hidden_states']
        batch_size = hidden_states.size(0)

        for b in range(batch_size):
            boundaries = chunk_boundaries[b]

            for boundary in boundaries[1:-1]:  # Skip first and last
                if boundary < hidden_states.size(1):
                    # Add noise around boundary
                    window = 10
                    start = max(0, boundary - window)
                    end = min(hidden_states.size(1), boundary + window)

                    noise = torch.randn_like(hidden_states[b, start:end])
                    hidden_states[b, start:end] += noise * self.config.coherence_noise_std

        data['hidden_states'] = hidden_states
        return data

    def compute_distance(self, x_orig: torch.Tensor, x_pert: torch.Tensor) -> float:
        """Compute Earth Mover's Distance for chunk perturbations."""
        # Simplified: use chunk-level feature distance
        if x_orig.dim() == 2:  # (batch, seq_len)
            # Convert to chunk representations
            chunk_size = self.config.chunk_size
            orig_chunks = x_orig.view(x_orig.size(0), -1, chunk_size)
            pert_chunks = x_pert.view(x_pert.size(0), -1, chunk_size)

            # Compute chunk-wise distances
            distances = (orig_chunks != pert_chunks).float().mean(dim=-1)
            return distances.mean().item()
        else:
            # For hidden states, use L2 distance
            return torch.norm(x_orig - x_pert, p=2).item() / x_orig.numel()


class ChunkEmbeddingPerturbation(nn.Module):
    """
    Learnable chunk-level perturbation for embeddings.

    Learns to generate adversarial perturbations at the chunk level.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_chunks: int = 16,
        epsilon: float = 0.2,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.num_chunks = num_chunks

        # Learnable chunk-level perturbations
        self.chunk_perturbations = nn.Parameter(torch.randn(num_chunks, hidden_dim) * 0.01)

    def forward(
        self,
        hidden_states: torch.Tensor,
        chunk_boundaries: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Apply chunk-level perturbation to hidden states.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            chunk_boundaries: Chunk boundary positions

        Returns:
            Perturbed hidden states
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Default: equal-sized chunks
        if chunk_boundaries is None:
            chunk_size = seq_len // self.num_chunks
            chunk_boundaries = list(range(0, seq_len + 1, max(1, chunk_size)))

        perturbed = hidden_states.clone()

        for i in range(min(len(chunk_boundaries) - 1, self.num_chunks)):
            start = chunk_boundaries[i]
            end = chunk_boundaries[i + 1] if i + 1 < len(chunk_boundaries) else seq_len

            # Normalize perturbation
            pert = F.normalize(self.chunk_perturbations[i], dim=0) * self.epsilon
            perturbed[:, start:end] += pert.unsqueeze(0).unsqueeze(0)

        return perturbed


# =============================================================================
# Level 3: Global (Document-Level) Perturbations
# =============================================================================

class GlobalPerturbation(BasePerturbation):
    """
    Document-level perturbations affecting long-range dependencies.

    Perturbation types:
    - Length scaling: Extend/compress documents
    - Structure perturbation: Add/remove sections
    - Domain shift: Change document style/domain
    """

    def __init__(self, config: PerturbationConfig, vocab_size: int = 50000):
        self.config = config
        self.vocab_size = vocab_size

    def apply(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        target_length: Optional[int] = None,
        perturbation_type: str = "all",
    ) -> Dict[str, torch.Tensor]:
        """
        Apply global perturbations.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask
            hidden_states: Hidden representations
            target_length: Target length for scaling
            perturbation_type: Type of perturbation

        Returns:
            Dictionary with perturbed tensors
        """
        result = {
            'input_ids': input_ids.clone(),
            'attention_mask': attention_mask.clone() if attention_mask is not None else None,
            'hidden_states': hidden_states.clone() if hidden_states is not None else None,
        }

        if perturbation_type in ['scale', 'all']:
            result = self._apply_length_scaling(result, target_length)

        if perturbation_type in ['structure', 'all']:
            result = self._apply_structure_perturbation(result)

        if perturbation_type in ['domain', 'all']:
            result = self._apply_domain_shift(result)

        return result

    def _apply_length_scaling(
        self,
        data: Dict,
        target_length: Optional[int] = None,
    ) -> Dict:
        """Scale document length through repetition or interpolation."""
        input_ids = data['input_ids']
        batch_size, seq_len = input_ids.shape

        if target_length is None:
            scale = random.uniform(*self.config.length_scale_range)
            target_length = int(seq_len * scale)

        if target_length == seq_len:
            return data

        new_input_ids = torch.zeros(batch_size, target_length, dtype=input_ids.dtype, device=input_ids.device)

        for b in range(batch_size):
            if target_length > seq_len:
                # Extend by repetition with variation
                repeats = (target_length + seq_len - 1) // seq_len
                extended = input_ids[b].repeat(repeats)[:target_length]

                # Add variation to repeated sections
                for i in range(seq_len, target_length, seq_len):
                    end_idx = min(i + seq_len, target_length)
                    section_len = end_idx - i
                    # Randomly perturb some tokens
                    mask = torch.rand(section_len, device=input_ids.device) < 0.1
                    random_tokens = torch.randint(4, self.vocab_size, (section_len,), device=input_ids.device)
                    extended[i:end_idx] = torch.where(mask, random_tokens, extended[i:end_idx])

                new_input_ids[b] = extended
            else:
                # Compress by sampling
                indices = torch.linspace(0, seq_len - 1, target_length).long()
                new_input_ids[b] = input_ids[b, indices]

        data['input_ids'] = new_input_ids

        # Update attention mask
        if data['attention_mask'] is not None:
            new_mask = torch.ones(batch_size, target_length, dtype=data['attention_mask'].dtype,
                                 device=data['attention_mask'].device)
            data['attention_mask'] = new_mask

        return data

    def _apply_structure_perturbation(self, data: Dict) -> Dict:
        """Perturb document structure by adding/removing sections."""
        input_ids = data['input_ids']
        batch_size, seq_len = input_ids.shape

        for b in range(batch_size):
            if random.random() < self.config.structure_perturb_prob:
                # Remove a random section
                section_size = random.randint(seq_len // 10, seq_len // 4)
                start = random.randint(seq_len // 4, seq_len - section_size - seq_len // 4)

                # Shift remaining content
                input_ids[b, start:seq_len - section_size] = input_ids[b, start + section_size:seq_len].clone()
                # Fill end with padding or repeated content
                input_ids[b, seq_len - section_size:] = input_ids[b, :section_size].clone()

        data['input_ids'] = input_ids
        return data

    def _apply_domain_shift(self, data: Dict) -> Dict:
        """Apply domain/style shift through vocabulary perturbation."""
        if data['hidden_states'] is None:
            return data

        hidden_states = data['hidden_states']

        # Apply global transformation to hidden states
        # This simulates a domain shift in representation space
        shift_direction = torch.randn(hidden_states.size(-1), device=hidden_states.device)
        shift_direction = F.normalize(shift_direction, dim=0)

        shift_magnitude = self.config.domain_shift_strength
        data['hidden_states'] = hidden_states + shift_direction * shift_magnitude

        return data

    def compute_distance(self, x_orig: torch.Tensor, x_pert: torch.Tensor) -> float:
        """Compute KL-divergence-like distance for global perturbations."""
        # Use length ratio and token distribution difference
        orig_len = x_orig.size(-1)
        pert_len = x_pert.size(-1)

        length_diff = abs(orig_len - pert_len) / max(orig_len, pert_len)

        # Token distribution difference
        if x_orig.dim() == 2:
            orig_dist = torch.bincount(x_orig.flatten(), minlength=100)[:100].float()
            pert_dist = torch.bincount(x_pert.flatten(), minlength=100)[:100].float()

            orig_dist = orig_dist / orig_dist.sum()
            pert_dist = pert_dist / pert_dist.sum()

            # KL divergence (with smoothing)
            eps = 1e-8
            kl = (orig_dist * torch.log((orig_dist + eps) / (pert_dist + eps))).sum()

            return length_diff + kl.item()
        else:
            return length_diff


class GlobalContextPerturbation(nn.Module):
    """
    Learnable global context perturbation.

    Learns adversarial global context shifts that affect long-range dependencies.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_global_vectors: int = 8,
        epsilon: float = 0.5,
    ):
        super().__init__()
        self.epsilon = epsilon

        # Global context vectors that get added based on document features
        self.global_vectors = nn.Parameter(torch.randn(num_global_vectors, hidden_dim) * 0.01)
        self.selector = nn.Linear(hidden_dim, num_global_vectors)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply global context perturbation.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            Perturbed hidden states
        """
        # Compute global document representation
        global_repr = hidden_states.mean(dim=1)  # (batch, hidden_dim)

        # Select perturbation based on document
        weights = F.softmax(self.selector(global_repr), dim=-1)  # (batch, num_global_vectors)

        # Compute weighted perturbation
        perturbation = torch.einsum('bn,nd->bd', weights, self.global_vectors)  # (batch, hidden_dim)

        # Normalize and scale
        perturbation = F.normalize(perturbation, dim=-1) * self.epsilon

        # Add to all positions
        return hidden_states + perturbation.unsqueeze(1)


# =============================================================================
# Hierarchical Perturbation Manager
# =============================================================================

class HierarchicalPerturbationManager:
    """
    Manager for coordinating perturbations across all three levels.

    Ensures consistency and controls the overall perturbation budget.
    """

    def __init__(
        self,
        config: PerturbationConfig,
        vocab_size: int = 50000,
        hidden_dim: int = 768,
    ):
        self.config = config
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Initialize perturbation modules
        self.local = LocalPerturbation(config, vocab_size)
        self.chunk = ChunkPerturbation(config, vocab_size)
        self.global_ = GlobalPerturbation(config, vocab_size)

        # Learnable perturbation modules
        self.position_pert = PositionEncodingPerturbation(
            epsilon=config.local_epsilon,
            hidden_dim=hidden_dim,
        )
        self.chunk_pert = ChunkEmbeddingPerturbation(
            epsilon=config.chunk_epsilon,
            hidden_dim=hidden_dim,
        )
        self.global_pert = GlobalContextPerturbation(
            epsilon=config.global_epsilon,
            hidden_dim=hidden_dim,
        )

    def apply_all_perturbations(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        chunk_boundaries: Optional[List[List[int]]] = None,
        levels: List[str] = ['local', 'chunk', 'global'],
    ) -> Dict[str, torch.Tensor]:
        """
        Apply perturbations at specified levels.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            hidden_states: Hidden representations
            chunk_boundaries: Paragraph boundaries
            levels: Which levels to apply ('local', 'chunk', 'global')

        Returns:
            Dictionary with perturbed tensors
        """
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'hidden_states': hidden_states,
        }

        if 'local' in levels:
            result = self.local.apply(
                result['input_ids'],
                result['attention_mask'],
                hidden_states=result['hidden_states'],
            )

        if 'chunk' in levels:
            result = self.chunk.apply(
                result['input_ids'],
                result['attention_mask'],
                chunk_boundaries=chunk_boundaries,
                hidden_states=result['hidden_states'],
            )

        if 'global' in levels:
            result = self.global_.apply(
                result['input_ids'],
                result['attention_mask'],
                hidden_states=result['hidden_states'],
            )

        return result

    def apply_learnable_perturbations(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        chunk_boundaries: Optional[List[int]] = None,
        levels: List[str] = ['local', 'chunk', 'global'],
    ) -> torch.Tensor:
        """
        Apply learnable perturbations to hidden states.

        Args:
            hidden_states: Model hidden states
            position_embeddings: Position embeddings (if separate)
            chunk_boundaries: Chunk boundaries
            levels: Which levels to apply

        Returns:
            Perturbed hidden states
        """
        perturbed = hidden_states

        if 'local' in levels and position_embeddings is not None:
            perturbed = self.position_pert(perturbed)

        if 'chunk' in levels:
            perturbed = self.chunk_pert(perturbed, chunk_boundaries)

        if 'global' in levels:
            perturbed = self.global_pert(perturbed)

        return perturbed

    def compute_total_distance(
        self,
        orig: Dict[str, torch.Tensor],
        pert: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Compute distances at each level.

        Returns:
            Dictionary with distances per level
        """
        distances = {}

        if 'input_ids' in orig and 'input_ids' in pert:
            distances['local'] = self.local.compute_distance(orig['input_ids'], pert['input_ids'])
            distances['chunk'] = self.chunk.compute_distance(orig['input_ids'], pert['input_ids'])

        if 'hidden_states' in orig and orig['hidden_states'] is not None:
            distances['global'] = self.global_.compute_distance(
                orig.get('hidden_states', orig['input_ids']),
                pert.get('hidden_states', pert['input_ids'])
            )

        return distances

    def check_constraints(self, distances: Dict[str, float]) -> bool:
        """
        Check if perturbations are within uncertainty set constraints.

        Args:
            distances: Distances at each level

        Returns:
            True if all constraints satisfied
        """
        if distances.get('local', 0) > self.config.local_epsilon:
            return False
        if distances.get('chunk', 0) > self.config.chunk_epsilon:
            return False
        if distances.get('global', 0) > self.config.global_epsilon:
            return False
        return True

    def get_learnable_parameters(self) -> List[nn.Parameter]:
        """Get all learnable perturbation parameters."""
        params = []
        params.extend(self.position_pert.parameters())
        params.extend(self.chunk_pert.parameters())
        params.extend(self.global_pert.parameters())
        return params


# =============================================================================
# Utility Functions
# =============================================================================

def create_worst_case_batch(
    batch: Dict[str, torch.Tensor],
    manager: HierarchicalPerturbationManager,
    num_samples: int = 5,
) -> List[Dict[str, torch.Tensor]]:
    """
    Generate multiple perturbed versions of a batch for worst-case selection.

    Args:
        batch: Original batch
        manager: Perturbation manager
        num_samples: Number of perturbations to generate

    Returns:
        List of perturbed batches
    """
    perturbed_batches = []

    for _ in range(num_samples):
        perturbed = manager.apply_all_perturbations(
            batch['input_ids'],
            batch.get('attention_mask'),
            batch.get('hidden_states'),
        )
        perturbed_batches.append(perturbed)

    return perturbed_batches


def select_worst_case(
    perturbed_batches: List[Dict[str, torch.Tensor]],
    losses: List[torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Select the worst-case perturbation based on loss.

    Args:
        perturbed_batches: List of perturbed batches
        losses: Corresponding losses

    Returns:
        Worst-case batch and loss
    """
    worst_idx = max(range(len(losses)), key=lambda i: losses[i].item())
    return perturbed_batches[worst_idx], losses[worst_idx]


if __name__ == "__main__":
    # Test the perturbation system
    print("Testing Hierarchical Perturbation System...")

    # Create config
    config = PerturbationConfig()

    # Create manager
    manager = HierarchicalPerturbationManager(config, vocab_size=1000, hidden_dim=64)

    # Create test data
    batch_size, seq_len = 2, 256
    input_ids = torch.randint(4, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    hidden_states = torch.randn(batch_size, seq_len, 64)

    # Apply perturbations
    result = manager.apply_all_perturbations(
        input_ids, attention_mask, hidden_states
    )

    print(f"Original shape: {input_ids.shape}")
    print(f"Perturbed shape: {result['input_ids'].shape}")

    # Compute distances
    orig_data = {'input_ids': input_ids, 'hidden_states': hidden_states}
    distances = manager.compute_total_distance(orig_data, result)
    print(f"Distances: {distances}")

    # Check constraints
    within_constraints = manager.check_constraints(distances)
    print(f"Within constraints: {within_constraints}")

    # Test learnable perturbations
    perturbed_hidden = manager.apply_learnable_perturbations(hidden_states)
    print(f"Learnable perturbed shape: {perturbed_hidden.shape}")

    print("Hierarchical Perturbation System test complete!")
