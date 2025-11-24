"""
Transformer Language Model with Position Encoding Variants

This module implements:
1. Base transformer language model architecture
2. Multiple position encoding methods:
   - Learned absolute positions
   - Rotary Position Embedding (RoPE)
   - Attention with Linear Biases (ALiBi)
   - No Position Encoding (NoPE) baseline
3. Efficient attention variants for long sequences
4. Integration with hierarchical perturbations

The architecture is designed for length generalization experiments.
"""

import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Configuration for Transformer Language Model."""
    vocab_size: int = 50000
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_dim: int = 3072
    max_seq_len: int = 8192
    dropout: float = 0.1
    attention_dropout: float = 0.1
    position_encoding: str = "rope"  # 'learned', 'rope', 'alibi', 'none'
    tie_word_embeddings: bool = True
    layer_norm_eps: float = 1e-6
    rope_base: float = 10000.0
    rope_scaling: Optional[Dict] = None  # For length scaling
    use_flash_attention: bool = False


# =============================================================================
# Position Encoding Implementations
# =============================================================================

class LearnedPositionEncoding(nn.Module):
    """Standard learned absolute position embeddings."""

    def __init__(self, max_seq_len: int, hidden_dim: int):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_dim)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate position embeddings.

        Args:
            seq_len: Sequence length
            device: Device to create tensor on

        Returns:
            Position embeddings (1, seq_len, hidden_dim)
        """
        if seq_len > self.max_seq_len:
            # Extrapolate by repeating positions (not ideal, for baseline comparison)
            positions = torch.arange(seq_len, device=device) % self.max_seq_len
        else:
            positions = torch.arange(seq_len, device=device)

        return self.position_embeddings(positions).unsqueeze(0)


class RotaryPositionEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    RoPE encodes position information by rotating the query and key vectors.
    This enables length extrapolation through the rotational invariance property.

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        scaling_type: str = None,  # 'linear', 'dynamic', 'yarn'
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor
        self.scaling_type = scaling_type

        # Precompute frequencies
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        """Precompute cosine and sine cache for rotary embeddings."""
        # Compute inverse frequencies
        dim = self.dim
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))

        # Apply scaling if specified
        if self.scaling_type == 'linear':
            inv_freq = inv_freq / self.scaling_factor
        elif self.scaling_type == 'dynamic':
            # NTK-aware scaling
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_seq_len) - (self.scaling_factor - 1)
            ) ** (dim / (dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        elif self.scaling_type == 'yarn':
            # YaRN scaling (Yet another RoPE extensioN)
            low_freq_factor = 1.0
            high_freq_factor = 4.0
            original_max_len = self.max_seq_len

            low_freq_wavelen = original_max_len / low_freq_factor
            high_freq_wavelen = original_max_len / high_freq_factor

            for i in range(0, dim, 2):
                freq = inv_freq[i // 2]
                wavelen = 2 * math.pi / freq

                if wavelen < high_freq_wavelen:
                    # High frequency: no scaling
                    pass
                elif wavelen > low_freq_wavelen:
                    # Low frequency: scale by factor
                    inv_freq[i // 2] = freq / self.scaling_factor
                else:
                    # Interpolate
                    smooth = (original_max_len / wavelen - low_freq_factor) / (
                        high_freq_factor - low_freq_factor
                    )
                    inv_freq[i // 2] = (1 - smooth) * freq / self.scaling_factor + smooth * freq

        self.register_buffer('inv_freq', inv_freq)

        # Create position indices
        positions = torch.arange(seq_len).float()
        freqs = torch.outer(positions, self.inv_freq)

        # Cache cos and sin
        self.register_buffer('cos_cache', freqs.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cache', freqs.sin().unsqueeze(0).unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor.

        Args:
            x: Input tensor (batch, num_heads, seq_len, head_dim)
            position_ids: Optional position indices

        Returns:
            Rotated tensor
        """
        seq_len = x.size(2)

        # Extend cache if needed
        if seq_len > self.cos_cache.size(2):
            self._set_cos_sin_cache(seq_len)
            self.cos_cache = self.cos_cache.to(x.device)
            self.sin_cache = self.sin_cache.to(x.device)

        if position_ids is not None:
            cos = self.cos_cache[0, 0, position_ids]
            sin = self.sin_cache[0, 0, position_ids]
        else:
            cos = self.cos_cache[:, :, :seq_len]
            sin = self.sin_cache[:, :, :seq_len]

        return self._apply_rotary(x, cos, sin)

    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary transformation."""
        # Split into two halves
        x1, x2 = x[..., :x.size(-1) // 2], x[..., x.size(-1) // 2:]

        # Rotate
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin,
        ], dim=-1)

        return rotated


class ALiBiPositionEncoding(nn.Module):
    """
    Attention with Linear Biases (ALiBi).

    ALiBi adds a linear bias to attention scores based on position distance.
    This provides natural length extrapolation without learned parameters.

    Reference: Press et al., "Train Short, Test Long"
    """

    def __init__(self, num_heads: int, max_seq_len: int = 8192):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Compute slopes for each head
        # Slopes decrease geometrically: 2^(-8/n), 2^(-16/n), ...
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)

        # Precompute bias matrix
        self._set_bias_cache(max_seq_len)

    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        """Compute ALiBi slopes for each attention head."""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            # Handle non-power-of-2 heads
            closest_power = 2 ** math.floor(math.log2(num_heads))
            slopes = (
                get_slopes_power_of_2(closest_power) +
                get_slopes_power_of_2(2 * closest_power)[0::2][:num_heads - closest_power]
            )

        return torch.tensor(slopes).float()

    def _set_bias_cache(self, seq_len: int):
        """Precompute bias matrix for given sequence length."""
        # Create distance matrix: positions[i] - positions[j]
        positions = torch.arange(seq_len)
        distances = positions.unsqueeze(0) - positions.unsqueeze(1)

        # Only apply bias for causal attention (j <= i)
        # Make positive distances (future positions) very negative
        distances = distances.float()
        distances = torch.where(distances > 0, torch.tensor(-1e9), distances)

        # Compute bias: slopes * abs(distances)
        # Shape: (num_heads, seq_len, seq_len)
        bias = self.slopes.unsqueeze(1).unsqueeze(2) * distances.abs().unsqueeze(0)

        self.register_buffer('bias_cache', bias)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Get ALiBi bias matrix.

        Args:
            seq_len: Sequence length
            device: Device

        Returns:
            Bias matrix (1, num_heads, seq_len, seq_len)
        """
        if seq_len > self.bias_cache.size(1):
            self._set_bias_cache(seq_len)
            self.bias_cache = self.bias_cache.to(device)

        return self.bias_cache[:, :seq_len, :seq_len].unsqueeze(0)


# =============================================================================
# Attention Implementations
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with support for various position encodings.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        position_encoding: str = "rope",
        max_seq_len: int = 8192,
        rope_base: float = 10000.0,
        rope_scaling: Optional[Dict] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.position_encoding = position_encoding

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        # Position encoding
        if position_encoding == "rope":
            scaling_factor = rope_scaling.get('factor', 1.0) if rope_scaling else 1.0
            scaling_type = rope_scaling.get('type', None) if rope_scaling else None
            self.rope = RotaryPositionEncoding(
                self.head_dim,
                max_seq_len,
                rope_base,
                scaling_factor,
                scaling_type,
            )
        elif position_encoding == "alibi":
            self.alibi = ALiBiPositionEncoding(num_heads, max_seq_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len) or (batch, 1, seq_len, seq_len)
            position_ids: Optional position indices
            past_key_value: Cached key/value for generation
            output_attentions: Whether to return attention weights

        Returns:
            Output tensor and optional attention weights
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape: (batch, seq_len, hidden_dim) -> (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply position encoding
        if self.position_encoding == "rope":
            q = self.rope(q, position_ids)
            k = self.rope(k, position_ids)

        # Handle past key/value for generation
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply ALiBi bias
        if self.position_encoding == "alibi":
            alibi_bias = self.alibi(k.size(2), hidden_states.device)
            attn_scores = attn_scores + alibi_bias

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, k.size(2), device=hidden_states.device) * float('-inf'),
            diagonal=k.size(2) - seq_len + 1
        )
        attn_scores = attn_scores + causal_mask

        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute output
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        attn_output = self.out_proj(attn_output)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# =============================================================================
# Transformer Blocks
# =============================================================================

class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.attention = MultiHeadAttention(
            config.hidden_dim,
            config.num_heads,
            config.attention_dropout,
            config.position_encoding,
            config.max_seq_len,
            config.rope_base,
            config.rope_scaling,
        )

        self.attention_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.intermediate_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Self-attention
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask, position_ids, past_key_value)[0]
        hidden_states = residual + hidden_states

        # Feed-forward
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Full Language Model
# =============================================================================

class TransformerLanguageModel(nn.Module):
    """
    Transformer-based language model for length generalization experiments.

    Supports multiple position encoding methods for comparison.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Learned position embeddings (only if using 'learned')
        if config.position_encoding == "learned":
            self.position_embeddings = LearnedPositionEncoding(config.max_seq_len, config.hidden_dim)
        else:
            self.position_embeddings = None

        self.embedding_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Output normalization
        self.output_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)

        # Language model head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with small values for stability."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embedding layer."""
        return self.token_embeddings

    def set_input_embeddings(self, embeddings: nn.Embedding):
        """Set input embedding layer."""
        self.token_embeddings = embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            position_ids: Position IDs (batch, seq_len)
            labels: Target labels for LM loss (batch, seq_len)
            output_hidden_states: Whether to return all hidden states

        Returns:
            Dictionary with logits, loss, and optionally hidden states
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        hidden_states = self.token_embeddings(input_ids)

        # Add position embeddings if using learned positions
        if self.position_embeddings is not None:
            pos_emb = self.position_embeddings(seq_len, input_ids.device)
            hidden_states = hidden_states + pos_emb

        hidden_states = self.embedding_dropout(hidden_states)

        # Store hidden states if requested
        all_hidden_states = [hidden_states] if output_hidden_states else None

        # Pass through transformer blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        # Output normalization
        hidden_states = self.output_norm(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # Ignore padding
            )

        outputs = {
            'logits': logits,
            'loss': loss,
            'last_hidden_state': hidden_states,
        }

        if output_hidden_states:
            outputs['hidden_states'] = all_hidden_states

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Prompt tokens (batch, prompt_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold

        Returns:
            Generated token IDs (batch, prompt_len + max_new_tokens)
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Get logits for last position
            outputs = self.forward(generated)
            next_token_logits = outputs['logits'][:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

        return generated


# =============================================================================
# Model Factory
# =============================================================================

def create_model(
    model_size: str = "small",
    position_encoding: str = "rope",
    vocab_size: int = 50000,
    max_seq_len: int = 8192,
    rope_scaling: Optional[Dict] = None,
) -> TransformerLanguageModel:
    """
    Create a transformer language model with specified configuration.

    Args:
        model_size: 'tiny', 'small', 'medium', 'large'
        position_encoding: 'learned', 'rope', 'alibi', 'none'
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        rope_scaling: RoPE scaling configuration

    Returns:
        TransformerLanguageModel instance
    """
    configs = {
        'tiny': TransformerConfig(
            vocab_size=vocab_size,
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            intermediate_dim=1024,
            max_seq_len=max_seq_len,
            position_encoding=position_encoding,
            rope_scaling=rope_scaling,
        ),
        'small': TransformerConfig(
            vocab_size=vocab_size,
            hidden_dim=512,
            num_layers=6,
            num_heads=8,
            intermediate_dim=2048,
            max_seq_len=max_seq_len,
            position_encoding=position_encoding,
            rope_scaling=rope_scaling,
        ),
        'medium': TransformerConfig(
            vocab_size=vocab_size,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            intermediate_dim=3072,
            max_seq_len=max_seq_len,
            position_encoding=position_encoding,
            rope_scaling=rope_scaling,
        ),
        'large': TransformerConfig(
            vocab_size=vocab_size,
            hidden_dim=1024,
            num_layers=24,
            num_heads=16,
            intermediate_dim=4096,
            max_seq_len=max_seq_len,
            position_encoding=position_encoding,
            rope_scaling=rope_scaling,
        ),
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    return TransformerLanguageModel(configs[model_size])


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Pre-trained Model Loading (HuggingFace Integration)
# =============================================================================

class HuggingFaceModelWrapper(nn.Module):
    """
    Wrapper for HuggingFace transformer models.

    Provides a unified interface for experiments with pre-trained models.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        position_encoding: str = "rope",  # Override position encoding
        max_seq_len: int = 8192,
        rope_scaling: Optional[Dict] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.position_encoding = position_encoding

        try:
            from transformers import AutoModelForCausalLM, AutoConfig

            # Load model
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

            # Modify for longer sequences if needed
            if max_seq_len > config.max_position_embeddings:
                self._extend_position_embeddings(max_seq_len, rope_scaling)

            self.hidden_dim = config.hidden_size
            self.vocab_size = config.vocab_size
            self.available = True

        except ImportError:
            print("transformers library not available. Using custom model instead.")
            self.model = create_model('small', position_encoding, max_seq_len=max_seq_len)
            self.hidden_dim = 512
            self.vocab_size = 50000
            self.available = False

    def _extend_position_embeddings(self, max_seq_len: int, rope_scaling: Optional[Dict]):
        """Extend position embeddings for longer sequences."""
        # This is model-specific and may need adjustment
        if hasattr(self.model.config, 'rope_scaling'):
            self.model.config.rope_scaling = rope_scaling

        # For GPT-2 style models, extend learned positions
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wpe'):
            old_wpe = self.model.transformer.wpe
            old_max_len = old_wpe.weight.size(0)

            if max_seq_len > old_max_len:
                new_wpe = nn.Embedding(max_seq_len, old_wpe.weight.size(1))
                # Copy old weights
                new_wpe.weight.data[:old_max_len] = old_wpe.weight.data
                # Initialize new positions by interpolation
                for i in range(old_max_len, max_seq_len):
                    idx = (i % old_max_len)
                    new_wpe.weight.data[i] = old_wpe.weight.data[idx]
                self.model.transformer.wpe = new_wpe

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with unified interface."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
        )

        result = {
            'logits': outputs.logits,
            'loss': outputs.loss,
            'last_hidden_state': outputs.hidden_states[-1] if output_hidden_states else None,
        }

        if output_hidden_states:
            result['hidden_states'] = outputs.hidden_states

        return result


if __name__ == "__main__":
    # Test the model
    print("Testing Transformer Language Model...")

    # Create model
    model = create_model('tiny', position_encoding='rope', vocab_size=1000, max_seq_len=2048)
    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size, seq_len = 2, 512
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = input_ids.clone()

    outputs = model(input_ids, labels=labels, output_hidden_states=True)

    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Hidden states: {len(outputs['hidden_states'])} layers")

    # Test generation
    prompt = torch.randint(0, 1000, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")

    # Test different position encodings
    for pos_enc in ['learned', 'rope', 'alibi', 'none']:
        model = create_model('tiny', position_encoding=pos_enc, vocab_size=1000)
        outputs = model(input_ids)
        print(f"Position encoding '{pos_enc}': logits shape {outputs['logits'].shape}")

    # Test RoPE scaling
    rope_scaling = {'type': 'linear', 'factor': 2.0}
    model = create_model('tiny', position_encoding='rope', vocab_size=1000, rope_scaling=rope_scaling)
    long_input = torch.randint(0, 1000, (1, 4096))
    outputs = model(long_input)
    print(f"RoPE with scaling: logits shape {outputs['logits'].shape}")

    print("Transformer Language Model test complete!")
