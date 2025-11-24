"""
NLP Dataset Infrastructure for Hierarchical DRO Length Generalization Experiments

This module provides:
1. Text dataset loaders with variable sequence lengths
2. Tokenization and preprocessing utilities
3. Length-based data grouping and sampling
4. Synthetic long-document generation for controlled experiments

Datasets supported:
- WikiText-2/103: Language modeling benchmark
- PG19: Long-form text (books)
- ArXiv: Scientific papers with structure
- Synthetic: Controlled length generation
"""

import os
import math
import random
from typing import Optional, Dict, List, Tuple, Union, Iterator
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np

try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers library not installed. Using simple tokenizer.")


@dataclass
class TextSample:
    """Container for a text sample with metadata."""
    text: str
    tokens: Optional[torch.Tensor] = None
    length: int = 0
    document_id: str = ""
    chunk_boundaries: Optional[List[int]] = None  # Paragraph boundaries

    def __post_init__(self):
        if self.tokens is not None:
            self.length = len(self.tokens)


class SimpleTokenizer:
    """Simple word-level tokenizer for experiments without transformers library."""

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.bos_token_id = 3

        # Initialize special tokens
        self.word_to_id['<pad>'] = 0
        self.word_to_id['<eos>'] = 1
        self.word_to_id['<unk>'] = 2
        self.word_to_id['<bos>'] = 3
        self.id_to_word[0] = '<pad>'
        self.id_to_word[1] = '<eos>'
        self.id_to_word[2] = '<unk>'
        self.id_to_word[3] = '<bos>'
        self.next_id = 4

    def fit(self, texts: List[str], min_freq: int = 2):
        """Build vocabulary from texts."""
        word_counts: Dict[str, int] = defaultdict(int)
        for text in texts:
            for word in text.lower().split():
                word_counts[word] += 1

        # Sort by frequency and take top vocab_size - 4 words
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        for word, count in sorted_words[:self.vocab_size - 4]:
            if count >= min_freq:
                self.word_to_id[word] = self.next_id
                self.id_to_word[self.next_id] = word
                self.next_id += 1

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        for word in text.lower().split():
            tokens.append(self.word_to_id.get(word, self.unk_token_id))
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        words = []
        for tid in token_ids:
            if tid in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            words.append(self.id_to_word.get(tid, '<unk>'))
        return ' '.join(words)

    def __call__(self, text: str, return_tensors: str = None,
                 truncation: bool = False, max_length: int = None,
                 padding: str = None) -> Dict:
        """Tokenize with HuggingFace-like interface."""
        tokens = self.encode(text)
        if truncation and max_length:
            tokens = tokens[:max_length]
        if padding == 'max_length' and max_length:
            tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))

        result = {'input_ids': tokens, 'attention_mask': [1 if t != self.pad_token_id else 0 for t in tokens]}
        if return_tensors == 'pt':
            result = {k: torch.tensor([v]) for k, v in result.items()}
        return result


class TokenizerWrapper:
    """Wrapper to provide unified interface for different tokenizers."""

    def __init__(self, tokenizer_name: str = "gpt2", vocab_size: int = 50000):
        self.tokenizer_name = tokenizer_name

        if HAS_TRANSFORMERS and tokenizer_name != "simple":
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.pad_token_id = self.tokenizer.pad_token_id
                self.eos_token_id = self.tokenizer.eos_token_id
                self.vocab_size = len(self.tokenizer)
            except Exception as e:
                print(f"Failed to load {tokenizer_name}: {e}. Using simple tokenizer.")
                self.tokenizer = SimpleTokenizer(vocab_size)
                self.pad_token_id = 0
                self.eos_token_id = 1
                self.vocab_size = vocab_size
        else:
            self.tokenizer = SimpleTokenizer(vocab_size)
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.vocab_size = vocab_size

    def encode(self, text: str, max_length: Optional[int] = None,
               truncation: bool = True, return_tensors: str = None) -> Union[List[int], torch.Tensor]:
        """Encode text to tokens."""
        if isinstance(self.tokenizer, SimpleTokenizer):
            result = self.tokenizer(text, truncation=truncation, max_length=max_length,
                                   return_tensors=return_tensors)
            return result['input_ids']
        else:
            result = self.tokenizer(text, truncation=truncation, max_length=max_length,
                                   return_tensors=return_tensors)
            return result['input_ids']

    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Decode tokens to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return self.tokenizer.decode(token_ids)

    def fit(self, texts: List[str]):
        """Fit tokenizer on texts (only for SimpleTokenizer)."""
        if isinstance(self.tokenizer, SimpleTokenizer):
            self.tokenizer.fit(texts)
            self.vocab_size = self.tokenizer.next_id


class LengthGroupedDataset(Dataset):
    """
    Dataset that groups sequences by length for efficient batching.

    This enables:
    1. Training on specific length ranges
    2. Evaluating length generalization
    3. Curriculum learning from short to long
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: TokenizerWrapper,
        length_buckets: List[Tuple[int, int]] = None,  # [(min, max), ...]
        max_length: int = 8192,
        stride: int = None,  # For sliding window
        add_chunk_boundaries: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length // 2
        self.add_chunk_boundaries = add_chunk_boundaries

        # Default length buckets (in tokens)
        if length_buckets is None:
            length_buckets = [
                (64, 512),      # Short
                (512, 2048),    # Medium
                (2048, 8192),   # Long
                (8192, 32768),  # Very Long
            ]
        self.length_buckets = length_buckets

        # Process texts into samples
        self.samples: List[TextSample] = []
        self.bucket_indices: Dict[int, List[int]] = defaultdict(list)

        self._process_texts(texts)

    def _find_chunk_boundaries(self, text: str) -> List[int]:
        """Find paragraph/section boundaries in text."""
        boundaries = [0]
        paragraphs = text.split('\n\n')
        pos = 0
        for para in paragraphs:
            pos += len(para) + 2  # +2 for \n\n
            boundaries.append(pos)
        return boundaries

    def _process_texts(self, texts: List[str]):
        """Process texts into tokenized samples with length grouping."""
        for doc_id, text in enumerate(texts):
            # Find chunk boundaries before tokenization
            chunk_boundaries = self._find_chunk_boundaries(text) if self.add_chunk_boundaries else None

            # Tokenize
            tokens = self.tokenizer.encode(text, max_length=None, truncation=False)
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.squeeze(0)
            else:
                tokens = torch.tensor(tokens)

            # Create samples with sliding window for long documents
            if len(tokens) <= self.max_length:
                sample = TextSample(
                    text=text,
                    tokens=tokens,
                    document_id=f"doc_{doc_id}",
                    chunk_boundaries=chunk_boundaries
                )
                self.samples.append(sample)

                # Add to appropriate bucket
                bucket_idx = self._get_bucket_index(len(tokens))
                if bucket_idx >= 0:
                    self.bucket_indices[bucket_idx].append(len(self.samples) - 1)
            else:
                # Sliding window for long documents
                for start in range(0, len(tokens) - self.max_length + 1, self.stride):
                    end = start + self.max_length
                    sample = TextSample(
                        text=text[start:end] if start < len(text) else "",
                        tokens=tokens[start:end],
                        document_id=f"doc_{doc_id}_chunk_{start}",
                        chunk_boundaries=chunk_boundaries
                    )
                    self.samples.append(sample)

                    bucket_idx = self._get_bucket_index(self.max_length)
                    if bucket_idx >= 0:
                        self.bucket_indices[bucket_idx].append(len(self.samples) - 1)

    def _get_bucket_index(self, length: int) -> int:
        """Get the bucket index for a given length."""
        for i, (min_len, max_len) in enumerate(self.length_buckets):
            if min_len <= length < max_len:
                return i
        return -1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            'input_ids': sample.tokens,
            'length': sample.length,
            'document_id': sample.document_id,
        }

    def get_bucket_samples(self, bucket_idx: int) -> List[int]:
        """Get sample indices for a specific length bucket."""
        return self.bucket_indices[bucket_idx]

    def get_length_distribution(self) -> Dict[str, int]:
        """Get distribution of samples across length buckets."""
        return {
            f"{self.length_buckets[i][0]}-{self.length_buckets[i][1]}": len(indices)
            for i, indices in self.bucket_indices.items()
        }


class LengthBucketSampler(Sampler):
    """
    Sampler that samples from specific length buckets.

    Supports:
    - Single bucket sampling (for evaluation at specific length)
    - Weighted sampling across buckets (for training)
    - Curriculum sampling (short to long)
    """

    def __init__(
        self,
        dataset: LengthGroupedDataset,
        batch_size: int,
        bucket_weights: Optional[Dict[int, float]] = None,
        curriculum: bool = False,
        curriculum_progress: float = 0.0,  # 0.0 = start (short), 1.0 = end (all)
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.curriculum = curriculum
        self.curriculum_progress = curriculum_progress

        # Default: uniform weighting
        if bucket_weights is None:
            bucket_weights = {i: 1.0 for i in range(len(dataset.length_buckets))}
        self.bucket_weights = bucket_weights

        # Normalize weights
        total_weight = sum(bucket_weights.values())
        self.bucket_probs = {k: v / total_weight for k, v in bucket_weights.items()}

    def __iter__(self) -> Iterator[List[int]]:
        # Collect all valid indices based on curriculum progress
        if self.curriculum:
            # Gradually include longer buckets
            max_bucket = int(self.curriculum_progress * len(self.dataset.length_buckets))
            max_bucket = max(1, min(max_bucket, len(self.dataset.length_buckets)))
            valid_buckets = list(range(max_bucket))
        else:
            valid_buckets = list(self.bucket_weights.keys())

        # Collect indices from valid buckets
        all_indices = []
        for bucket_idx in valid_buckets:
            indices = self.dataset.get_bucket_samples(bucket_idx)
            weight = self.bucket_probs.get(bucket_idx, 0)
            # Repeat indices based on weight
            all_indices.extend(indices * max(1, int(weight * 10)))

        # Shuffle
        random.shuffle(all_indices)

        # Yield batches
        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        total = sum(len(self.dataset.get_bucket_samples(i)) for i in self.bucket_weights.keys())
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size

    def update_curriculum(self, progress: float):
        """Update curriculum progress (0.0 to 1.0)."""
        self.curriculum_progress = min(1.0, max(0.0, progress))


class SyntheticLongDocumentDataset(Dataset):
    """
    Synthetic dataset for controlled length generalization experiments.

    Features:
    - Controllable document length
    - Controllable structure (paragraphs, sections)
    - Controllable complexity patterns
    - Ground truth for evaluation
    """

    def __init__(
        self,
        tokenizer: TokenizerWrapper,
        num_samples: int = 10000,
        length_range: Tuple[int, int] = (512, 8192),
        vocab_subset_size: int = 5000,
        num_patterns: int = 10,  # Number of distinct patterns to learn
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.length_range = length_range
        self.vocab_subset_size = min(vocab_subset_size, tokenizer.vocab_size)
        self.num_patterns = num_patterns

        np.random.seed(seed)
        random.seed(seed)

        # Create pattern templates
        self.patterns = self._create_patterns()

        # Generate samples
        self.samples = self._generate_samples()

    def _create_patterns(self) -> List[Dict]:
        """Create learnable patterns that span different ranges."""
        patterns = []
        for i in range(self.num_patterns):
            # Each pattern has:
            # - A trigger sequence
            # - A response that should follow
            # - A span (local, chunk, global)
            trigger_len = random.randint(3, 10)
            response_len = random.randint(3, 10)

            # Use subset of vocabulary for cleaner patterns
            trigger = [random.randint(4, self.vocab_subset_size) for _ in range(trigger_len)]
            response = [random.randint(4, self.vocab_subset_size) for _ in range(response_len)]

            span = random.choice(['local', 'chunk', 'global'])
            distance = {
                'local': random.randint(1, 50),
                'chunk': random.randint(100, 500),
                'global': random.randint(1000, 4000),
            }[span]

            patterns.append({
                'trigger': trigger,
                'response': response,
                'span': span,
                'distance': distance,
                'pattern_id': i,
            })
        return patterns

    def _generate_samples(self) -> List[Dict]:
        """Generate synthetic samples with embedded patterns."""
        samples = []

        for _ in range(self.num_samples):
            length = random.randint(*self.length_range)

            # Generate base sequence (random tokens)
            sequence = [random.randint(4, self.vocab_subset_size) for _ in range(length)]

            # Embed patterns
            embedded_patterns = []
            for pattern in self.patterns:
                if random.random() < 0.3:  # 30% chance to include each pattern
                    # Find position for trigger
                    trigger_pos = random.randint(0, max(0, length - pattern['distance'] - len(pattern['response'])))
                    response_pos = trigger_pos + pattern['distance']

                    if response_pos + len(pattern['response']) < length:
                        # Insert trigger
                        for j, tok in enumerate(pattern['trigger']):
                            if trigger_pos + j < length:
                                sequence[trigger_pos + j] = tok

                        # Insert response
                        for j, tok in enumerate(pattern['response']):
                            if response_pos + j < length:
                                sequence[response_pos + j] = tok

                        embedded_patterns.append({
                            'pattern_id': pattern['pattern_id'],
                            'trigger_pos': trigger_pos,
                            'response_pos': response_pos,
                            'span': pattern['span'],
                        })

            samples.append({
                'tokens': torch.tensor(sequence),
                'length': length,
                'patterns': embedded_patterns,
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        return {
            'input_ids': sample['tokens'],
            'length': sample['length'],
            'patterns': sample['patterns'],
        }

    def get_pattern_accuracy(self, predictions: torch.Tensor, sample_idx: int) -> Dict[str, float]:
        """
        Evaluate pattern prediction accuracy for a sample.

        Args:
            predictions: Model predictions (seq_len, vocab_size) or (seq_len,)
            sample_idx: Sample index

        Returns:
            Accuracy metrics per pattern type
        """
        sample = self.samples[sample_idx]

        results = {'local': [], 'chunk': [], 'global': []}

        for pattern_info in sample['patterns']:
            pattern = self.patterns[pattern_info['pattern_id']]
            response_pos = pattern_info['response_pos']

            # Check if response was predicted correctly
            correct = 0
            total = len(pattern['response'])

            for j, expected_tok in enumerate(pattern['response']):
                pos = response_pos + j
                if pos < len(predictions):
                    if predictions.dim() == 2:
                        pred_tok = predictions[pos].argmax().item()
                    else:
                        pred_tok = predictions[pos].item()
                    if pred_tok == expected_tok:
                        correct += 1

            accuracy = correct / total if total > 0 else 0
            results[pattern_info['span']].append(accuracy)

        return {
            span: np.mean(accs) if accs else 0.0
            for span, accs in results.items()
        }


def collate_variable_length(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate function for variable length sequences.

    Pads sequences to the maximum length in the batch.
    """
    max_len = max(item['input_ids'].size(0) for item in batch)

    input_ids = []
    attention_mask = []
    lengths = []

    for item in batch:
        seq = item['input_ids']
        seq_len = seq.size(0)

        # Pad sequence
        padded = torch.full((max_len,), pad_token_id, dtype=seq.dtype)
        padded[:seq_len] = seq
        input_ids.append(padded)

        # Create attention mask
        mask = torch.zeros(max_len, dtype=torch.long)
        mask[:seq_len] = 1
        attention_mask.append(mask)

        lengths.append(seq_len)

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'lengths': torch.tensor(lengths),
    }


# =============================================================================
# Dataset Loaders for Real Datasets
# =============================================================================

def load_wikitext(
    tokenizer: TokenizerWrapper,
    split: str = "train",
    version: str = "103",
    cache_dir: str = "./data",
) -> LengthGroupedDataset:
    """
    Load WikiText-2 or WikiText-103 dataset.

    WikiText-103 contains 103M tokens from Wikipedia articles.
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset(f"wikitext", f"wikitext-{version}-raw-v1",
                              split=split, cache_dir=cache_dir)
        texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100]
    except ImportError:
        print("datasets library not installed. Generating synthetic data.")
        texts = _generate_synthetic_wiki_text(1000)
    except Exception as e:
        print(f"Failed to load WikiText: {e}. Generating synthetic data.")
        texts = _generate_synthetic_wiki_text(1000)

    return LengthGroupedDataset(texts, tokenizer)


def load_pg19(
    tokenizer: TokenizerWrapper,
    split: str = "train",
    cache_dir: str = "./data",
    max_books: int = 100,
) -> LengthGroupedDataset:
    """
    Load PG19 dataset (Project Gutenberg books).

    Contains full books, excellent for long-range dependency testing.
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("pg19", split=split, cache_dir=cache_dir)
        texts = [item['text'] for item in list(dataset)[:max_books]]
    except ImportError:
        print("datasets library not installed. Generating synthetic data.")
        texts = _generate_synthetic_book_text(50)
    except Exception as e:
        print(f"Failed to load PG19: {e}. Generating synthetic data.")
        texts = _generate_synthetic_book_text(50)

    return LengthGroupedDataset(texts, tokenizer, max_length=16384)


def _generate_synthetic_wiki_text(num_articles: int = 1000) -> List[str]:
    """Generate synthetic Wikipedia-like text for testing."""
    texts = []
    vocab = ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'have', 'had',
             'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'about', 'into',
             'through', 'during', 'before', 'after', 'above', 'below', 'between',
             'this', 'that', 'these', 'those', 'which', 'who', 'whom', 'whose',
             'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
             'not', 'only', 'also', 'even', 'just', 'still', 'already', 'always',
             'system', 'method', 'process', 'theory', 'concept', 'principle',
             'model', 'approach', 'technique', 'strategy', 'structure', 'function',
             'analysis', 'research', 'study', 'experiment', 'observation', 'result',
             'data', 'information', 'knowledge', 'understanding', 'explanation',
             'development', 'growth', 'change', 'evolution', 'progress', 'history',
             'science', 'technology', 'engineering', 'mathematics', 'physics',
             'chemistry', 'biology', 'medicine', 'psychology', 'philosophy',
             'language', 'literature', 'art', 'music', 'culture', 'society',
             'economy', 'politics', 'government', 'law', 'education', 'health']

    for _ in range(num_articles):
        # Generate article with multiple paragraphs
        num_paragraphs = random.randint(3, 15)
        paragraphs = []

        for _ in range(num_paragraphs):
            num_sentences = random.randint(3, 10)
            sentences = []

            for _ in range(num_sentences):
                num_words = random.randint(10, 30)
                sentence = ' '.join(random.choices(vocab, k=num_words))
                sentence = sentence.capitalize() + '.'
                sentences.append(sentence)

            paragraphs.append(' '.join(sentences))

        texts.append('\n\n'.join(paragraphs))

    return texts


def _generate_synthetic_book_text(num_books: int = 50) -> List[str]:
    """Generate synthetic book-like text for testing."""
    texts = []

    for _ in range(num_books):
        # Generate longer content with chapters
        num_chapters = random.randint(5, 20)
        chapters = []

        for ch in range(num_chapters):
            chapter_title = f"Chapter {ch + 1}"

            # Each chapter has multiple paragraphs
            num_paragraphs = random.randint(10, 30)
            paragraphs = [_generate_synthetic_paragraph() for _ in range(num_paragraphs)]

            chapters.append(f"{chapter_title}\n\n" + '\n\n'.join(paragraphs))

        texts.append('\n\n'.join(chapters))

    return texts


def _generate_synthetic_paragraph() -> str:
    """Generate a synthetic paragraph."""
    vocab = ['the', 'a', 'is', 'was', 'had', 'with', 'for', 'that', 'this', 'from',
             'he', 'she', 'they', 'it', 'we', 'you', 'his', 'her', 'their', 'our',
             'said', 'looked', 'walked', 'thought', 'felt', 'saw', 'heard', 'knew',
             'room', 'door', 'window', 'house', 'street', 'city', 'world', 'time',
             'day', 'night', 'morning', 'evening', 'year', 'moment', 'life', 'death',
             'hand', 'face', 'eyes', 'voice', 'heart', 'mind', 'body', 'soul',
             'never', 'always', 'sometimes', 'perhaps', 'suddenly', 'slowly', 'quickly',
             'small', 'large', 'old', 'young', 'dark', 'light', 'cold', 'warm']

    num_sentences = random.randint(3, 8)
    sentences = []

    for _ in range(num_sentences):
        num_words = random.randint(8, 25)
        sentence = ' '.join(random.choices(vocab, k=num_words))
        sentence = sentence.capitalize() + '.'
        sentences.append(sentence)

    return ' '.join(sentences)


# =============================================================================
# Data Loaders Factory
# =============================================================================

def create_dataloaders(
    tokenizer: TokenizerWrapper,
    dataset_name: str = "wikitext",
    batch_size: int = 8,
    train_length_range: Tuple[int, int] = (512, 2048),
    eval_length_ranges: List[Tuple[int, int]] = None,
    num_workers: int = 4,
    curriculum: bool = False,
) -> Dict[str, DataLoader]:
    """
    Create train and evaluation dataloaders.

    Args:
        tokenizer: Tokenizer wrapper
        dataset_name: Dataset to load ('wikitext', 'pg19', 'synthetic')
        batch_size: Batch size
        train_length_range: Length range for training
        eval_length_ranges: List of length ranges for evaluation
        num_workers: Number of data loading workers
        curriculum: Whether to use curriculum learning

    Returns:
        Dictionary of dataloaders {'train': ..., 'eval_512': ..., 'eval_2048': ...}
    """
    if eval_length_ranges is None:
        eval_length_ranges = [
            (512, 1024),
            (1024, 2048),
            (2048, 4096),
            (4096, 8192),
            (8192, 16384),
            (16384, 32768),
        ]

    # Load dataset
    if dataset_name == "wikitext":
        train_dataset = load_wikitext(tokenizer, split="train")
        eval_dataset = load_wikitext(tokenizer, split="validation")
    elif dataset_name == "pg19":
        train_dataset = load_pg19(tokenizer, split="train")
        eval_dataset = load_pg19(tokenizer, split="validation")
    elif dataset_name == "synthetic":
        train_dataset = SyntheticLongDocumentDataset(tokenizer, num_samples=10000)
        eval_dataset = SyntheticLongDocumentDataset(tokenizer, num_samples=1000, seed=123)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataloaders = {}

    # Training dataloader with optional curriculum
    train_sampler = LengthBucketSampler(
        train_dataset,
        batch_size=batch_size,
        curriculum=curriculum,
        curriculum_progress=0.0 if curriculum else 1.0,
    )
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=lambda b: collate_variable_length(b, tokenizer.pad_token_id),
        num_workers=num_workers,
    )
    dataloaders['train_sampler'] = train_sampler  # For curriculum updates

    # Evaluation dataloaders for each length range
    for min_len, max_len in eval_length_ranges:
        bucket_idx = train_dataset._get_bucket_index((min_len + max_len) // 2)
        if bucket_idx >= 0 and len(train_dataset.get_bucket_samples(bucket_idx)) > 0:
            eval_sampler = LengthBucketSampler(
                eval_dataset,
                batch_size=batch_size,
                bucket_weights={bucket_idx: 1.0},
            )
            dataloaders[f'eval_{min_len}_{max_len}'] = DataLoader(
                eval_dataset,
                batch_sampler=eval_sampler,
                collate_fn=lambda b: collate_variable_length(b, tokenizer.pad_token_id),
                num_workers=num_workers,
            )

    return dataloaders


if __name__ == "__main__":
    # Test the data infrastructure
    print("Testing NLP Dataset Infrastructure...")

    # Create tokenizer
    tokenizer = TokenizerWrapper("simple")

    # Generate and fit on synthetic data
    synthetic_texts = _generate_synthetic_wiki_text(100)
    tokenizer.fit(synthetic_texts)

    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create dataset
    dataset = LengthGroupedDataset(synthetic_texts, tokenizer)
    print(f"Dataset size: {len(dataset)}")
    print(f"Length distribution: {dataset.get_length_distribution()}")

    # Test synthetic long document dataset
    synthetic_dataset = SyntheticLongDocumentDataset(tokenizer, num_samples=100)
    print(f"Synthetic dataset size: {len(synthetic_dataset)}")

    # Test collate function
    batch = [dataset[i] for i in range(4)]
    collated = collate_variable_length(batch)
    print(f"Collated batch shape: {collated['input_ids'].shape}")

    print("NLP Dataset Infrastructure test complete!")
