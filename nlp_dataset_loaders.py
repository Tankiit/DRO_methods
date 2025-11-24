"""
Comprehensive Dataset Loaders for NLP Length Generalization Experiments

This module provides dataset loaders for various NLP datasets suitable
for studying length generalization:

Language Modeling Datasets:
- WikiText-2/103: Wikipedia articles (standard LM benchmark)
- PG19: Full books from Project Gutenberg (very long documents)
- OpenWebText: Web text data
- C4: Colossal Clean Crawled Corpus

Long Document Datasets:
- ArXiv: Scientific papers with LaTeX structure
- GovReport: Government reports for summarization
- BookSum: Book chapter summarization
- NarrativeQA: Story comprehension

Long-Context Benchmarks:
- SCROLLS: Long document understanding benchmark
- LongBench: Comprehensive long-context evaluation

Each dataset is processed to support:
1. Variable length sequences
2. Length-based grouping
3. Curriculum learning
4. Cross-length evaluation
"""

import os
import json
import random
import hashlib
from typing import Optional, Dict, List, Tuple, Union, Iterator
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import warnings

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np

# Check for optional dependencies
try:
    from datasets import load_dataset, DatasetDict
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    warnings.warn("'datasets' library not installed. Some datasets will not be available.")

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    description: str
    source: str
    avg_length: int  # Average tokens per document
    max_length: int  # Typical maximum length
    has_structure: bool  # Whether documents have clear structure
    domains: List[str]
    license: str
    download_size: str
    requires_auth: bool = False


# Dataset registry with metadata
DATASET_REGISTRY: Dict[str, DatasetInfo] = {
    "wikitext-2": DatasetInfo(
        name="WikiText-2",
        description="Small Wikipedia dataset for language modeling",
        source="wikitext",
        avg_length=500,
        max_length=4096,
        has_structure=True,
        domains=["wikipedia", "encyclopedia"],
        license="CC BY-SA 3.0",
        download_size="~4MB",
    ),
    "wikitext-103": DatasetInfo(
        name="WikiText-103",
        description="Large Wikipedia dataset (103M tokens)",
        source="wikitext",
        avg_length=3500,
        max_length=16384,
        has_structure=True,
        domains=["wikipedia", "encyclopedia"],
        license="CC BY-SA 3.0",
        download_size="~180MB",
    ),
    "pg19": DatasetInfo(
        name="PG19",
        description="Full books from Project Gutenberg (1919 and earlier)",
        source="pg19",
        avg_length=70000,
        max_length=500000,
        has_structure=True,
        domains=["books", "literature", "fiction", "non-fiction"],
        license="Public Domain",
        download_size="~11GB",
    ),
    "openwebtext": DatasetInfo(
        name="OpenWebText",
        description="Open-source recreation of WebText",
        source="openwebtext",
        avg_length=800,
        max_length=8192,
        has_structure=False,
        domains=["web", "news", "blogs"],
        license="Various",
        download_size="~40GB",
    ),
    "arxiv": DatasetInfo(
        name="ArXiv Papers",
        description="Scientific papers from ArXiv",
        source="scientific_papers",
        avg_length=8000,
        max_length=50000,
        has_structure=True,
        domains=["science", "mathematics", "physics", "cs"],
        license="Various (mostly CC)",
        download_size="~8GB",
    ),
    "pubmed": DatasetInfo(
        name="PubMed Abstracts",
        description="Biomedical literature abstracts",
        source="scientific_papers",
        avg_length=300,
        max_length=2048,
        has_structure=True,
        domains=["medicine", "biology", "health"],
        license="Various",
        download_size="~3GB",
    ),
    "govreport": DatasetInfo(
        name="GovReport",
        description="Long government reports for summarization",
        source="ccdv/govreport-summarization",
        avg_length=9000,
        max_length=50000,
        has_structure=True,
        domains=["government", "policy", "legal"],
        license="Public Domain",
        download_size="~700MB",
    ),
    "booksum": DatasetInfo(
        name="BookSum",
        description="Book chapter summarization dataset",
        source="kmfoda/booksum",
        avg_length=5000,
        max_length=30000,
        has_structure=True,
        domains=["books", "literature"],
        license="Various",
        download_size="~2GB",
    ),
    "scrolls-qasper": DatasetInfo(
        name="SCROLLS Qasper",
        description="Question answering on scientific papers",
        source="tau/scrolls",
        avg_length=4500,
        max_length=16384,
        has_structure=True,
        domains=["science", "qa"],
        license="CC BY 4.0",
        download_size="~100MB",
        requires_auth=False,
    ),
    "scrolls-narrative_qa": DatasetInfo(
        name="SCROLLS NarrativeQA",
        description="Story comprehension benchmark",
        source="tau/scrolls",
        avg_length=60000,
        max_length=200000,
        has_structure=True,
        domains=["books", "qa"],
        license="Apache 2.0",
        download_size="~500MB",
    ),
    "scrolls-quality": DatasetInfo(
        name="SCROLLS QuALITY",
        description="Multiple choice QA on long texts",
        source="tau/scrolls",
        avg_length=5000,
        max_length=16384,
        has_structure=True,
        domains=["articles", "qa"],
        license="CC BY 4.0",
        download_size="~200MB",
    ),
    "c4": DatasetInfo(
        name="C4",
        description="Colossal Clean Crawled Corpus",
        source="c4",
        avg_length=500,
        max_length=8192,
        has_structure=False,
        domains=["web"],
        license="ODC-BY",
        download_size="~350GB",
    ),
    "the_pile": DatasetInfo(
        name="The Pile",
        description="Large diverse text corpus",
        source="EleutherAI/pile",
        avg_length=2000,
        max_length=32768,
        has_structure=False,
        domains=["web", "books", "code", "academic"],
        license="Various",
        download_size="~800GB",
        requires_auth=True,
    ),
}


def list_available_datasets() -> Dict[str, DatasetInfo]:
    """List all available datasets with their information."""
    return DATASET_REGISTRY


def get_dataset_info(name: str) -> DatasetInfo:
    """Get information about a specific dataset."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]


# =============================================================================
# Base Dataset Classes
# =============================================================================

class LongDocumentDataset(Dataset):
    """
    Base class for long document datasets with length-based organization.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 8192,
        min_length: int = 64,
        stride: int = None,
        return_overflowing_tokens: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.stride = stride or max_length // 2
        self.return_overflowing_tokens = return_overflowing_tokens

        # Process texts into samples
        self.samples = []
        self.length_to_indices = defaultdict(list)

        self._process_texts(texts)

    def _process_texts(self, texts: List[str]):
        """Process raw texts into tokenized samples."""
        for doc_idx, text in enumerate(texts):
            if not text or len(text.strip()) < 10:
                continue

            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.squeeze().tolist()
            if not isinstance(tokens, list):
                tokens = list(tokens)

            # Handle long documents with sliding window
            if len(tokens) <= self.max_length:
                if len(tokens) >= self.min_length:
                    self._add_sample(tokens, doc_idx, 0)
            else:
                # Sliding window
                for start in range(0, len(tokens) - self.min_length, self.stride):
                    end = min(start + self.max_length, len(tokens))
                    chunk_tokens = tokens[start:end]
                    if len(chunk_tokens) >= self.min_length:
                        self._add_sample(chunk_tokens, doc_idx, start)

    def _add_sample(self, tokens: List[int], doc_idx: int, offset: int):
        """Add a sample and update length index."""
        sample_idx = len(self.samples)
        self.samples.append({
            'tokens': tokens,
            'length': len(tokens),
            'doc_idx': doc_idx,
            'offset': offset,
        })

        # Index by length bucket
        length_bucket = self._get_length_bucket(len(tokens))
        self.length_to_indices[length_bucket].append(sample_idx)

    def _get_length_bucket(self, length: int) -> int:
        """Get bucket index for a given length."""
        buckets = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        for i, bucket in enumerate(buckets):
            if length <= bucket:
                return i
        return len(buckets)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        tokens = torch.tensor(sample['tokens'], dtype=torch.long)

        return {
            'input_ids': tokens,
            'length': sample['length'],
            'doc_idx': sample['doc_idx'],
        }

    def get_samples_by_length(self, min_len: int, max_len: int) -> List[int]:
        """Get sample indices within a length range."""
        indices = []
        for bucket, bucket_indices in self.length_to_indices.items():
            for idx in bucket_indices:
                length = self.samples[idx]['length']
                if min_len <= length < max_len:
                    indices.append(idx)
        return indices

    def get_length_distribution(self) -> Dict[str, int]:
        """Get distribution of samples across length buckets."""
        bucket_names = ['≤128', '129-256', '257-512', '513-1024', '1025-2048',
                       '2049-4096', '4097-8192', '8193-16384', '16385-32768', '>32768']
        return {
            bucket_names[bucket]: len(indices)
            for bucket, indices in sorted(self.length_to_indices.items())
        }


class StreamingLongDocumentDataset(IterableDataset):
    """
    Streaming dataset for very large corpora that don't fit in memory.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        max_length: int = 8192,
        min_length: int = 64,
        split: str = "train",
        streaming: bool = True,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.split = split
        self.streaming = streaming

        if not HAS_DATASETS:
            raise ImportError("'datasets' library required for streaming datasets")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        dataset = load_dataset(self.dataset_name, split=self.split, streaming=self.streaming)

        for item in dataset:
            text = self._extract_text(item)
            if not text or len(text) < 10:
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.squeeze().tolist()

            # Yield chunks
            if len(tokens) <= self.max_length:
                if len(tokens) >= self.min_length:
                    yield {
                        'input_ids': torch.tensor(tokens, dtype=torch.long),
                        'length': len(tokens),
                    }
            else:
                stride = self.max_length // 2
                for start in range(0, len(tokens) - self.min_length, stride):
                    end = min(start + self.max_length, len(tokens))
                    chunk = tokens[start:end]
                    if len(chunk) >= self.min_length:
                        yield {
                            'input_ids': torch.tensor(chunk, dtype=torch.long),
                            'length': len(chunk),
                        }

    def _extract_text(self, item: Dict) -> str:
        """Extract text field from dataset item."""
        # Handle different dataset formats
        if 'text' in item:
            return item['text']
        elif 'content' in item:
            return item['content']
        elif 'article' in item:
            return item['article']
        elif 'document' in item:
            return item['document']
        return ""


# =============================================================================
# Specific Dataset Loaders
# =============================================================================

class WikiTextLoader:
    """Loader for WikiText-2 and WikiText-103."""

    def __init__(self, version: str = "103", cache_dir: str = "./data"):
        self.version = version
        self.cache_dir = cache_dir

    def load(
        self,
        tokenizer,
        split: str = "train",
        max_length: int = 8192,
        max_samples: int = None,
    ) -> LongDocumentDataset:
        """Load WikiText dataset."""
        if not HAS_DATASETS:
            raise ImportError("'datasets' library required for WikiText")

        dataset = load_dataset(
            "wikitext",
            f"wikitext-{self.version}-raw-v1",
            split=split,
            cache_dir=self.cache_dir,
        )

        # Filter empty texts and combine paragraphs into documents
        texts = []
        current_doc = []

        for item in dataset:
            text = item['text'].strip()
            if text.startswith('=') and text.endswith('='):
                # New section header - start new document
                if current_doc:
                    texts.append('\n'.join(current_doc))
                    current_doc = []
            if text:
                current_doc.append(text)

        if current_doc:
            texts.append('\n'.join(current_doc))

        # Filter by length
        texts = [t for t in texts if len(t) > 100]

        if max_samples:
            texts = texts[:max_samples]

        print(f"WikiText-{self.version}: {len(texts)} documents")

        return LongDocumentDataset(texts, tokenizer, max_length=max_length)


class PG19Loader:
    """Loader for PG19 (Project Gutenberg books)."""

    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = cache_dir

    def load(
        self,
        tokenizer,
        split: str = "train",
        max_length: int = 16384,
        max_books: int = 100,
    ) -> LongDocumentDataset:
        """Load PG19 dataset."""
        if not HAS_DATASETS:
            raise ImportError("'datasets' library required for PG19")

        dataset = load_dataset("pg19", split=split, cache_dir=self.cache_dir)

        texts = []
        for i, item in enumerate(dataset):
            if max_books and i >= max_books:
                break
            texts.append(item['text'])

        print(f"PG19: {len(texts)} books")

        return LongDocumentDataset(texts, tokenizer, max_length=max_length)


class ArXivLoader:
    """Loader for ArXiv scientific papers."""

    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = cache_dir

    def load(
        self,
        tokenizer,
        split: str = "train",
        max_length: int = 16384,
        max_papers: int = 1000,
    ) -> LongDocumentDataset:
        """Load ArXiv dataset."""
        if not HAS_DATASETS:
            raise ImportError("'datasets' library required for ArXiv")

        dataset = load_dataset(
            "scientific_papers",
            "arxiv",
            split=split,
            cache_dir=self.cache_dir,
        )

        texts = []
        for i, item in enumerate(dataset):
            if max_papers and i >= max_papers:
                break
            # Combine abstract and article
            text = f"Abstract: {item['abstract']}\n\n{item['article']}"
            texts.append(text)

        print(f"ArXiv: {len(texts)} papers")

        return LongDocumentDataset(texts, tokenizer, max_length=max_length)


class GovReportLoader:
    """Loader for GovReport long government reports."""

    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = cache_dir

    def load(
        self,
        tokenizer,
        split: str = "train",
        max_length: int = 16384,
        max_reports: int = 500,
    ) -> LongDocumentDataset:
        """Load GovReport dataset."""
        if not HAS_DATASETS:
            raise ImportError("'datasets' library required for GovReport")

        dataset = load_dataset(
            "ccdv/govreport-summarization",
            split=split,
            cache_dir=self.cache_dir,
        )

        texts = []
        for i, item in enumerate(dataset):
            if max_reports and i >= max_reports:
                break
            texts.append(item['report'])

        print(f"GovReport: {len(texts)} reports")

        return LongDocumentDataset(texts, tokenizer, max_length=max_length)


class SCROLLSLoader:
    """Loader for SCROLLS long-context benchmarks."""

    SUBSETS = ['qasper', 'narrative_qa', 'quality', 'summ_screen_fd',
               'gov_report', 'contract_nli']

    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = cache_dir

    def load(
        self,
        tokenizer,
        subset: str = "qasper",
        split: str = "train",
        max_length: int = 16384,
        max_samples: int = 500,
    ) -> LongDocumentDataset:
        """Load SCROLLS subset."""
        if not HAS_DATASETS:
            raise ImportError("'datasets' library required for SCROLLS")

        if subset not in self.SUBSETS:
            raise ValueError(f"Unknown SCROLLS subset: {subset}. Available: {self.SUBSETS}")

        dataset = load_dataset(
            "tau/scrolls",
            subset,
            split=split,
            cache_dir=self.cache_dir,
        )

        texts = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            # SCROLLS uses 'input' field for documents
            texts.append(item['input'])

        print(f"SCROLLS-{subset}: {len(texts)} documents")

        return LongDocumentDataset(texts, tokenizer, max_length=max_length)


class OpenWebTextLoader:
    """Loader for OpenWebText."""

    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = cache_dir

    def load(
        self,
        tokenizer,
        split: str = "train",
        max_length: int = 8192,
        max_samples: int = 10000,
        streaming: bool = False,
    ) -> Union[LongDocumentDataset, StreamingLongDocumentDataset]:
        """Load OpenWebText dataset."""
        if not HAS_DATASETS:
            raise ImportError("'datasets' library required for OpenWebText")

        if streaming:
            return StreamingLongDocumentDataset(
                "openwebtext",
                tokenizer,
                max_length=max_length,
                split=split,
            )

        dataset = load_dataset(
            "openwebtext",
            split=split,
            cache_dir=self.cache_dir,
        )

        texts = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            texts.append(item['text'])

        print(f"OpenWebText: {len(texts)} documents")

        return LongDocumentDataset(texts, tokenizer, max_length=max_length)


class C4Loader:
    """Loader for C4 (Colossal Clean Crawled Corpus)."""

    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = cache_dir

    def load(
        self,
        tokenizer,
        split: str = "train",
        max_length: int = 8192,
        max_samples: int = 10000,
        streaming: bool = True,  # Default to streaming due to size
    ) -> Union[LongDocumentDataset, StreamingLongDocumentDataset]:
        """Load C4 dataset."""
        if not HAS_DATASETS:
            raise ImportError("'datasets' library required for C4")

        if streaming:
            return StreamingLongDocumentDataset(
                "c4",
                tokenizer,
                max_length=max_length,
                split=split,
            )

        dataset = load_dataset(
            "c4",
            "en",
            split=split,
            cache_dir=self.cache_dir,
            streaming=False,
        )

        texts = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            texts.append(item['text'])

        print(f"C4: {len(texts)} documents")

        return LongDocumentDataset(texts, tokenizer, max_length=max_length)


class BookSumLoader:
    """Loader for BookSum chapter summarization."""

    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = cache_dir

    def load(
        self,
        tokenizer,
        split: str = "train",
        max_length: int = 16384,
        max_chapters: int = 500,
    ) -> LongDocumentDataset:
        """Load BookSum dataset."""
        if not HAS_DATASETS:
            raise ImportError("'datasets' library required for BookSum")

        dataset = load_dataset(
            "kmfoda/booksum",
            split=split,
            cache_dir=self.cache_dir,
        )

        texts = []
        for i, item in enumerate(dataset):
            if max_chapters and i >= max_chapters:
                break
            texts.append(item['chapter'])

        print(f"BookSum: {len(texts)} chapters")

        return LongDocumentDataset(texts, tokenizer, max_length=max_length)


# =============================================================================
# Synthetic Dataset for Controlled Experiments
# =============================================================================

class SyntheticLengthDataset(Dataset):
    """
    Synthetic dataset for controlled length generalization experiments.

    Features:
    - Controllable document length distribution
    - Embedded long-range patterns for evaluation
    - Controllable complexity levels
    - Ground truth for pattern detection
    """

    def __init__(
        self,
        tokenizer,
        num_samples: int = 10000,
        length_distribution: str = "uniform",  # 'uniform', 'exponential', 'power_law'
        min_length: int = 256,
        max_length: int = 8192,
        num_patterns: int = 20,
        vocab_subset: int = 5000,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.vocab_size = min(vocab_subset, getattr(tokenizer, 'vocab_size', 10000))
        self.num_patterns = num_patterns
        self.length_distribution = length_distribution
        self.min_length = min_length
        self.max_length = max_length

        np.random.seed(seed)
        random.seed(seed)

        # Create patterns
        self.patterns = self._create_patterns()

        # Generate samples
        self.samples = self._generate_samples(num_samples)

    def _create_patterns(self) -> List[Dict]:
        """Create patterns at different distance scales."""
        patterns = []

        for i in range(self.num_patterns):
            # Pattern type determines distance
            pattern_type = random.choice(['local', 'medium', 'long', 'very_long'])

            distance_ranges = {
                'local': (5, 50),
                'medium': (100, 500),
                'long': (500, 2000),
                'very_long': (2000, 5000),
            }

            trigger_len = random.randint(3, 8)
            response_len = random.randint(3, 8)

            trigger = [random.randint(10, self.vocab_size - 1) for _ in range(trigger_len)]
            response = [random.randint(10, self.vocab_size - 1) for _ in range(response_len)]

            distance = random.randint(*distance_ranges[pattern_type])

            patterns.append({
                'id': i,
                'trigger': trigger,
                'response': response,
                'distance': distance,
                'type': pattern_type,
            })

        return patterns

    def _generate_samples(self, num_samples: int) -> List[Dict]:
        """Generate samples with embedded patterns."""
        samples = []

        for _ in range(num_samples):
            # Sample length
            if self.length_distribution == 'uniform':
                length = random.randint(self.min_length, self.max_length)
            elif self.length_distribution == 'exponential':
                length = int(np.random.exponential(self.max_length / 4))
                length = max(self.min_length, min(self.max_length, length))
            elif self.length_distribution == 'power_law':
                alpha = 2.0
                length = int(self.min_length * (random.random() ** (-1 / (alpha - 1))))
                length = min(self.max_length, length)
            else:
                length = random.randint(self.min_length, self.max_length)

            # Generate base sequence
            sequence = [random.randint(10, self.vocab_size - 1) for _ in range(length)]

            # Embed patterns
            embedded = []
            for pattern in self.patterns:
                if random.random() < 0.3:  # 30% chance to include each pattern
                    if pattern['distance'] + len(pattern['response']) < length:
                        trigger_pos = random.randint(0, length - pattern['distance'] - len(pattern['response']))
                        response_pos = trigger_pos + pattern['distance']

                        # Insert trigger
                        for j, tok in enumerate(pattern['trigger']):
                            if trigger_pos + j < length:
                                sequence[trigger_pos + j] = tok

                        # Insert response
                        for j, tok in enumerate(pattern['response']):
                            if response_pos + j < length:
                                sequence[response_pos + j] = tok

                        embedded.append({
                            'pattern_id': pattern['id'],
                            'trigger_pos': trigger_pos,
                            'response_pos': response_pos,
                            'type': pattern['type'],
                        })

            samples.append({
                'tokens': sequence,
                'length': length,
                'patterns': embedded,
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            'input_ids': torch.tensor(sample['tokens'], dtype=torch.long),
            'length': sample['length'],
            'patterns': sample['patterns'],
        }

    def evaluate_pattern_accuracy(
        self,
        predictions: torch.Tensor,
        sample_idx: int,
    ) -> Dict[str, float]:
        """Evaluate pattern prediction accuracy by type."""
        sample = self.samples[sample_idx]
        results = defaultdict(list)

        for emb in sample['patterns']:
            pattern = self.patterns[emb['pattern_id']]
            response_pos = emb['response_pos']

            correct = 0
            total = len(pattern['response'])

            for j, expected in enumerate(pattern['response']):
                pos = response_pos + j
                if pos < len(predictions):
                    if predictions.dim() > 1:
                        pred = predictions[pos].argmax().item()
                    else:
                        pred = predictions[pos].item()
                    if pred == expected:
                        correct += 1

            results[emb['type']].append(correct / total if total > 0 else 0)

        return {ptype: np.mean(accs) if accs else 0 for ptype, accs in results.items()}


# =============================================================================
# Dataset Factory and Utilities
# =============================================================================

def load_dataset_by_name(
    name: str,
    tokenizer,
    split: str = "train",
    max_length: int = 8192,
    max_samples: int = None,
    cache_dir: str = "./data",
    **kwargs,
) -> Dataset:
    """
    Load a dataset by name.

    Args:
        name: Dataset name (see DATASET_REGISTRY)
        tokenizer: Tokenizer to use
        split: Data split ('train', 'validation', 'test')
        max_length: Maximum sequence length
        max_samples: Maximum number of samples (None for all)
        cache_dir: Directory for caching datasets
        **kwargs: Additional dataset-specific arguments

    Returns:
        Dataset instance
    """
    if name == "wikitext-2" or name == "wikitext-103":
        version = name.split("-")[1]
        loader = WikiTextLoader(version, cache_dir)
        return loader.load(tokenizer, split, max_length, max_samples)

    elif name == "pg19":
        loader = PG19Loader(cache_dir)
        return loader.load(tokenizer, split, max_length, max_samples or 100)

    elif name == "arxiv":
        loader = ArXivLoader(cache_dir)
        return loader.load(tokenizer, split, max_length, max_samples or 1000)

    elif name == "govreport":
        loader = GovReportLoader(cache_dir)
        return loader.load(tokenizer, split, max_length, max_samples or 500)

    elif name == "openwebtext":
        loader = OpenWebTextLoader(cache_dir)
        streaming = kwargs.get('streaming', False)
        return loader.load(tokenizer, split, max_length, max_samples, streaming)

    elif name == "c4":
        loader = C4Loader(cache_dir)
        streaming = kwargs.get('streaming', True)
        return loader.load(tokenizer, split, max_length, max_samples, streaming)

    elif name == "booksum":
        loader = BookSumLoader(cache_dir)
        return loader.load(tokenizer, split, max_length, max_samples or 500)

    elif name.startswith("scrolls-"):
        subset = name.replace("scrolls-", "")
        loader = SCROLLSLoader(cache_dir)
        return loader.load(tokenizer, subset, split, max_length, max_samples or 500)

    elif name == "synthetic":
        return SyntheticLengthDataset(
            tokenizer,
            num_samples=max_samples or 10000,
            max_length=max_length,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")


def create_length_stratified_splits(
    dataset: Dataset,
    length_buckets: List[Tuple[int, int]] = None,
    eval_ratio: float = 0.1,
) -> Dict[str, List[int]]:
    """
    Create train/eval splits stratified by sequence length.

    Args:
        dataset: Dataset to split
        length_buckets: List of (min_len, max_len) tuples
        eval_ratio: Fraction for evaluation

    Returns:
        Dictionary with 'train' and 'eval' index lists per bucket
    """
    if length_buckets is None:
        length_buckets = [
            (64, 512),
            (512, 1024),
            (1024, 2048),
            (2048, 4096),
            (4096, 8192),
            (8192, 16384),
        ]

    splits = {
        'train': defaultdict(list),
        'eval': defaultdict(list),
    }

    for idx in range(len(dataset)):
        item = dataset[idx]
        length = item.get('length', len(item['input_ids']))

        for bucket_idx, (min_len, max_len) in enumerate(length_buckets):
            if min_len <= length < max_len:
                if random.random() < eval_ratio:
                    splits['eval'][bucket_idx].append(idx)
                else:
                    splits['train'][bucket_idx].append(idx)
                break

    return splits


def collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Collate function for variable length batches."""
    max_len = max(item['input_ids'].size(0) for item in batch)

    input_ids = []
    attention_mask = []
    lengths = []

    for item in batch:
        seq = item['input_ids']
        seq_len = seq.size(0)

        # Pad
        padded = torch.full((max_len,), pad_token_id, dtype=seq.dtype)
        padded[:seq_len] = seq
        input_ids.append(padded)

        # Mask
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
# Main / Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Available Datasets for Length Generalization Experiments")
    print("=" * 60)

    for name, info in DATASET_REGISTRY.items():
        print(f"\n{name}:")
        print(f"  Description: {info.description}")
        print(f"  Avg Length: {info.avg_length:,} tokens")
        print(f"  Max Length: {info.max_length:,} tokens")
        print(f"  Domains: {', '.join(info.domains)}")
        print(f"  Size: {info.download_size}")

    # Test synthetic dataset
    print("\n" + "=" * 60)
    print("Testing Synthetic Dataset")
    print("=" * 60)

    # Simple tokenizer mock
    class MockTokenizer:
        vocab_size = 10000
        pad_token_id = 0

        def encode(self, text, add_special_tokens=True):
            return list(range(len(text.split())))

    tokenizer = MockTokenizer()

    dataset = SyntheticLengthDataset(
        tokenizer,
        num_samples=100,
        min_length=256,
        max_length=4096,
    )

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample length: {sample['length']}")
    print(f"Number of patterns in sample: {len(sample['patterns'])}")

    print("\nDataset testing complete!")
