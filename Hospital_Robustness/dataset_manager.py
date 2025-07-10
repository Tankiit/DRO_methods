#!/usr/bin/env python3
"""Dataset Manager CLI
Ensures required medical datasets exist locally.

Usage:
  python3 dataset_manager.py --datasets mimic_cxr nih_chest --data_dir ~/research/data
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from typing import Dict, List

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover
    requests = None

# ------------------------------------------------------------------ #
# ------------------------ Dataset registry ------------------------ #
# ------------------------------------------------------------------ #

DATASETS: Dict[str, Dict[str, str | None]] = {
    "mimic_cxr": {
        "url": None,
        "note": "Sign up and download from https://physionet.org/content/mimic-cxr/2.0.0/.",
    },
    "chexpert": {
        "url": None,
        "note": "Register and download from https://stanfordmlgroup.github.io/competitions/chexpert/",
    },
    "nih_chest": {
        "url": None,
        "note": "Request dataset at https://nihcc.app.box.com/v/ChestXray-NIHCC.",
    },
    "camelyon17": {
        "url": None,
        "note": "Request dataset at https://camelyon17.grand-challenge.org/.",
    },
}


# ------------------------------------------------------------------ #
# -------------------------- Core helpers -------------------------- #
# ------------------------------------------------------------------ #

def ensure_dataset(dataset: str, data_dir: str, overwrite: bool = False) -> str:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choices: {list(DATASETS.keys())}")

    target_dir = os.path.join(os.path.expanduser(data_dir), dataset)

    if os.path.exists(target_dir) and not overwrite:
        print(f"✓ {dataset} already prepared at {target_dir}")
        return target_dir

    os.makedirs(target_dir, exist_ok=True)

    url = DATASETS[dataset]["url"]
    if url is None:
        print(textwrap.dedent(
            f"""
            ⚠️  Manual step required for {dataset}.
               {DATASETS[dataset]['note']}
               Place/extract files into: {target_dir}
            """
        ))
        return target_dir

    if requests is None:
        print("requests not installed; cannot auto-download. Skipping.")
        return target_dir

    # Simple HTTP download (no resume)
    out_path = os.path.join(target_dir, os.path.basename(url))
    print(f"⬇️  Downloading {dataset} from {url} → {out_path}")

    with requests.get(url, stream=True, timeout=60) as r:  # type: ignore[attr-defined]
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print("Download complete. Extract the data if necessary.")
    return target_dir


# ------------------------------------------------------------------ #
# ----------------------------- CLI -------------------------------- #
# ------------------------------------------------------------------ #

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets locally.")
    parser.add_argument(
        "--datasets",
        "-d",
        nargs="+",
        default=list(DATASETS.keys()),
        help=f"Datasets to check (default: all). Choices: {list(DATASETS.keys())}",
    )
    parser.add_argument(
        "--data_dir",
        "-p",
        default="~/research/data",
        help="Base directory for datasets (default: ~/research/data).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing directories.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None):
    args = parse_args(argv)
    for ds in args.datasets:
        try:
            path = ensure_dataset(ds, args.data_dir, overwrite=args.overwrite)
            print(f"{ds}: ready at {path}\n")
        except Exception as exc:
            print(f"[Error] {ds}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main() 