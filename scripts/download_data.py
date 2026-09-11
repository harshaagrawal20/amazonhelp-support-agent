#!/usr/bin/env python3
"""
Dataset download and discovery script using KaggleHub.

Downloads the Twitter Customer Support dataset ('thoughtvector/customer-support-on-twitter')
and discovers the files, sizes, and locations.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_DATASET_HANDLE = "thoughtvector/customer-support-on-twitter"


def format_bytes(num_bytes: int) -> str:
    """Format bytes into a human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def list_dataset_files(dataset_path: Path) -> List[Dict[str, any]]:
    """
    List all files in the downloaded dataset directory with metadata.
    """
    files_info = []
    if not dataset_path.exists():
        return files_info

    for root, _, filenames in os.walk(dataset_path):
        for filename in sorted(filenames):
            full_path = Path(root) / filename
            size = full_path.stat().st_size
            files_info.append({
                "name": filename,
                "path": str(full_path),
                "size_bytes": size,
                "size_human": format_bytes(size),
            })
    return files_info


def save_dataset_pointer(dataset_path: Path, pointer_file: Path = Path("data/raw/dataset_location.txt")) -> None:
    """
    Save the absolute path of the downloaded dataset to a local reference file
    so downstream scripts and notebooks can locate it without re-downloading.
    """
    pointer_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pointer_file, "w", encoding="utf-8") as f:
        f.write(str(dataset_path.resolve()))
    print(f"[INFO] Saved dataset path pointer to: {pointer_file}")


def get_cached_dataset_path(pointer_file: Path = Path("data/raw/dataset_location.txt")) -> Optional[Path]:
    """
    Retrieve dataset path from local pointer file if it exists and is valid.
    """
    if pointer_file.exists():
        with open(pointer_file, "r", encoding="utf-8") as f:
            path_str = f.read().strip()
            if path_str and Path(path_str).exists():
                return Path(path_str)
    return None


def find_dataset_csv(dataset_dir: Optional[Path] = None, filename: str = "twcs.csv") -> Path:
    """
    Locate twcs.csv (or sample.csv) within dataset directory, checking subdirectories recursively.
    """
    base_dir = dataset_dir or get_cached_dataset_path()
    if base_dir is None or not base_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found. Please run download_dataset() first.")

    # Check directly in base_dir
    direct_path = base_dir / filename
    if direct_path.exists() and direct_path.is_file():
        return direct_path

    # Search subdirectories recursively
    for p in base_dir.rglob(filename):
        if p.is_file():
            return p

    raise FileNotFoundError(f"Could not find '{filename}' in {base_dir} or any subdirectory.")



def download_dataset(dataset_handle: str = DEFAULT_DATASET_HANDLE, force_download: bool = False) -> Path:
    """
    Download or retrieve dataset using kagglehub.
    """
    # Check if pointer already points to existing dataset directory
    if not force_download:
        cached_path = get_cached_dataset_path()
        if cached_path is not None and cached_path.exists():
            print(f"[INFO] Existing dataset found via pointer at: {cached_path}")
            return cached_path

    print(f"[INFO] Downloading dataset '{dataset_handle}' via kagglehub...")
    import kagglehub

    download_path = kagglehub.dataset_download(dataset_handle)
    path_obj = Path(download_path)
    print(f"[SUCCESS] Dataset downloaded/located at: {path_obj}")

    save_dataset_pointer(path_obj)
    return path_obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Twitter Customer Support dataset via KaggleHub.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_HANDLE,
        help=f"Kaggle dataset handle (default: {DEFAULT_DATASET_HANDLE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download via kagglehub even if local pointer exists",
    )
    args = parser.parse_args()

    try:
        dataset_path = download_dataset(dataset_handle=args.dataset, force_download=args.force)
        files = list_dataset_files(dataset_path)

        print("\n" + "=" * 60)
        print("DATASET INVENTORY")
        print("=" * 60)
        print(f"Directory: {dataset_path}")
        print(f"Total files: {len(files)}")
        for f in files:
            print(f" - {f['name']} ({f['size_human']}, {f['size_bytes']} bytes)")
        print("=" * 60 + "\n")
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to download or inspect dataset: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
