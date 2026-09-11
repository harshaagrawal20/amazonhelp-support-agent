"""
Unit tests for download_data utility functions.
"""

from pathlib import Path
import pytest

from scripts.download_data import (
    format_bytes,
    list_dataset_files,
    save_dataset_pointer,
    get_cached_dataset_path,
)


def test_format_bytes():
    assert format_bytes(500) == "500.00 B"
    assert format_bytes(1024) == "1.00 KB"
    assert format_bytes(1024 * 1024) == "1.00 MB"
    assert format_bytes(1024 * 1024 * 1024) == "1.00 GB"


def test_list_dataset_files(tmp_path: Path):
    # Non-existent directory returns empty list
    non_existent = tmp_path / "does_not_exist"
    assert list_dataset_files(non_existent) == []

    # Create dummy files
    file1 = tmp_path / "sample_a.csv"
    file2 = tmp_path / "sample_b.txt"
    file1.write_text("col1,col2\n1,2\n", encoding="utf-8")
    file2.write_text("Hello world", encoding="utf-8")

    files = list_dataset_files(tmp_path)
    assert len(files) == 2
    filenames = [f["name"] for f in files]
    assert "sample_a.csv" in filenames
    assert "sample_b.txt" in filenames
    assert all("size_bytes" in f and "size_human" in f for f in files)


def test_save_and_get_dataset_pointer(tmp_path: Path):
    target_data_dir = tmp_path / "mock_dataset"
    target_data_dir.mkdir()
    pointer_file = tmp_path / "sub" / "dataset_pointer.txt"

    # Should be None before saving
    assert get_cached_dataset_path(pointer_file) is None

    # Save pointer
    save_dataset_pointer(target_data_dir, pointer_file)
    assert pointer_file.exists()

    # Retrieve pointer
    cached_path = get_cached_dataset_path(pointer_file)
    assert cached_path is not None
    assert cached_path.resolve() == target_data_dir.resolve()


def test_find_dataset_csv(tmp_path: Path):
    from scripts.download_data import find_dataset_csv

    # Subdirectory structure mimicking kagglehub: root/twcs/twcs.csv
    sub_dir = tmp_path / "twcs"
    sub_dir.mkdir()
    target_csv = sub_dir / "twcs.csv"
    target_csv.write_text("tweet_id,author_id\n1,brand1\n", encoding="utf-8")

    found_path = find_dataset_csv(dataset_dir=tmp_path, filename="twcs.csv")
    assert found_path.resolve() == target_csv.resolve()

