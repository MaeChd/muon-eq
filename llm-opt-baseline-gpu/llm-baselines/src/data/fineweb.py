import numpy as np
from pathlib import Path

HEADER_MAGIC = 20240520
HEADER_VERSION = 1
HEADER_INT32_COUNT = 256
TOKEN_DTYPE = np.uint16


def _build_memmap_spec(path):
    path = Path(path)
    token_dtype = np.dtype(TOKEN_DTYPE)
    file_size = path.stat().st_size

    header = np.fromfile(path, dtype=np.int32, count=HEADER_INT32_COUNT)
    if len(header) >= 3 and int(header[0]) == HEADER_MAGIC:
        if int(header[1]) != HEADER_VERSION:
            raise ValueError(
                f"Unsupported fineweb shard version in {path}: "
                f"expected {HEADER_VERSION}, got {int(header[1])}"
            )
        offset_bytes = HEADER_INT32_COUNT * np.dtype(np.int32).itemsize
        num_tokens = int(header[2])
        expected_size = offset_bytes + num_tokens * token_dtype.itemsize
        if file_size < expected_size:
            raise ValueError(
                f"Fineweb shard is smaller than header claims: {path}\n"
                f"header_tokens={num_tokens} expected_bytes={expected_size} actual_bytes={file_size}"
            )
        return {
            "path": str(path),
            "dtype": str(token_dtype),
            "offset_bytes": offset_bytes,
            "length": num_tokens,
        }

    if file_size % token_dtype.itemsize != 0:
        raise ValueError(
            f"Raw bin file size is not divisible by token dtype size: {path}"
        )

    return {
        "path": str(path),
        "dtype": str(token_dtype),
        "offset_bytes": 0,
        "length": file_size // token_dtype.itemsize,
    }


def _find_sharded_split_files(root, split):
    patterns = [
        f"fineweb_{split}_*.bin",
        f"{split}_*.bin",
    ]

    matches = []
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches
    return []


def _resolve_local_bin_dataset(datasets_dir, dataset_name, candidate_subdirs):
    base_dir = Path(datasets_dir).expanduser()
    search_roots = [base_dir]
    search_roots.extend(base_dir / subdir for subdir in candidate_subdirs)

    checked_paths = []
    for root in search_roots:
        checked_paths.append(str(root))

        train_path = root / "train.bin"
        val_path = root / "val.bin"
        if train_path.is_file() and val_path.is_file():
            return {
                "train": _build_memmap_spec(train_path),
                "val": _build_memmap_spec(val_path),
            }

        train_shards = _find_sharded_split_files(root, "train")
        val_shards = _find_sharded_split_files(root, "val")
        if train_shards and val_shards:
            return {
                "train": [_build_memmap_spec(path) for path in train_shards],
                "val": [_build_memmap_spec(path) for path in val_shards],
            }

    checked_str = "\n".join(f"  - {path}" for path in checked_paths)
    raise FileNotFoundError(
        f"Local dataset '{dataset_name}' not found.\n"
        "Expected either:\n"
        "  - single-file bins: train.bin and val.bin\n"
        "  - sharded bins: fineweb_train_*.bin and fineweb_val_*.bin\n"
        f"in one of:\n{checked_str}\n"
        "Automatic download from Hugging Face is disabled."
    )


def get_fineweb_data(datasets_dir, num_proc=40):
    del num_proc
    return _resolve_local_bin_dataset(
        datasets_dir,
        dataset_name="fineweb",
        candidate_subdirs=["fineweb-100BT", "fineweb"],
    )


def get_fineweb10b_data(datasets_dir, num_proc=40):
    del num_proc
    return _resolve_local_bin_dataset(
        datasets_dir,
        dataset_name="fineweb10b",
        candidate_subdirs=["fineweb-10b", "fineweb10b"],
    )


if __name__ == "__main__":
    get_fineweb_data("./datasets/")
