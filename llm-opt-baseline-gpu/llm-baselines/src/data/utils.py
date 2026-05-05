from pathlib import Path
from typing import Dict
import bisect

import numpy as np
import torch
import torch.distributed as dist

from .arxiv import get_arxiv_2000, get_arxiv_full
from .benchmarks import SUPPORTED_TASK_MAP
from .c4 import get_c4_data
from .fineweb import get_fineweb10b_data, get_fineweb_data
from .fineweb_edu import get_fineweb_edu_data
from .openwebtext2 import get_openwebtext2_data
from .redpajama import get_redpajama_data, get_redpajamav2_data
from .shakespeare import get_shakespeare_data
from .slimpajama import get_slimpajama_data
from .wikitext import get_wikitext_data


def _normalize_memmap_source(source, default_dtype=np.uint16):
    if isinstance(source, (str, Path)):
        path = Path(source)
        dtype = np.dtype(default_dtype)
        file_size = path.stat().st_size
        if file_size % dtype.itemsize != 0:
            raise ValueError(
                f"Memmap file size is not divisible by dtype size: {path}"
            )
        return {
            "path": path,
            "dtype": dtype,
            "offset_bytes": 0,
            "length": file_size // dtype.itemsize,
        }

    if isinstance(source, dict):
        path = Path(source["path"])
        dtype = np.dtype(source.get("dtype", default_dtype))
        offset_bytes = int(source.get("offset_bytes", 0))
        length = source.get("length")
        if length is None:
            file_size = path.stat().st_size - offset_bytes
            if file_size % dtype.itemsize != 0:
                raise ValueError(
                    f"Memmap remainder is not divisible by dtype size: {path}"
                )
            length = file_size // dtype.itemsize
        return {
            "path": path,
            "dtype": dtype,
            "offset_bytes": offset_bytes,
            "length": int(length),
        }

    raise TypeError(f"Unsupported memmap source type: {type(source)!r}")


class ShardedMemmapDataset:
    def __init__(self, paths, dtype=np.uint16):
        self.sources = [_normalize_memmap_source(path, default_dtype=dtype) for path in paths]
        self.dtype = np.dtype(dtype)
        self.lengths = [source["length"] for source in self.sources]
        self.offsets = np.cumsum([0] + self.lengths)
        self.total_length = int(self.offsets[-1])
        self._memmaps = [None] * len(self.sources)

    def __len__(self):
        return self.total_length

    def _get_memmap(self, shard_idx):
        memmap = self._memmaps[shard_idx]
        if memmap is None:
            source = self.sources[shard_idx]
            memmap = np.memmap(
                source["path"],
                dtype=source["dtype"],
                mode="r",
                offset=source["offset_bytes"],
                shape=(source["length"],),
            )
            self._memmaps[shard_idx] = memmap
        return memmap

    def _find_shard(self, idx):
        return bisect.bisect_right(self.offsets, idx) - 1

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.total_length)
            if step != 1:
                raise ValueError("ShardedMemmapDataset only supports contiguous slices.")
            if start >= stop:
                return np.empty((0,), dtype=self.dtype)

            parts = []
            cursor = start
            while cursor < stop:
                shard_idx = self._find_shard(cursor)
                shard_start = self.offsets[shard_idx]
                shard_end = self.offsets[shard_idx + 1]
                take = min(stop, shard_end) - cursor
                local_start = cursor - shard_start
                local_end = local_start + take
                parts.append(np.asarray(self._get_memmap(shard_idx)[local_start:local_end]))
                cursor += take

            return parts[0] if len(parts) == 1 else np.concatenate(parts)

        if idx < 0:
            idx += self.total_length
        if not 0 <= idx < self.total_length:
            raise IndexError(idx)

        shard_idx = self._find_shard(idx)
        local_idx = idx - self.offsets[shard_idx]
        return self._get_memmap(shard_idx)[local_idx]


def get_dataset(args) -> Dict[str, np.ndarray]:
    """Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
    contained in its own python file. The expected format at the moment is a dictionary of np.memmap
    containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data.
    """
    if args.dataset == "wikitext":
        return get_wikitext_data(args.datasets_dir)
    if args.dataset == "shakespeare-char":
        return get_shakespeare_data(args.datasets_dir)
    if args.dataset == "arxiv2000":
        return get_arxiv_2000(args.datasets_dir)
    if args.dataset == "arxiv":
        return get_arxiv_full(args.datasets_dir)
    if args.dataset == "arxiv+wiki":
        arxiv_data = get_arxiv_full(args.datasets_dir)
        wiki_data = get_wikitext_data(args.datasets_dir)
        train_data = np.concatenate((arxiv_data["train"], wiki_data["train"]))
        val_data = np.concatenate((arxiv_data["val"], wiki_data["val"]))
        return {"train": train_data, "val": val_data}
    if args.dataset == "openwebtext2":
        return get_openwebtext2_data(args.datasets_dir)
    if args.dataset == "redpajama":
        return get_redpajama_data(args.datasets_dir)
    if args.dataset == "redpajamav2":
        return get_redpajamav2_data(args.datasets_dir)
    if args.dataset == "slimpajama":
        return get_slimpajama_data(args.datasets_dir)
    if args.dataset == "fineweb":
        return get_fineweb_data(args.datasets_dir)
    if args.dataset == "fineweb10b":
        return get_fineweb10b_data(args.datasets_dir)
    if args.dataset == "finewebedu":
        return get_fineweb_edu_data(args.datasets_dir)
    if args.dataset == "c4":
        return get_c4_data(args.datasets_dir)
    if args.dataset in SUPPORTED_TASK_MAP:
        return get_benchmark_task(args.dataset)
    else:
        raise NotImplementedError(f"Unknow dataset key '{args.dataset}'")


def get_benchmark_task(name, **kwargs):
    """Fetch the right benchmark task given by the name parameter. The logic for each task is
    contained in its own python file.
    """
    try:
        fn = SUPPORTED_TASK_MAP[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: {sorted(SUPPORTED_TASK_MAP.keys())}"
        )
    return fn(**kwargs)


class DataReader:
    def __init__(
        self,
        data_src,
        batch_size,
        sequence_length,
        seed=1337,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=False,
    ):
        self.data_path = None
        self.data_paths = None
        self.data_spec = None

        if isinstance(data_src, (str, Path, dict)):
            self.data_spec = _normalize_memmap_source(data_src)
            self.data_path = self.data_spec["path"]
            self.data_paths = None
            self.keep_in_ram = keep_in_ram
            if keep_in_ram:
                self.data = np.array(
                    np.memmap(
                        self.data_spec["path"],
                        dtype=self.data_spec["dtype"],
                        mode="r",
                        offset=self.data_spec["offset_bytes"],
                        shape=(self.data_spec["length"],),
                    )
                )
            else:
                self.data = None
        elif isinstance(data_src, (list, tuple)):
            self.data_path = None
            self.data_paths = [_normalize_memmap_source(path) for path in data_src]
            self.keep_in_ram = keep_in_ram
            if keep_in_ram:
                raise ValueError(
                    "keep_in_ram=True is not supported for sharded datasets."
                )
            self.data = ShardedMemmapDataset(self.data_paths)
        elif isinstance(data_src, (np.ndarray, np.memmap)):
            self.data_path = None
            self.data_paths = None
            self.data = data_src
            self.keep_in_ram = True

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.seed = seed
        self.with_replacement = with_replacement

        self.num_tokens = len(self._get_data())

        if auto_shard and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            print(
                f"Distributed DataReader Initialized for Worker {self.rank}/{self.world_size}"
            )
        else:
            self.world_size = 1
            self.rank = 0

        # Sampling without replacement
        self.last_epoch = None
        self.order = None
        self.epoch_offset = None
        self.step = 0
        self.num_batches_of_seqlen = 0
        if not with_replacement:
            self._shuffle_epoch(0)

    def __len__(self):
        # Length in valid start indices for a sequence
        # Extra -1 to have a valid next token for the final token of the last idx
        return self.num_tokens - self.sequence_length - 1

    def _get_data(self):
        if self.data is not None:
            return self.data
        else:
            # Construct the memmap each time to avoid a memory leak per NanoGPT
            # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
            return np.memmap(
                self.data_spec["path"],
                dtype=self.data_spec["dtype"],
                mode="r",
                offset=self.data_spec["offset_bytes"],
                shape=(self.data_spec["length"],),
            )

    def __getitem__(self, idx):
        # Return the underlying datapoint, no random sampling, no worker sharding
        assert 0 <= idx < len(self)
        data = self._get_data()
        x = torch.from_numpy(data[idx : idx + self.sequence_length].astype(np.int64))
        y = torch.from_numpy(
            data[idx + 1 : idx + self.sequence_length + 1].astype(torch.int64)
        )
        return x, y

    def set_step(self, step):
        self.step = step

    def sample_batch(self):
        data = self._get_data()

        if self.with_replacement:
            idxs = self._sample_with_replacement(self.step)
        else:
            idxs = self._sample_without_replacement(self.step)
        self.step += 1

        xy = np.stack([data[i : i + self.sequence_length + 1] for i in idxs]).astype(
            np.int64
        )
        x = torch.from_numpy(xy[:, :-1]).contiguous()
        y = torch.from_numpy(xy[:, 1:]).contiguous()
        return x, y

    def _sample_with_replacement(self, idx):
        # Return an array of token indices of length self.batch_size
        # Sampled with replacement, can get repeats at any time
        seed = self.seed + idx * self.world_size + self.rank
        rng = np.random.default_rng(seed)
        return rng.integers(len(self), self.batch_size)

    def _shuffle_epoch(self, epoch):
        seed = self.seed + epoch
        rng = np.random.default_rng(seed)
        # Drop one sequence to allow different offsets per epoch:
        self.order = rng.permutation((len(self)) // self.sequence_length - 1)
        # Shift all sequences in this epoch by this amount:
        self.epoch_offset = rng.integers(self.sequence_length)
        self.last_epoch = epoch
        self.num_batches_of_seqlen = (
            len(self.order) // self.batch_size
        )  # Drops remainder batch

    def _sample_without_replacement(self, step):
        # Return an array of token indices of length self.batch_size
        # Sampled without replacement, cycle all sequences before potential repeats
        # Sequences are randomly offset in every epoch as well
        batch_idx = self.world_size * step + self.rank
        epoch_length = self.num_batches_of_seqlen

        epoch = batch_idx // epoch_length
        if epoch != self.last_epoch:
            self._shuffle_epoch(epoch)
        epoch_idx = batch_idx % epoch_length

        start = epoch_idx * self.batch_size
        end = start + self.batch_size
        return self.order[start:end] * self.sequence_length + self.epoch_offset

    def num_batches(self):
        if self.with_replacement:
            return self.num_tokens // self.batch_size
        return self.num_batches_of_seqlen
