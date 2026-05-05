#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import os
import glob
from typing import Iterable, List, Optional

import datasets
import datasets.distributed
import torch
from transformers import AutoTokenizer

from data.dataloader import PreprocessedIterableDataset


TRAIN_DATA_GLOB = "c4-train.*.json.gz"
TRAIN_SHUFFLE_SEED = 2026


def _normalize_path(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _sorted_data_files(data_dir: str, pattern: str) -> List[str]:
    search_root = _normalize_path(data_dir)
    data_files = sorted(
        _normalize_path(path)
        for path in glob.glob(os.path.join(search_root, pattern))
    )
    if not data_files:
        raise FileNotFoundError(f"No data files matched {pattern} under {search_root}")
    return data_files


def _parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    values = [int(part) for part in parts if part]
    return values or None


def _iter_selected_files(data_files: List[str], max_files: Optional[int]) -> Iterable[str]:
    if max_files is None:
        yield from data_files
        return
    for path in data_files[:max_files]:
        yield path


def _is_text_suspicious(text: str) -> bool:
    if not text:
        return True
    if "\x00" in text:
        return True
    return False


def _raw_scan(args) -> int:
    data_files = _sorted_data_files(args.data_dir, args.data_glob)
    stats = {
        "files_seen": 0,
        "records_seen": 0,
        "json_errors": 0,
        "non_dict_records": 0,
        "missing_text": 0,
        "non_string_text": 0,
        "empty_text": 0,
        "suspicious_text": 0,
    }
    examples = []

    for file_path in _iter_selected_files(data_files, args.max_files):
        stats["files_seen"] += 1
        try:
            with gzip.open(file_path, "rt", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if args.max_records is not None and stats["records_seen"] >= args.max_records:
                        break
                    stats["records_seen"] += 1
                    try:
                        record = json.loads(line)
                    except Exception as exc:
                        stats["json_errors"] += 1
                        if len(examples) < args.sample_limit:
                            examples.append(
                                {
                                    "kind": "json_error",
                                    "file": file_path,
                                    "line": line_no,
                                    "error": repr(exc),
                                }
                            )
                        continue

                    if not isinstance(record, dict):
                        stats["non_dict_records"] += 1
                        if len(examples) < args.sample_limit:
                            examples.append(
                                {
                                    "kind": "non_dict_record",
                                    "file": file_path,
                                    "line": line_no,
                                    "type": type(record).__name__,
                                }
                            )
                        continue

                    text = record.get("text")
                    if text is None:
                        stats["missing_text"] += 1
                        if len(examples) < args.sample_limit:
                            examples.append(
                                {
                                    "kind": "missing_text",
                                    "file": file_path,
                                    "line": line_no,
                                }
                            )
                        continue

                    if not isinstance(text, str):
                        stats["non_string_text"] += 1
                        if len(examples) < args.sample_limit:
                            examples.append(
                                {
                                    "kind": "non_string_text",
                                    "file": file_path,
                                    "line": line_no,
                                    "type": type(text).__name__,
                                }
                            )
                        continue

                    if text == "":
                        stats["empty_text"] += 1
                        if len(examples) < args.sample_limit:
                            examples.append(
                                {
                                    "kind": "empty_text",
                                    "file": file_path,
                                    "line": line_no,
                                }
                            )

                    if _is_text_suspicious(text):
                        stats["suspicious_text"] += 1
                        if len(examples) < args.sample_limit:
                            examples.append(
                                {
                                    "kind": "suspicious_text",
                                    "file": file_path,
                                    "line": line_no,
                                    "preview": text[:120],
                                }
                            )

                if args.max_records is not None and stats["records_seen"] >= args.max_records:
                    break
        except Exception as exc:
            stats["json_errors"] += 1
            if len(examples) < args.sample_limit:
                examples.append(
                    {
                        "kind": "file_read_error",
                        "file": file_path,
                        "error": repr(exc),
                    }
                )

    print(json.dumps({"mode": "raw", "stats": stats, "examples": examples}, ensure_ascii=False, indent=2))
    return 0 if stats["json_errors"] == 0 else 1


def _summarize_batch(batch, pad_idx: Optional[int]):
    input_ids = batch["input_ids"].detach().reshape(-1).contiguous()
    attention_mask = batch["attention_mask"].detach().reshape(-1).contiguous()
    digest = hashlib.sha1(input_ids.numpy().tobytes()).hexdigest()[:16]
    unique_tokens = int(input_ids.unique().numel())
    if pad_idx is not None and pad_idx >= 0:
        non_pad_tokens = int((input_ids != pad_idx).sum().item())
    else:
        non_pad_tokens = int(input_ids.numel())
    return {
        "sha1": digest,
        "shape": tuple(batch["input_ids"].shape),
        "token_min": int(input_ids.min().item()),
        "token_max": int(input_ids.max().item()),
        "token_sum": int(input_ids.sum().item()),
        "attention_sum": int(attention_mask.sum().item()),
        "non_pad_tokens": non_pad_tokens,
        "unique_tokens": unique_tokens,
        "first_tokens": input_ids[: min(8, input_ids.numel())].tolist(),
    }


def _dump_batch(dump_dir: Optional[str], rank: int, batch_idx: int, summary: dict, batch: dict, state: dict):
    if not dump_dir:
        return None
    os.makedirs(dump_dir, exist_ok=True)
    payload = {
        "rank": int(rank),
        "batch_idx": int(batch_idx),
        "batch_summary": dict(summary),
        "dataset_state": dict(state),
        "batch": {
            key: value.detach().cpu().clone()
            for key, value in batch.items()
        },
    }
    output_path = os.path.join(
        dump_dir,
        f"rank{rank:04d}_batch{batch_idx:06d}_{summary['sha1']}.pt",
    )
    torch.save(payload, output_path)
    return output_path


def _build_rank_dataset(args, tokenizer, rank: int):
    data_files = _sorted_data_files(args.data_dir, args.data_glob)
    data = datasets.load_dataset(
        "json",
        data_files=data_files,
        split="train",
        streaming=True,
    )
    data = data.shuffle(seed=args.shuffle_seed)
    if args.world_size > 1:
        data = datasets.distributed.split_dataset_by_node(
            data,
            rank=rank,
            world_size=args.world_size,
        )
    dataset = PreprocessedIterableDataset(
        data,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        tokenizer_batch_size=args.tokenizer_batch_size,
        text_buffer_size=args.text_buffer_size,
    )
    return dataset


def _packed_scan(args) -> int:
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        model_max_length=args.max_length,
        use_fast=True,
    )
    pad_idx = tokenizer.pad_token_id
    vocab_size = int(args.vocab_size) if args.vocab_size is not None else len(tokenizer)
    ranks = _parse_int_list(args.ranks)
    if ranks is None:
        ranks = list(range(args.world_size))

    results = []
    bad_batches = 0

    for rank in ranks:
        dataset = _build_rank_dataset(args, tokenizer, rank)
        iterator = iter(dataset)
        for batch_idx, batch in enumerate(iterator, start=1):
            if batch_idx < args.start_batch_idx:
                continue
            if batch_idx >= args.start_batch_idx + args.num_batches:
                break

            summary = _summarize_batch(batch, pad_idx)
            valid_token_range = 0 <= summary["token_min"] and summary["token_max"] < vocab_size
            state = dataset.state_dict()
            record = {
                "mode": "packed",
                "rank": int(rank),
                "batch_idx": int(batch_idx),
                "valid_token_range": bool(valid_token_range),
                "raw_examples_seen": int(state["raw_examples_seen"]),
                "remaining_token_buffer": int(len(state["token_buffer"])),
                "summary": summary,
            }
            dump_path = _dump_batch(args.dump_dir, rank, batch_idx, summary, batch, state)
            if dump_path is not None:
                record["dump_path"] = dump_path

            if not valid_token_range:
                bad_batches += 1

            results.append(record)
            print(json.dumps(record, ensure_ascii=False))

    final_summary = {
        "mode": "packed_summary",
        "world_size": int(args.world_size),
        "ranks": ranks,
        "start_batch_idx": int(args.start_batch_idx),
        "num_batches": int(args.num_batches),
        "captured_batches": len(results),
        "bad_batches": int(bad_batches),
    }
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    return 0 if bad_batches == 0 else 1


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Validate raw C4 files and inspect the packed training stream used by pretrain_c4_dist.py"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    raw_parser = subparsers.add_parser("raw", help="Scan raw json.gz files for corruption or malformed records.")
    raw_parser.add_argument("--data_dir", required=True, type=str)
    raw_parser.add_argument("--data_glob", default=TRAIN_DATA_GLOB, type=str)
    raw_parser.add_argument("--max_files", type=int, default=None)
    raw_parser.add_argument("--max_records", type=int, default=None)
    raw_parser.add_argument("--sample_limit", type=int, default=20)

    packed_parser = subparsers.add_parser(
        "packed",
        help="Rebuild the packed token stream with the same shuffle/split/tokenize/pack pipeline and inspect target microbatches.",
    )
    packed_parser.add_argument("--data_dir", required=True, type=str)
    packed_parser.add_argument("--tokenizer_path", required=True, type=str)
    packed_parser.add_argument("--data_glob", default=TRAIN_DATA_GLOB, type=str)
    packed_parser.add_argument("--shuffle_seed", default=TRAIN_SHUFFLE_SEED, type=int)
    packed_parser.add_argument("--world_size", default=1, type=int)
    packed_parser.add_argument("--ranks", default=None, type=str,
                               help="Comma-separated rank list. Default: all ranks in [0, world_size).")
    packed_parser.add_argument("--batch_size", required=True, type=int,
                               help="Per-rank micro batch size used in training.")
    packed_parser.add_argument("--max_length", default=4096, type=int)
    packed_parser.add_argument("--vocab_size", default=None, type=int,
                               help="Optional model vocab size. When unset, uses len(tokenizer).")
    packed_parser.add_argument("--tokenizer_batch_size", default=64, type=int)
    packed_parser.add_argument("--text_buffer_size", default=256, type=int)
    packed_parser.add_argument("--start_batch_idx", required=True, type=int,
                               help="1-based per-rank microbatch index, equal to training global_step for that rank stream.")
    packed_parser.add_argument("--num_batches", default=1, type=int)
    packed_parser.add_argument("--dump_dir", default=None, type=str)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.mode == "raw":
        raise SystemExit(_raw_scan(args))
    if args.mode == "packed":
        raise SystemExit(_packed_scan(args))
    raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
