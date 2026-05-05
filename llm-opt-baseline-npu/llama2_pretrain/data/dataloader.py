import itertools

import torch
from torch.utils.data import IterableDataset, get_worker_info


class PreprocessedIterableDataset(IterableDataset):
    def __init__(
        self,
        data,
        tokenizer,
        batch_size,
        max_length,
        tokenizer_batch_size=None,
        text_buffer_size=None,
        drop_last=True,
    ):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.drop_last = drop_last
        self.tokens_per_batch = self.batch_size * self.max_length
        self.tokenizer_batch_size = tokenizer_batch_size or max(self.batch_size * 8, 32)
        self.text_buffer_size = text_buffer_size or self.tokenizer_batch_size
        self._resume_state = None
        self._runtime_state = None
        self._stateful_iteration_enabled = False

    def __iter__(self):
        iter_data = self._get_worker_iterator()
        worker_info = get_worker_info()
        if worker_info is not None:
            if self._resume_state is not None:
                raise RuntimeError(
                    "PreprocessedIterableDataset resume state requires num_workers=0. "
                    "Disable dataloader workers when using save_steps/continue_from."
                )
            self._stateful_iteration_enabled = False
            raw_examples_seen = 0
            token_buffer = []
            token_offset = 0
            batches_to_skip = 0
        else:
            self._stateful_iteration_enabled = True
            state = self._resume_state or {
                "raw_examples_seen": 0,
                "token_buffer": [],
                "token_offset": 0,
                "batches_to_skip": 0,
            }
            raw_examples_seen = int(state["raw_examples_seen"])
            token_buffer = list(state["token_buffer"])
            token_offset = int(state["token_offset"])
            batches_to_skip = int(state.get("batches_to_skip", 0))
            if raw_examples_seen > 0:
                iter_data = itertools.islice(iter_data, raw_examples_seen, None)
            self._update_runtime_state(raw_examples_seen, token_buffer, token_offset, batches_to_skip)

        text_buffer = []
        reached_end = False

        while True:
            while len(token_buffer) - token_offset < self.tokens_per_batch and not reached_end:
                while len(text_buffer) < self.text_buffer_size and not reached_end:
                    try:
                        example = next(iter_data)
                    except StopIteration:
                        reached_end = True
                        break

                    raw_examples_seen += 1
                    text = example.get("text", "")
                    if text:
                        text_buffer.append(text)

                if text_buffer:
                    self._extend_token_buffer(text_buffer, token_buffer)
                    text_buffer.clear()
                    self._update_runtime_state(raw_examples_seen, token_buffer, token_offset, batches_to_skip)

            while len(token_buffer) - token_offset >= self.tokens_per_batch:
                end = token_offset + self.tokens_per_batch
                flat_tokens = torch.tensor(token_buffer[token_offset:end], dtype=torch.long)
                token_offset = end

                if token_offset >= self.tokens_per_batch * 4:
                    del token_buffer[:token_offset]
                    token_offset = 0

                self._update_runtime_state(raw_examples_seen, token_buffer, token_offset, batches_to_skip)

                if batches_to_skip > 0:
                    batches_to_skip -= 1
                    self._update_runtime_state(raw_examples_seen, token_buffer, token_offset, batches_to_skip)
                    continue

                yield self._format_batch(flat_tokens, self.batch_size)

            if reached_end:
                break

        if not self.drop_last:
            remaining_tokens = len(token_buffer) - token_offset
            remaining_sequences = remaining_tokens // self.max_length
            if remaining_sequences > 0:
                end = token_offset + remaining_sequences * self.max_length
                flat_tokens = torch.tensor(token_buffer[token_offset:end], dtype=torch.long)
                token_offset = end
                if token_offset >= len(token_buffer):
                    token_buffer = []
                    token_offset = 0
                self._update_runtime_state(raw_examples_seen, token_buffer, token_offset, batches_to_skip)
                if batches_to_skip > 0:
                    batches_to_skip -= 1
                    self._update_runtime_state(raw_examples_seen, token_buffer, token_offset, batches_to_skip)
                else:
                    yield self._format_batch(flat_tokens, remaining_sequences)

        self._update_runtime_state(raw_examples_seen, token_buffer, token_offset, batches_to_skip)
        if batches_to_skip > 0:
            raise RuntimeError(f"Unable to skip {batches_to_skip} more batches while restoring dataset state")

    def _get_worker_iterator(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return iter(self.data)

        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        if hasattr(self.data, "shard"):
            return iter(self.data.shard(num_workers, worker_id))
        return itertools.islice(self.data, worker_id, None, num_workers)

    def _extend_token_buffer(self, texts, token_buffer):
        for start in range(0, len(texts), self.tokenizer_batch_size):
            batch_texts = texts[start:start + self.tokenizer_batch_size]
            encoded_batch = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                add_special_tokens=True,
                return_attention_mask=False,
            )

            for input_ids in encoded_batch["input_ids"]:
                if not input_ids:
                    continue
                token_buffer.extend(input_ids)

    def _format_batch(self, flat_tokens, batch_size):
        input_ids = flat_tokens.view(batch_size, self.max_length)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _update_runtime_state(self, raw_examples_seen, token_buffer, token_offset, batches_to_skip):
        if not self._stateful_iteration_enabled:
            return
        self._runtime_state = {
            "raw_examples_seen": int(raw_examples_seen),
            "token_buffer": token_buffer,
            "token_offset": int(token_offset),
            "batches_to_skip": int(batches_to_skip),
        }

    def state_dict(self):
        if self._runtime_state is None:
            state = self._resume_state or {
                "raw_examples_seen": 0,
                "token_buffer": [],
                "token_offset": 0,
                "batches_to_skip": 0,
            }
        else:
            state = self._runtime_state

        token_buffer = list(state["token_buffer"][state["token_offset"]:])
        return {
            "raw_examples_seen": int(state["raw_examples_seen"]),
            "token_buffer": token_buffer,
            "token_offset": 0,
            "batches_to_skip": int(state.get("batches_to_skip", 0)),
        }

    def load_state_dict(self, state):
        state = state or {}
        token_buffer = list(state.get("token_buffer", []))
        token_offset = int(state.get("token_offset", 0))
        raw_examples_seen = int(state.get("raw_examples_seen", 0))
        batches_to_skip = int(state.get("batches_to_skip", 0))

        if raw_examples_seen < 0:
            raise ValueError("raw_examples_seen must be >= 0")
        if batches_to_skip < 0:
            raise ValueError("batches_to_skip must be >= 0")
        if token_offset < 0 or token_offset > len(token_buffer):
            raise ValueError("token_offset is out of range for token_buffer")

        self._resume_state = {
            "raw_examples_seen": raw_examples_seen,
            "token_buffer": token_buffer,
            "token_offset": token_offset,
            "batches_to_skip": batches_to_skip,
        }
        self._runtime_state = None
