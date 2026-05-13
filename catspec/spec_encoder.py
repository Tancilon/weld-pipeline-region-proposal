"""Deterministic CatSpec tokenization and embedding for the mini PoC."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from catspec.schema import load_catspec


PAD_ID = 0
NO_SPEC_ID = 1
DEFAULT_VOCAB_SIZE = 4096


def _stable_token_id(token: str, vocab_size: int = DEFAULT_VOCAB_SIZE) -> int:
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return 2 + (int(digest[:12], 16) % (vocab_size - 2))


def _tokenize_value(prefix: str, value: Any) -> list[str]:
    if isinstance(value, dict):
        tokens: list[str] = []
        for key in sorted(value):
            tokens.extend(_tokenize_value(f"{prefix}.{key}" if prefix else str(key), value[key]))
        return tokens
    if isinstance(value, list):
        tokens = [f"{prefix}.len:{len(value)}"]
        for idx, item in enumerate(value):
            tokens.extend(_tokenize_value(f"{prefix}[{idx}]", item))
        return tokens
    return [f"{prefix}:{value}"]


def _spec_dict_from_source(source: str | Path | dict[str, Any] | None) -> dict[str, Any] | None:
    if source is None:
        return None
    if isinstance(source, dict):
        return source
    return load_catspec(source)


def spec_to_tokens(source: str | Path | dict[str, Any] | None) -> list[str]:
    """Convert a CatSpec YAML or auto-GT row into stable textual tokens."""

    spec = _spec_dict_from_source(source)
    if spec is None:
        return ["<no_spec>"]

    tokens: list[str] = [f"category:{spec.get('category')}"]
    if "parts" in spec:
        tokens.extend(_tokenize_value("parts", spec["parts"]))
    if "welds" in spec:
        tokens.extend(_tokenize_value("welds", spec["welds"]))
    if "generated_locus" in spec:
        tokens.extend(_tokenize_value("generated_locus", spec["generated_locus"]))
    if "weld_meta" in spec:
        tokens.extend(_tokenize_value("weld_meta", spec["weld_meta"]))
    if "topology" in spec:
        tokens.extend(_tokenize_value("topology", spec["topology"]))
    return tokens


def spec_to_token_ids(
    source: str | Path | dict[str, Any] | None,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> list[int]:
    """Convert a CatSpec source to stable token ids."""

    tokens = spec_to_tokens(source)
    if tokens == ["<no_spec>"]:
        return [NO_SPEC_ID]
    return [_stable_token_id(token, vocab_size=vocab_size) for token in tokens]


def batch_encode_specs(
    sources: list[str | Path | dict[str, Any] | None],
    max_tokens: int = 96,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> dict[str, torch.Tensor]:
    """Pad CatSpec token ids and masks into a batch."""

    encoded = [spec_to_token_ids(source, vocab_size=vocab_size)[:max_tokens] for source in sources]
    token_ids = torch.zeros((len(encoded), max_tokens), dtype=torch.long)
    token_mask = torch.zeros((len(encoded), max_tokens), dtype=torch.bool)
    for row, ids in enumerate(encoded):
        length = min(len(ids), max_tokens)
        token_ids[row, :length] = torch.tensor(ids[:length], dtype=torch.long)
        token_mask[row, :length] = True
    return {"token_ids": token_ids, "token_mask": token_mask}


def _deterministic_embedding_table(vocab_size: int, embedding_dim: int) -> torch.Tensor:
    ids = torch.arange(vocab_size, dtype=torch.float32).unsqueeze(1)
    dims = torch.arange(embedding_dim, dtype=torch.float32).unsqueeze(0) + 1.0
    table = torch.sin(ids * dims * 0.017) + torch.cos(ids * dims * 0.013)
    table[PAD_ID].zero_()
    return table


class SpecEncoder(nn.Module):
    """Frozen deterministic embedding table with masked mean pooling."""

    def __init__(
        self,
        embedding_dim: int = 64,
        max_tokens: int = 96,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding.from_pretrained(
            _deterministic_embedding_table(vocab_size, embedding_dim),
            freeze=True,
            padding_idx=PAD_ID,
        )

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        mask = token_mask.to(dtype=embedded.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (embedded * mask).sum(dim=1) / denom
