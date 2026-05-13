from pathlib import Path

import torch

from catspec.spec_encoder import SpecEncoder, batch_encode_specs, spec_to_token_ids


def test_spec_token_ids_are_stable_and_distinguish_specs():
    square = Path("specs/categories/square_tube.yaml")
    bellmouth = Path("specs/categories/bellmouth.yaml")

    first = spec_to_token_ids(square)
    second = spec_to_token_ids(square)
    other = spec_to_token_ids(bellmouth)

    assert first == second
    assert first != other
    assert len(first) > 4


def test_batch_encode_specs_pads_and_masks():
    batch = batch_encode_specs(
        [
            Path("specs/categories/square_tube.yaml"),
            Path("specs/categories/bellmouth.yaml"),
        ],
        max_tokens=32,
    )

    assert batch["token_ids"].shape == (2, 32)
    assert batch["token_mask"].shape == (2, 32)
    assert batch["token_ids"].dtype == torch.long
    assert batch["token_mask"].dtype == torch.bool
    assert batch["token_mask"].sum(dim=1).min().item() > 4
    assert torch.all(batch["token_ids"][~batch["token_mask"]] == 0)


def test_spec_encoder_is_deterministic_and_distinguishes_specs():
    encoder = SpecEncoder(embedding_dim=24, max_tokens=32)
    batch = batch_encode_specs(
        [
            Path("specs/categories/square_tube.yaml"),
            Path("specs/categories/bellmouth.yaml"),
        ],
        max_tokens=32,
    )

    emb1 = encoder(batch["token_ids"], batch["token_mask"])
    emb2 = encoder(batch["token_ids"], batch["token_mask"])

    assert emb1.shape == (2, 24)
    assert torch.allclose(emb1, emb2)
    assert not torch.allclose(emb1[0], emb1[1])


def test_spec_encoder_supports_no_spec_mode():
    batch = batch_encode_specs([None, None], max_tokens=8)
    encoder = SpecEncoder(embedding_dim=16, max_tokens=8)

    emb = encoder(batch["token_ids"], batch["token_mask"])

    assert batch["token_mask"].sum(dim=1).tolist() == [1, 1]
    assert emb.shape == (2, 16)
    assert torch.allclose(emb[0], emb[1])
