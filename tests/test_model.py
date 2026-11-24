from pathlib import Path

import pytest
import torch

from drumgen.config import ModelSettings, PieceType, Style
from drumgen.model import DrumTransformer, ModelCheckpoint

VOCAB = 158
PAD = 0
SMALL = ModelSettings(d_model=32, layers=2, heads=2, feedforward=64, dropout=0.0)


def small_model(context_length: int = 64) -> DrumTransformer:
    torch.manual_seed(0)
    return DrumTransformer(SMALL, vocab_size=VOCAB, context_length=context_length, pad_id=PAD).eval()


def conditions(length: int, style: Style = Style.ROCK, rows: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    styles = torch.full((rows, length), style.index, dtype=torch.long)
    types = torch.full((rows, length), PieceType.GROOVE.index, dtype=torch.long)
    return styles, types


def test_logits_cover_the_vocabulary_at_every_position() -> None:
    model = small_model()
    tokens = torch.randint(1, VOCAB, (3, 20))

    logits = model(tokens, *conditions(20, rows=3))

    assert logits.shape == (3, 20, VOCAB)


def test_changing_a_future_token_does_not_change_past_logits() -> None:
    model = small_model()
    tokens = torch.randint(1, VOCAB, (1, 30))
    changed = tokens.clone()
    changed[0, 20:] = (changed[0, 20:] + 7) % VOCAB

    with torch.no_grad():
        before = model(tokens, *conditions(30))
        after = model(changed, *conditions(30))

    assert torch.allclose(before[0, :20], after[0, :20], atol=1e-6)
    assert not torch.allclose(before[0, 20:], after[0, 20:])


def test_style_reaches_the_logits() -> None:
    model = small_model()
    tokens = torch.randint(1, VOCAB, (1, 10))

    with torch.no_grad():
        rock = model(tokens, *conditions(10, Style.ROCK))
        jazz = model(tokens, *conditions(10, Style.JAZZ))

    assert not torch.allclose(rock, jazz)


def test_sequence_longer_than_the_context_is_refused() -> None:
    model = small_model(context_length=16)

    with pytest.raises(ValueError, match="16"):
        model(torch.ones((1, 17), dtype=torch.long), *conditions(17))


def test_p1_model_has_between_three_and_four_million_parameters() -> None:
    model = DrumTransformer(ModelSettings(), vocab_size=VOCAB, context_length=768, pad_id=PAD)

    assert 3_000_000 <= model.parameter_count <= 4_000_000


def test_checkpoint_round_trip_restores_the_same_logits(tmp_path: Path) -> None:
    model = small_model()
    tokens = torch.randint(1, VOCAB, (2, 12))
    path = tmp_path / "best.pt"

    ModelCheckpoint.of(model, tokenizer_file=Path("data/processed/tokenizer.json"), epoch=3).save(path)
    checkpoint = ModelCheckpoint.load(path)
    restored = checkpoint.build_model().eval()

    with torch.no_grad():
        assert torch.equal(model(tokens, *conditions(12, rows=2)), restored(tokens, *conditions(12, rows=2)))
    assert checkpoint.styles == [style.value for style in Style]
    assert checkpoint.epoch == 3
    assert checkpoint.tokenizer_file == Path("data/processed/tokenizer.json")


def test_cached_decoding_reproduces_the_full_forward() -> None:
    model = small_model()
    tokens = torch.randint(1, VOCAB, (3, 25))
    styles, types = conditions(25, rows=3)
    styles[1, 10:] = Style.FUNK.index

    with torch.no_grad():
        full = model(tokens, styles, types)
        cache = model.empty_cache(rows=3, device=torch.device("cpu"))
        chunks = [model.forward_cached(tokens[:, :5], styles[:, :5], types[:, :5], cache)]
        for position in range(5, 25):
            step = slice(position, position + 1)
            chunks.append(model.forward_cached(tokens[:, step], styles[:, step], types[:, step], cache))

    assert torch.allclose(torch.cat(chunks, dim=1), full, atol=1e-5)
    assert cache.length == 25


def test_cache_refuses_to_grow_past_the_context() -> None:
    model = small_model(context_length=8)
    cache = model.empty_cache(rows=1, device=torch.device("cpu"))

    with torch.no_grad():
        model.forward_cached(torch.ones((1, 8), dtype=torch.long), *conditions(8), cache)
        with pytest.raises(ValueError, match="8"):
            model.forward_cached(torch.ones((1, 1), dtype=torch.long), *conditions(1), cache)
