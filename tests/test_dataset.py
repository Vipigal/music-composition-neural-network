import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from symusic.core import ScoreTick

from drumgen.config import DataSettings, PieceType, Style, TokenizerSettings
from drumgen.dataset import Corpus, EvaluationWindows, TrainingWindows, WindowBatch
from drumgen.tokenizer import DrumTokenizer

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DRAWS = 10_000


@pytest.fixture(scope="module")
def tokenizer() -> DrumTokenizer:
    return DrumTokenizer.from_settings(TokenizerSettings())


@pytest.fixture(scope="module")
def corpus(tokenizer: DrumTokenizer) -> Corpus:
    entries = [
        ("funk_117_beat.mid", Style.ROCK, PieceType.GROOVE),
        ("jazz_120_beat.mid", Style.JAZZ, PieceType.GROOVE),
        ("rock_115_fill.mid", Style.SOUL, PieceType.FILL),
    ]
    sequences = []
    for fixture, _, _ in entries:
        ids = tokenizer.encode(ScoreTick(FIXTURES / fixture))
        sequences.append(np.array([tokenizer.bos_id, *ids, tokenizer.eos_id], dtype=np.int32))
    return Corpus(
        sequences=sequences,
        bar_starts=[np.array(tokenizer.bar_starts(list(sequence)), dtype=np.int32) for sequence in sequences],
        styles=[style for _, style, _ in entries],
        types=[piece_type for _, _, piece_type in entries],
    )


def windows(corpus: Corpus, tokenizer: DrumTokenizer, **overrides: float) -> TrainingWindows:
    settings = DataSettings(context_length=256, **overrides)
    return TrainingWindows(corpus, tokenizer, settings, seed=7)


def test_every_window_starts_at_bos_or_bar(corpus: Corpus, tokenizer: DrumTokenizer) -> None:
    dataset = windows(corpus, tokenizer)

    first_tokens = {int(dataset[index].tokens[0]) for index in range(500)}

    assert first_tokens <= {tokenizer.bos_id, tokenizer.bar_id}
    assert tokenizer.bar_id in first_tokens


def test_window_targets_are_the_inputs_shifted_by_one(corpus: Corpus, tokenizer: DrumTokenizer) -> None:
    window = windows(corpus, tokenizer, velocity_scale_min=1.0, velocity_scale_max=1.0)[3]

    assert window.tokens.shape == window.targets.shape
    assert len(window.tokens) <= 256
    assert window.tokens[1:].tolist() == window.targets[:-1].tolist()


def test_style_frequency_follows_inverse_square_root_weights(corpus: Corpus, tokenizer: DrumTokenizer) -> None:
    dataset = windows(corpus, tokenizer, null_style_probability=0.0)
    bars = {style: int(len(starts)) for style, starts in zip(corpus.styles, corpus.bar_starts, strict=True)}
    total = sum(math.sqrt(count) for count in bars.values())

    drawn = Counter(list(Style)[int(dataset[index].styles[0])] for index in range(DRAWS))

    for style, count in bars.items():
        assert drawn[style] / DRAWS == pytest.approx(math.sqrt(count) / total, abs=0.02)


def test_null_style_replaces_twenty_percent_of_windows(corpus: Corpus, tokenizer: DrumTokenizer) -> None:
    dataset = windows(corpus, tokenizer)

    nulls = sum(int(dataset[index].styles[0]) == Style.NULL.index for index in range(DRAWS))

    assert nulls / DRAWS == pytest.approx(0.2, abs=0.02)


def test_style_and_type_repeat_on_every_position(corpus: Corpus, tokenizer: DrumTokenizer) -> None:
    dataset = windows(corpus, tokenizer)

    for index in range(50):
        window = dataset[index]
        assert len(set(window.styles.tolist())) == 1
        assert len(set(window.types.tolist())) == 1
        assert len(window.styles) == len(window.tokens)


def test_type_is_never_dropped(corpus: Corpus, tokenizer: DrumTokenizer) -> None:
    dataset = windows(corpus, tokenizer, null_style_probability=1.0)

    types = {(int(dataset[index].types[0])) for index in range(2000)}

    assert types == {PieceType.GROOVE.index, PieceType.FILL.index}


def test_velocity_scaling_only_changes_velocity_tokens(corpus: Corpus, tokenizer: DrumTokenizer) -> None:
    plain = windows(corpus, tokenizer, velocity_scale_min=1.0, velocity_scale_max=1.0)
    scaled = windows(corpus, tokenizer)

    changed = 0
    for index in range(50):
        for before, after in zip(plain[index].tokens.tolist(), scaled[index].tokens.tolist(), strict=True):
            if before != after:
                changed += 1
                assert tokenizer.token_of(before).startswith("Velocity_")
    assert changed > 0


def test_same_seed_and_epoch_give_the_same_windows(corpus: Corpus, tokenizer: DrumTokenizer) -> None:
    first, second = windows(corpus, tokenizer), windows(corpus, tokenizer)

    assert first[5].tokens.tolist() == second[5].tokens.tolist()
    second.set_epoch(1)
    assert any(first[index].tokens.tolist() != second[index].tokens.tolist() for index in range(10))


def test_collate_pads_with_pad_and_keeps_shapes(corpus: Corpus, tokenizer: DrumTokenizer) -> None:
    dataset = windows(corpus, tokenizer)
    items = [dataset[index] for index in range(8)]

    batch = WindowBatch.collate(items, tokenizer.pad_id)

    longest = max(len(item.tokens) for item in items)
    assert batch.tokens.shape == batch.targets.shape == batch.styles.shape == batch.types.shape == (8, longest)
    shortest = min(range(8), key=lambda index: len(items[index].tokens))
    assert batch.targets[shortest, len(items[shortest].tokens) :].eq(tokenizer.pad_id).all()


def test_evaluation_windows_score_every_token_once(corpus: Corpus, tokenizer: DrumTokenizer) -> None:
    dataset = EvaluationWindows(corpus, tokenizer, DataSettings(context_length=256))

    scored = sum(int(dataset[index].targets.ne(tokenizer.pad_id).sum()) for index in range(len(dataset)))

    assert scored == sum(len(sequence) - 1 for sequence in corpus.sequences)


def test_evaluation_windows_keep_the_true_style_and_start_at_bars(corpus: Corpus, tokenizer: DrumTokenizer) -> None:
    dataset = EvaluationWindows(corpus, tokenizer, DataSettings(context_length=256))

    for index in range(len(dataset)):
        window = dataset[index]
        assert int(window.tokens[0]) in {tokenizer.bos_id, tokenizer.bar_id}
        assert int(window.styles[0]) != Style.NULL.index


def test_evaluation_refuses_a_bar_longer_than_the_context(corpus: Corpus, tokenizer: DrumTokenizer) -> None:
    with pytest.raises(ValueError, match="compasso"):
        EvaluationWindows(corpus, tokenizer, DataSettings(context_length=16))
