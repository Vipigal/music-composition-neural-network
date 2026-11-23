import csv
import shutil
from pathlib import Path

import numpy as np
import pytest
from symusic.core import ScoreTick

from drumgen.config import (
    ExperimentConfig,
    ExperimentRun,
    MarkovSettings,
    PathSettings,
    PieceType,
    Style,
    TokenizerSettings,
)
from drumgen.dataset import Corpus
from drumgen.markov import KneserNeyModel, MarkovBaseline, MarkovTrainer
from drumgen.prepare import Preparer
from drumgen.tokenizer import DrumTokenizer

FIXTURES = Path(__file__).resolve().parent / "fixtures"
BARS = 8


@pytest.fixture(scope="module")
def tokenizer() -> DrumTokenizer:
    return DrumTokenizer.from_settings(TokenizerSettings())


def ids(tokenizer: DrumTokenizer, *tokens: str) -> list[int]:
    return [tokenizer.id_of(token) for token in tokens]


def framed(tokenizer: DrumTokenizer, content: list[int]) -> np.ndarray:
    return np.array([tokenizer.bos_id, *content, tokenizer.eos_id], dtype=np.int32)


def emittable(tokenizer: DrumTokenizer) -> np.ndarray:
    base = np.ones(tokenizer.vocab_size)
    base[[tokenizer.pad_id, tokenizer.bos_id]] = 0.0
    return base / base.sum()


def second_order_pattern(tokenizer: DrumTokenizer, repeats: int) -> np.ndarray:
    first, second = ids(tokenizer, "Position_0", "Position_480")
    return framed(tokenizer, [first, first, second] * repeats)


def fixture_sequence(tokenizer: DrumTokenizer, name: str) -> np.ndarray:
    return framed(tokenizer, tokenizer.encode(ScoreTick(FIXTURES / name)))


def corpus_of(tokenizer: DrumTokenizer, entries: list[tuple[np.ndarray, Style, PieceType]]) -> Corpus:
    return Corpus(
        sequences=[sequence for sequence, _, _ in entries],
        bar_starts=[np.array(tokenizer.bar_starts(sequence.tolist()), dtype=np.int32) for sequence, _, _ in entries],
        styles=[style for _, style, _ in entries],
        types=[piece_type for _, _, piece_type in entries],
    )


@pytest.fixture(scope="module")
def groove_corpus(tokenizer: DrumTokenizer) -> Corpus:
    return corpus_of(
        tokenizer,
        [
            (fixture_sequence(tokenizer, "funk_117_beat.mid"), Style.FUNK, PieceType.GROOVE),
            (fixture_sequence(tokenizer, "jazz_120_beat.mid"), Style.JAZZ, PieceType.GROOVE),
        ],
    )


def test_distribution_sums_to_one_on_seen_and_unseen_contexts(tokenizer: DrumTokenizer) -> None:
    sequence = fixture_sequence(tokenizer, "funk_117_beat.mid")
    model = KneserNeyModel.fit([sequence], order=4, base=emittable(tokenizer))
    unseen = [tokenizer.bos_id, *ids(tokenizer, "PitchDrum_51", "PitchDrum_51", "PitchDrum_51")]

    for history in (sequence[:20].tolist(), unseen):
        assert model.distribution(history, order=4).sum() == pytest.approx(1.0, abs=1e-12)


def test_unseen_context_backs_off_to_a_positive_probability_for_every_token(tokenizer: DrumTokenizer) -> None:
    model = KneserNeyModel.fit([second_order_pattern(tokenizer, 10)], order=3, base=emittable(tokenizer))
    unseen = [tokenizer.bos_id, *ids(tokenizer, "PitchDrum_51", "Velocity_7", "MicroTiming_60")]

    distribution = model.distribution(unseen, order=3)

    emitted = np.ones(tokenizer.vocab_size, dtype=bool)
    emitted[[tokenizer.pad_id, tokenizer.bos_id]] = False
    assert (distribution[emitted] > 0).all()


def test_pad_and_bos_are_never_probable(tokenizer: DrumTokenizer) -> None:
    sequence = fixture_sequence(tokenizer, "jazz_120_beat.mid")
    model = KneserNeyModel.fit([sequence], order=6, base=emittable(tokenizer))

    for end in (1, 5, 40):
        distribution = model.distribution(sequence[:end].tolist(), order=6)
        assert distribution[tokenizer.pad_id] == 0.0
        assert distribution[tokenizer.bos_id] == 0.0


def test_second_order_pattern_scores_better_at_order_two(tokenizer: DrumTokenizer) -> None:
    model = KneserNeyModel.fit([second_order_pattern(tokenizer, 50)], order=2, base=emittable(tokenizer))
    held_out = second_order_pattern(tokenizer, 20)

    nll = {order: -model.token_log_probs(held_out, order).mean() for order in (0, 1, 2)}

    assert nll[2] < nll[1] < nll[0]


def test_backoff_favors_tokens_seen_after_many_contexts_over_frequent_ones(tokenizer: DrumTokenizer) -> None:
    anchor, frequent, diverse, *predecessors = ids(
        tokenizer, "Position_0", "Position_480", "Position_960", "PitchDrum_36", "PitchDrum_38", "PitchDrum_42"
    )
    content = [anchor, frequent] * 30 + [token for predecessor in predecessors for token in (predecessor, diverse)]
    model = KneserNeyModel.fit([framed(tokenizer, content)], order=1, base=emittable(tokenizer))
    unseen = [tokenizer.bos_id, tokenizer.id_of("MicroTiming_60")]

    distribution = model.distribution(unseen, order=1)

    assert distribution[diverse] > distribution[frequent]


def test_sequence_start_keeps_raw_counts_below_the_top_order(tokenizer: DrumTokenizer) -> None:
    common, rare = ids(tokenizer, "Position_0", "Position_480")
    sequences = [framed(tokenizer, [common])] * 3 + [framed(tokenizer, [rare])]
    model = KneserNeyModel.fit(sequences, order=3, base=emittable(tokenizer))

    distribution = model.distribution([tokenizer.bos_id], order=3)

    assert distribution[common] > 2 * distribution[rare]


def test_truncated_model_scores_like_a_model_fitted_at_that_order(tokenizer: DrumTokenizer) -> None:
    train = fixture_sequence(tokenizer, "funk_117_beat.mid")
    held_out = fixture_sequence(tokenizer, "jazz_120_beat.mid")
    base = emittable(tokenizer)

    truncated = KneserNeyModel.fit([train], order=6, base=base).truncated(2)
    direct = KneserNeyModel.fit([train], order=2, base=base)

    np.testing.assert_array_equal(truncated.token_log_probs(held_out, 2), direct.token_log_probs(held_out, 2))


def test_vectorized_scores_match_the_step_by_step_distribution(tokenizer: DrumTokenizer) -> None:
    train = fixture_sequence(tokenizer, "funk_117_beat.mid")
    held_out = fixture_sequence(tokenizer, "jazz_120_beat.mid")[:120]
    model = KneserNeyModel.fit([train], order=5, base=emittable(tokenizer))

    scores = model.token_log_probs(held_out, order=5)

    stepwise = [
        np.log(model.distribution(held_out[:position].tolist(), order=5)[held_out[position]])
        for position in range(1, len(held_out))
    ]
    np.testing.assert_allclose(scores, stepwise, rtol=1e-12)


def test_fills_are_left_out_of_the_chains(tokenizer: DrumTokenizer, groove_corpus: Corpus) -> None:
    fill = (fixture_sequence(tokenizer, "rock_115_fill.mid"), Style.FUNK, PieceType.FILL)
    with_fill = corpus_of(
        tokenizer,
        [*zip(groove_corpus.sequences, groove_corpus.styles, groove_corpus.types, strict=True), fill],
    )

    plain = MarkovBaseline.fit(groove_corpus, tokenizer, order=3)
    mixed = MarkovBaseline.fit(with_fill, tokenizer, order=3)

    history = [tokenizer.bos_id]
    np.testing.assert_array_equal(
        plain.chain(Style.FUNK).distribution(history, 3), mixed.chain(Style.FUNK).distribution(history, 3)
    )


def test_each_style_has_its_own_chain(tokenizer: DrumTokenizer, groove_corpus: Corpus) -> None:
    baseline = MarkovBaseline.fit(groove_corpus, tokenizer, order=3)
    history = [tokenizer.bos_id]

    funk = baseline.chain(Style.FUNK).distribution(history, 3)
    jazz = baseline.chain(Style.JAZZ).distribution(history, 3)

    assert not np.allclose(funk, jazz)
    assert set(baseline.styles) == set(Style.evaluated())


def test_sample_stops_at_the_eighth_bar_line(tokenizer: DrumTokenizer) -> None:
    content = tokenizer.encode(ScoreTick(FIXTURES / "jazz_120_beat.mid"))
    long_groove = framed(tokenizer, content * 5)
    corpus = corpus_of(tokenizer, [(long_groove, Style.JAZZ, PieceType.GROOVE)])
    baseline = MarkovBaseline.fit(corpus, tokenizer, order=8)

    sample = baseline.sample(Style.JAZZ, seed=3, bars=BARS)

    assert sample.ids[0] == tokenizer.bos_id
    assert not sample.ended_early
    assert sample.ids.count(tokenizer.bar_id) == BARS - 1
    assert tokenizer.count_bars(sample.ids) == BARS
    assert tokenizer.decode(sample.ids).note_num() > 0


def test_sample_reports_an_eos_before_the_last_bar(tokenizer: DrumTokenizer) -> None:
    short = framed(tokenizer, ids(tokenizer, "Position_0", "PitchDrum_36", "Velocity_103", "MicroTiming_0"))
    baseline = MarkovBaseline.fit(corpus_of(tokenizer, [(short, Style.ROCK, PieceType.GROOVE)]), tokenizer, order=4)

    sample = baseline.sample(Style.ROCK, seed=0, bars=BARS)

    assert sample.ended_early
    assert sample.ids[-1] == tokenizer.eos_id


def test_sample_is_reproducible_by_seed(tokenizer: DrumTokenizer, groove_corpus: Corpus) -> None:
    baseline = MarkovBaseline.fit(groove_corpus, tokenizer, order=4)

    first = baseline.sample(Style.FUNK, seed=11, bars=BARS)

    assert baseline.sample(Style.FUNK, seed=11, bars=BARS) == first
    assert baseline.sample(Style.FUNK, seed=12, bars=BARS) != first


def test_saved_model_reloads_with_the_chosen_order(
    tokenizer: DrumTokenizer, groove_corpus: Corpus, tmp_path: Path
) -> None:
    fitted = MarkovBaseline.fit(groove_corpus, tokenizer, order=6)
    chosen = fitted.with_order(2)
    path = tmp_path / "markov.npz"

    chosen.save(path)
    reloaded = MarkovBaseline.load(path, tokenizer)

    assert reloaded.order == 2
    held_out = groove_corpus.sequences[0]
    for style in (Style.FUNK, Style.JAZZ):
        np.testing.assert_array_equal(
            reloaded.chain(style).token_log_probs(held_out, 2), fitted.chain(style).token_log_probs(held_out, 2)
        )
    assert reloaded.sample(Style.FUNK, seed=5, bars=BARS) == chosen.sample(Style.FUNK, seed=5, bars=BARS)


CATALOG_HEADER = "drummer,session,id,style,bpm,beat_type,time_signature,midi_filename,audio_filename,duration,split"
CATALOG_ROWS = [
    ("d1/s1/1", "funk", "beat", "funk_117_beat.mid", "train"),
    ("d1/s1/2", "jazz", "beat", "jazz_120_beat.mid", "train"),
    ("d1/s1/3", "rock", "fill", "rock_115_fill.mid", "train"),
    ("d1/s2/4", "funk", "beat", "funk_117_beat.mid", "validation"),
    ("d1/s2/5", "jazz", "beat", "jazz_120_beat.mid", "validation"),
    ("d1/s2/6", "soul", "beat", "funk_117_beat.mid", "validation"),
    ("d1/s2/7", "rock", "fill", "rock_115_fill.mid", "validation"),
]


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> ExperimentRun:
    root = tmp_path_factory.mktemp("gmd")
    raw_dir = root / "raw"
    lines = [CATALOG_HEADER]
    for piece_id, style, beat_type, fixture, split in CATALOG_ROWS:
        midi_path = raw_dir / f"{piece_id}.mid"
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIXTURES / fixture, midi_path)
        drummer, session, _ = piece_id.split("/")
        lines.append(
            f"{drummer},{drummer}/{session},{piece_id},{style},120,{beat_type},4-4,{piece_id}.mid,,30.0,{split}"
        )
    raw_dir.joinpath("info.csv").write_text("\n".join(lines) + "\n")
    paths = PathSettings(raw_dir=raw_dir, processed_dir=root / "processed", runs_dir=root / "runs")
    config = ExperimentConfig(name="markov", paths=paths, markov=MarkovSettings(candidate_orders=[1, 2, 4]))
    Preparer(config).run()
    return MarkovTrainer(config).run()


def read_nll_table(run: ExperimentRun) -> list[dict[str, str]]:
    with (run.path / MarkovTrainer.NLL_FILE).open(newline="") as file:
        return list(csv.DictReader(file))


def test_run_writes_validation_nll_per_order_and_style(trained_run: ExperimentRun) -> None:
    rows = read_nll_table(trained_run)

    assert [int(row["order"]) for row in rows] == [1, 2, 4]
    assert list(rows[0]) == ["order", "funk", "jazz", "pooled"]
    for row in rows:
        assert float(row["funk"]) > 0
        assert float(row["pooled"]) > 0


def test_run_records_the_order_with_lowest_pooled_nll(trained_run: ExperimentRun) -> None:
    rows = read_nll_table(trained_run)
    best = min(rows, key=lambda row: float(row["pooled"]))

    reopened = ExperimentRun.open(trained_run.path)

    assert reopened.record.config.markov.order == int(best["order"])
    assert reopened.record.config.markov.candidate_orders == [1, 2, 4]


def test_run_saves_a_model_that_samples_after_reload(trained_run: ExperimentRun) -> None:
    config = trained_run.record.config
    tokenizer = DrumTokenizer.from_file(config.paths.tokenizer_file)

    baseline = MarkovBaseline.load(trained_run.path / MarkovTrainer.MODEL_FILE, tokenizer)

    assert baseline.order == config.markov.order
    sample = baseline.sample(Style.JAZZ, seed=0, bars=config.prepare.reference_bars)
    assert sample.ended_early or tokenizer.count_bars(sample.ids) == config.prepare.reference_bars
    assert tokenizer.decode(sample.ids).note_num() > 0
