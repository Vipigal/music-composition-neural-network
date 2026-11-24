import csv
import dataclasses
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from symusic.core import ScoreTick
from torch.nn import functional

from drumgen.config import (
    DataSettings,
    ExperimentConfig,
    ExperimentRun,
    ModelSettings,
    PathSettings,
    PieceType,
    Style,
    TokenizerSettings,
    TrainingSettings,
)
from drumgen.dataset import Corpus, EvaluationWindows, TrainingWindows, WindowBatch
from drumgen.model import DrumTransformer, ModelCheckpoint
from drumgen.prepare import Preparer
from drumgen.tokenizer import DrumTokenizer
from drumgen.train import TrainingStep, TrainingRun, ValidationLoss, WarmupCosineSchedule

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TINY = ModelSettings(d_model=64, layers=2, heads=2, feedforward=128, dropout=0.0)
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def tokenizer() -> DrumTokenizer:
    return DrumTokenizer.from_settings(TokenizerSettings())


def corpus_of(tokenizer: DrumTokenizer, entries: list[tuple[str, Style]]) -> Corpus:
    sequences = []
    for fixture, _ in entries:
        ids = tokenizer.encode(ScoreTick(FIXTURES / fixture))
        sequences.append(np.array([tokenizer.bos_id, *ids, tokenizer.eos_id], dtype=np.int32))
    return Corpus(
        sequences=sequences,
        bar_starts=[np.array(tokenizer.bar_starts(list(sequence)), dtype=np.int32) for sequence in sequences],
        styles=[style for _, style in entries],
        types=[PieceType.GROOVE for _ in entries],
    )


def tiny_model(tokenizer: DrumTokenizer, context_length: int = 128) -> DrumTransformer:
    torch.manual_seed(0)
    return DrumTransformer(TINY, tokenizer.vocab_size, context_length, tokenizer.pad_id)


def test_learning_rate_warms_up_linearly_then_decays_by_cosine() -> None:
    schedule = WarmupCosineSchedule(warmup_steps=10, total_steps=110)

    assert schedule.factor(0) == pytest.approx(0.1)
    assert schedule.factor(9) == pytest.approx(1.0)
    assert schedule.factor(60) == pytest.approx(0.5)
    assert schedule.factor(110) == pytest.approx(0.0)
    assert schedule.factor(500) == pytest.approx(0.0)


def test_single_batch_is_memorized(tokenizer: DrumTokenizer) -> None:
    corpus = corpus_of(tokenizer, [("funk_117_beat.mid", Style.FUNK), ("jazz_120_beat.mid", Style.JAZZ)])
    windows = TrainingWindows(corpus, tokenizer, DataSettings(context_length=128), seed=0)
    batch = WindowBatch.collate([windows[0], windows[1]], tokenizer.pad_id)
    model = tiny_model(tokenizer)
    settings = TrainingSettings(learning_rate=3e-3, warmup_steps=1, device="cpu")
    step = TrainingStep.create(model, settings, total_steps=300, device=CPU)

    losses = [step.run(batch) for _ in range(300)]
    first, last = losses[0], losses[-1]

    assert first > 4.0
    assert last < 0.1


def test_validation_loss_is_token_weighted_overall_and_per_style(tokenizer: DrumTokenizer) -> None:
    corpus = corpus_of(tokenizer, [("funk_117_beat.mid", Style.FUNK), ("jazz_120_beat.mid", Style.JAZZ)])
    windows = EvaluationWindows(corpus, tokenizer, DataSettings(context_length=128))
    model = tiny_model(tokenizer).eval()
    totals: dict[Style, tuple[float, int]] = {Style.FUNK: (0.0, 0), Style.JAZZ: (0.0, 0)}
    with torch.no_grad():
        for index in range(len(windows)):
            window = windows[index]
            logits = model(window.tokens[None], window.styles[None], window.types[None])[0]
            scored = window.targets != tokenizer.pad_id
            loss = functional.cross_entropy(logits[scored], window.targets[scored], reduction="sum")
            style = list(Style)[int(window.styles[0])]
            total, count = totals[style]
            totals[style] = (total + float(loss), count + int(scored.sum()))

    measured = ValidationLoss.measure(model, windows, tokenizer.pad_id, batch_size=3, device=CPU)

    for style, (total, count) in totals.items():
        assert measured.per_style[style] == pytest.approx(total / count, rel=1e-5)
    pooled = sum(total for total, _ in totals.values()) / sum(count for _, count in totals.values())
    assert measured.overall == pytest.approx(pooled, rel=1e-5)


CATALOG_HEADER = "drummer,session,id,style,bpm,beat_type,time_signature,midi_filename,audio_filename,duration,split"
CATALOG_ROWS = [
    ("d1/s1/1", "funk", "beat", "funk_117_beat.mid", "train"),
    ("d1/s1/2", "jazz", "beat", "jazz_120_beat.mid", "train"),
    ("d1/s1/3", "rock", "fill", "rock_115_fill.mid", "train"),
    ("d1/s2/4", "funk", "beat", "funk_117_beat.mid", "validation"),
    ("d1/s2/5", "jazz", "beat", "jazz_120_beat.mid", "validation"),
]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> PathSettings:
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
    Preparer(ExperimentConfig(name="prepare", paths=paths)).run()
    return paths


def training_config(paths: PathSettings, **overrides: float) -> ExperimentConfig:
    training = dataclasses.replace(TrainingSettings(batch_size=4, warmup_steps=2, device="cpu"), **overrides)
    return ExperimentConfig(
        name="transformer", paths=paths, data=DataSettings(context_length=128), model=TINY, training=training
    )


def read_history(run: ExperimentRun) -> list[dict[str, str]]:
    with (run.path / TrainingRun.HISTORY_FILE).open(newline="") as file:
        return list(csv.DictReader(file))


def test_run_writes_checkpoints_history_and_curves(prepared: PathSettings) -> None:
    run = TrainingRun(training_config(prepared, max_epochs=2)).run()

    history = read_history(run)
    assert [int(row["epoch"]) for row in history] == [1, 2]
    for row in history:
        assert math.isfinite(float(row["train_loss"]))
        assert math.isfinite(float(row["val_loss"]))
        assert math.isfinite(float(row["val_funk"]))
        assert row["val_rock"] == ""
        assert float(row["seconds"]) > 0
    assert (run.path / TrainingRun.CURVES_FILE).stat().st_size > 0
    best = ModelCheckpoint.load(run.path / TrainingRun.BEST_FILE)
    last = ModelCheckpoint.load(run.path / TrainingRun.LAST_FILE)
    best_epoch = min(history, key=lambda row: float(row["val_loss"]))["epoch"]
    assert best.epoch == int(best_epoch)
    assert last.epoch == 2
    assert best.tokenizer_file == prepared.tokenizer_file
    assert ExperimentRun.open(run.path).record.config.model == TINY


def test_training_stops_after_patience_epochs_without_improvement(prepared: PathSettings) -> None:
    run = TrainingRun(training_config(prepared, max_epochs=6, patience=2, learning_rate=0.0)).run()

    assert [int(row["epoch"]) for row in read_history(run)] == [1, 2, 3]
    assert ModelCheckpoint.load(run.path / TrainingRun.BEST_FILE).epoch == 1
