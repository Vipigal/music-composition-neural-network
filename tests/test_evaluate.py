import csv
import dataclasses
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from symusic.core import NoteTick, ScoreTick, TrackTick

from drumgen.config import (
    EvaluationSettings,
    ExperimentConfig,
    ExperimentRun,
    PathSettings,
    PrepareSettings,
    Split,
    Style,
    System,
)
from drumgen.evaluate import EvaluateCommand, EvaluationReport, Evaluator, ReferenceSet
from drumgen.metrics import GrooveMetric, SetMetric
from drumgen.tokenizer import TICKS_PER_QUARTER

REPO_ROOT = Path(__file__).resolve().parents[1]
BARS = 8
WINDOWS_PER_STYLE = 6
SWING = {Style.ROCK: 0.5, Style.JAZZ: 0.64, Style.FUNK: 0.52}


@dataclass(frozen=True)
class EvaluatedTree:
    run: Path
    report: EvaluationReport


def window(rng: np.random.Generator, swing: float) -> ScoreTick:
    score = ScoreTick(TICKS_PER_QUARTER)
    track = TrackTick("Drums", 0, True)
    for beat in range(BARS * 4):
        start = beat * TICKS_PER_QUARTER
        jitter = rng.integers(-12, 13, size=3)
        track.notes.append(NoteTick(start + max(int(jitter[0]), 0), 60, 42, int(rng.integers(70, 110))))
        offbeat = start + int(swing * TICKS_PER_QUARTER) + int(jitter[1])
        track.notes.append(NoteTick(offbeat, 60, 42, int(rng.integers(40, 80))))
        piece = 36 if beat % 2 == 0 else 38
        track.notes.append(NoteTick(start + max(int(jitter[2]), 0), 60, piece, int(rng.integers(80, 120))))
        if rng.random() < 0.5:
            track.notes.append(NoteTick(start + 3 * TICKS_PER_QUARTER // 4, 60, 36, int(rng.integers(50, 90))))
        if rng.random() < 0.5:
            track.notes.append(NoteTick(start + TICKS_PER_QUARTER // 4, 60, 38, int(rng.integers(20, 50))))
    score.tracks.append(track)
    return score


def write_windows(folder: Path, style: Style, seed: int) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    files = []
    for index in range(WINDOWS_PER_STYLE):
        path = folder / f"{style.value}_{index}.mid"
        window(rng, SWING[style]).dump_midi(path)
        files.append(path)
    return files


def config_for(root: Path, **evaluation: object) -> ExperimentConfig:
    return ExperimentConfig(
        name="avaliacao",
        paths=PathSettings(processed_dir=root / "processed", runs_dir=root / "runs"),
        evaluation=dataclasses.replace(EvaluationSettings(), **evaluation),
    )


def copy_folder(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.mid"):
        shutil.copy(path, target / path.name)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory: pytest.TempPathFactory) -> EvaluatedTree:
    root = tmp_path_factory.mktemp("avaliacao")
    config = config_for(root)
    for offset, style in enumerate(SWING):
        write_windows(config.paths.reference_dir(Split.VAL, style), style, seed=offset)
    run = ExperimentRun.create(config).path
    samples = run / "samples"
    for style in SWING:
        copy_folder(config.paths.reference_dir(Split.VAL, style), samples / "humano" / style.value)
    copy_folder(config.paths.reference_dir(Split.VAL, Style.ROCK), samples / "humano" / "rock_quantizado")
    copy_folder(config.paths.reference_dir(Split.VAL, Style.JAZZ), samples / "markov" / "rock")
    copy_folder(config.paths.reference_dir(Split.VAL, Style.ROCK), samples / "markov" / "jazz")
    copy_folder(config.paths.reference_dir(Split.VAL, Style.ROCK), samples / "humano" / "soul")
    copy_folder(config.paths.reference_dir(Split.VAL, Style.ROCK), samples / "humano" / "troca_rock-funk")
    copy_folder(config.paths.reference_dir(Split.VAL, Style.FUNK), samples / "humano" / "virada_funk")
    report = Evaluator(config).run(run)
    return EvaluatedTree(run=run, report=report)


def summary_row(rows: list[dict[str, str]], system: str, variant: str, style: str, metric: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if (row["system"], row["variant"], row["style"], row["metric"]) == (system, variant, style, metric)
    ]
    assert len(matches) == 1, (system, variant, style, metric)
    return matches[0]


def test_reference_comes_from_the_configured_split(tmp_path: Path) -> None:
    config = config_for(tmp_path, reference_split=Split.TEST)
    write_windows(config.paths.reference_dir(Split.TEST, Style.JAZZ), Style.JAZZ, seed=1)
    write_windows(config.paths.reference_dir(Split.VAL, Style.ROCK), Style.ROCK, seed=2)

    reference = ReferenceSet.load(config.paths, Split.TEST, BARS)

    assert reference.styles == [Style.JAZZ]
    assert all(sample.file.startswith("test/jazz/") for sample in reference.groups[Style.JAZZ].samples)


def test_writes_every_output_into_the_run(evaluated: EvaluatedTree) -> None:
    output = evaluated.run / "eval"

    assert evaluated.report.output_dir == output
    for name in ["per_sample.csv", "summary.csv", "style_matrix.csv"]:
        assert (output / name).stat().st_size > 0
    for metric in GrooveMetric:
        assert (output / f"distribution_{metric.value}.png").stat().st_size > 0
    for system, variant in [("humano", "original"), ("humano", "quantizado"), ("markov", "original")]:
        assert (output / f"style_matrix_{system}_{variant}.png").stat().st_size > 0


def test_per_sample_has_one_row_per_sample_and_per_reference_window(evaluated: EvaluatedTree) -> None:
    rows = read_csv(evaluated.run / "eval" / "per_sample.csv")

    evaluated_samples = 3 + 1 + 2 + 1
    assert len(rows) == (evaluated_samples + len(SWING)) * WINDOWS_PER_STYLE
    assert list(rows[0]) == ["system", "variant", "style", "file", *(metric.value for metric in GrooveMetric)]
    reference = [row for row in rows if row["system"] == "referencia"]
    assert {row["file"].split("/")[0] for row in reference} == {"val"}
    quantized = [row for row in rows if row["variant"] == "quantizado"]
    assert {(row["system"], row["style"]) for row in quantized} == {("humano", "rock")}
    assert all(row["file"].startswith("humano/rock_quantizado/") for row in quantized)


def test_human_windows_against_their_own_reference_overlap_completely(evaluated: EvaluatedTree) -> None:
    rows = read_csv(evaluated.run / "eval" / "summary.csv")

    for style in SWING:
        for metric in GrooveMetric:
            row = summary_row(rows, "humano", "original", style.value, metric.value)
            assert float(row["overlap_area"]) == pytest.approx(1.0, abs=1e-6)
            assert float(row["mean_distance"]) == pytest.approx(0.0)
            assert row["reference_split"] == "val"
        for metric in SetMetric:
            row = summary_row(rows, "humano", "original", style.value, metric.value)
            assert float(row["kl_divergence"]) == pytest.approx(0.0, abs=1e-9)
            assert row["positions"] == "4"


def test_jazz_labelled_as_rock_overlaps_less_than_rock(evaluated: EvaluatedTree) -> None:
    rows = read_csv(evaluated.run / "eval" / "summary.csv")

    own = summary_row(rows, "humano", "original", "rock", GrooveMetric.SWING.value)
    swapped = summary_row(rows, "markov", "original", "rock", GrooveMetric.SWING.value)

    assert float(swapped["overlap_area"]) < 0.2 < float(own["overlap_area"])
    assert float(swapped["mean_distance"]) > 0.1


def test_folders_without_reference_are_left_out_with_a_warning(evaluated: EvaluatedTree) -> None:
    rows = read_csv(evaluated.run / "eval" / "summary.csv")
    warnings = "\n".join(evaluated.report.warnings)

    assert {row["variant"] for row in rows} == {"original", "quantizado"}
    assert not [row for row in rows if row["style"] == "soul"]
    assert "troca_rock-funk" in warnings
    assert "virada_funk" in warnings
    assert "soul" in warnings


def test_style_matrix_has_the_diagonal_for_human_windows(evaluated: EvaluatedTree) -> None:
    rows = read_csv(evaluated.run / "eval" / "style_matrix.csv")

    human = [row for row in rows if (row["system"], row["variant"]) == ("humano", "original")]
    assert len(human) == 4 * len(SWING)
    assert {row["diagonal_wins"] for row in human} == {"3"}
    assert {row["diagonal_rows"] for row in human} == {"3"}
    assert all(GrooveMetric.DRUM_PATTERN_CONSISTENCY.value in row["excluded_metrics"].split(";") for row in human)
    diagonal = [row for row in human if row["generated_style"] == row["reference_style"]]
    assert all(float(row["distance"]) == pytest.approx(0.0) for row in diagonal)
    assert all(row["row_minimum"] == "True" for row in diagonal)
    swapped = [row for row in rows if (row["system"], row["generated_style"]) == ("markov", "rock")]
    nearest = min(swapped, key=lambda row: float(row["distance"]))
    assert nearest["reference_style"] == "jazz"


def test_output_goes_to_the_configured_folder(tmp_path: Path) -> None:
    run = tmp_path / "run"

    assert Evaluator(config_for(tmp_path)).output_dir(run) == run / "eval"
    assert Evaluator(config_for(tmp_path, output_dir=tmp_path / "results")).output_dir(run) == tmp_path / "results"


def test_run_with_other_window_length_is_rejected(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    run = ExperimentRun.create(dataclasses.replace(config, prepare=PrepareSettings(reference_bars=4))).path
    (run / "samples" / "humano" / "rock").mkdir(parents=True)

    with pytest.raises(ValueError, match="compassos"):
        Evaluator(config).run(run)


def test_run_without_samples_is_rejected(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    run = ExperimentRun.create(config).path

    with pytest.raises(FileNotFoundError, match="samples"):
        Evaluator(config).run(run)


def test_unknown_system_is_rejected(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    run = ExperimentRun.create(config).path
    (run / "samples" / "lstm" / "rock").mkdir(parents=True)

    with pytest.raises(ValueError, match="lstm"):
        Evaluator(config).run(run)


def test_sample_longer_than_the_window_names_the_file(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    write_windows(config.paths.reference_dir(Split.VAL, Style.ROCK), Style.ROCK, seed=0)
    run = ExperimentRun.create(config).path
    long_score = window(np.random.default_rng(0), 0.5)
    long_score.tracks[0].notes.append(NoteTick(BARS * 4 * TICKS_PER_QUARTER, 60, 36, 90))
    target = run / "samples" / "humano" / "rock"
    target.mkdir(parents=True)
    long_score.dump_midi(target / "0.mid")

    with pytest.raises(ValueError, match="0.mid"):
        Evaluator(config).run(run)


def test_command_reads_config_and_run() -> None:
    command = EvaluateCommand.parse(["--config", str(REPO_ROOT / "configs" / "eval.yaml"), "--run", "runs/x"])

    assert command.config.evaluation.reference_split is Split.VAL
    assert command.config.evaluation.output_dir is None
    assert command.run == Path("runs/x")
    assert System("humano") is System.HUMAN


def test_style_matrix_rows_and_columns_follow_the_style_order(evaluated: EvaluatedTree) -> None:
    rows = read_csv(evaluated.run / "eval" / "style_matrix.csv")

    human = [row for row in rows if (row["system"], row["variant"]) == ("humano", "original")]
    generated = list(dict.fromkeys(row["generated_style"] for row in human))
    reference = list(dict.fromkeys(row["reference_style"] for row in human))
    assert generated == ["rock", "funk", "jazz", "soul"]
    assert reference == ["rock", "funk", "jazz"]


def test_new_evaluation_removes_the_outputs_of_the_previous_one(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    write_windows(config.paths.reference_dir(Split.VAL, Style.ROCK), Style.ROCK, seed=0)
    run = ExperimentRun.create(config).path
    for name in ["rock", "rock_quantizado"]:
        copy_folder(config.paths.reference_dir(Split.VAL, Style.ROCK), run / "samples" / "humano" / name)
    output = Evaluator(config).run(run).output_dir
    notes = output / "notas.txt"
    notes.write_text("do usuário")
    shutil.rmtree(run / "samples" / "humano" / "rock_quantizado")

    Evaluator(config).run(run)

    assert not (output / "style_matrix_humano_quantizado.png").exists()
    assert (output / "style_matrix_humano_original.png").exists()
    assert {row["variant"] for row in read_csv(output / "summary.csv")} == {"original"}
    assert notes.read_text() == "do usuário"
