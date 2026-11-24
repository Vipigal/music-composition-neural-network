import csv
import dataclasses
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
from symusic.core import ScoreTick

from drumgen.config import (
    ExperimentConfig,
    ExperimentRun,
    GenerationSettings,
    MarkovSettings,
    ModelSettings,
    PathSettings,
    PieceType,
    SampleFolder,
    SamplingPoint,
    Style,
    StyleSwitch,
    System,
    TokenizerSettings,
    VariantKind,
)
from drumgen.evaluate import Evaluator
from drumgen.generate import (
    MANIFEST_COLUMNS,
    BatchGenerator,
    GenerateCommand,
    GeneratedSequence,
    Generation,
    MarkovGenerator,
    TransformerGenerator,
)
from drumgen.markov import MarkovBaseline, MarkovTrainer
from drumgen.model import ComputeDevice, DrumTransformer, ModelCheckpoint
from drumgen.prepare import Preparer
from drumgen.sampling import BarPlan, GenerationRequest, SpecialTokens, TokenSampler
from drumgen.tokenizer import TICKS_PER_QUARTER, DrumTokenizer

FIXTURES = Path(__file__).resolve().parent / "fixtures"
BARS = 8
SIXTEENTH = TICKS_PER_QUARTER // 4
BAR_CONTENT = [
    "Position_0",
    "PitchDrum_36",
    "MicroTiming_15",
    "Velocity_103",
    "Position_480",
    "PitchDrum_38",
    "MicroTiming_-30",
    "Velocity_63",
    "Position_960",
    "PitchDrum_42",
    "MicroTiming_0",
    "Velocity_79",
    "Position_1320",
    "PitchDrum_42",
    "MicroTiming_45",
    "Velocity_47",
]
NOTES_PER_BAR = 4
ALL_CAPABILITIES = frozenset({VariantKind.GUIDED, VariantKind.STYLE_SWITCH, VariantKind.FILL})
NO_CAPABILITIES: frozenset[VariantKind] = frozenset()
SMALL_MODEL = ModelSettings(d_model=32, layers=2, heads=2, feedforward=64, dropout=0.0)


@pytest.fixture(scope="module")
def tokenizer() -> DrumTokenizer:
    return DrumTokenizer.from_settings(TokenizerSettings())


@dataclass(frozen=True)
class ScriptedGenerator:
    tokenizer: DrumTokenizer
    capabilities: frozenset[VariantKind]
    early_seeds: frozenset[int] = frozenset()
    last_bar_eos_seeds: frozenset[int] = frozenset()
    dangling_pitch: bool = False
    bars_returned: int | None = None
    requests: list[GenerationRequest] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GeneratedSequence]:
        self.batch_sizes.append(len(requests))
        return [self._one(request) for request in requests]

    def _one(self, request: GenerationRequest) -> GeneratedSequence:
        self.requests.append(request)
        bar = [self.tokenizer.id_of(token) for token in BAR_CONTENT]
        if request.seed in self.early_seeds:
            return GeneratedSequence(ids=[self.tokenizer.bos_id, *bar, self.tokenizer.bar_id, *bar, self.tokenizer.eos_id])
        ids = [self.tokenizer.bos_id]
        for index in range(self.bars_returned or request.bars):
            if index:
                ids.append(self.tokenizer.bar_id)
            ids += bar
        if self.dangling_pitch:
            ids.append(self.tokenizer.id_of("PitchDrum_51"))
        if request.seed in self.last_bar_eos_seeds:
            return GeneratedSequence(ids=[*ids[:-4], self.tokenizer.eos_id])
        return GeneratedSequence(ids=ids)


def settings(**overrides: object) -> GenerationSettings:
    return dataclasses.replace(GenerationSettings(samples_per_style=3, first_seed=10), **overrides)


def batch(tokenizer: DrumTokenizer, generator: ScriptedGenerator, **overrides: object) -> BatchGenerator:
    return BatchGenerator(generator, tokenizer, settings(**overrides), bars=BARS, system=System.MARKOV)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def seeds_in(folder: Path) -> list[int]:
    return sorted(int(path.stem) for path in folder.glob("*.mid"))


def test_writes_one_folder_per_evaluated_style_with_seeded_files(tokenizer: DrumTokenizer, tmp_path: Path) -> None:
    generator = ScriptedGenerator(tokenizer, NO_CAPABILITIES)

    batch(tokenizer, generator).run(tmp_path)

    for style in Style.evaluated():
        assert seeds_in(tmp_path / style.value) == [10, 11, 12]
        assert seeds_in(tmp_path / f"{style.value}_quantizado") == [10, 11, 12]
    assert {path.name for path in tmp_path.iterdir() if path.is_dir()} == {
        name for style in Style.evaluated() for name in (style.value, f"{style.value}_quantizado")
    }


def test_written_samples_have_all_bars_and_drum_tracks(tokenizer: DrumTokenizer, tmp_path: Path) -> None:
    batch(tokenizer, ScriptedGenerator(tokenizer, NO_CAPABILITIES)).run(tmp_path)

    score = ScoreTick(tmp_path / "jazz" / "10.mid")

    assert all(track.is_drum for track in score.tracks)
    assert score.note_num() == BARS * NOTES_PER_BAR
    last_onset = max(note.time for track in score.tracks for note in track.notes)
    assert last_onset // (4 * TICKS_PER_QUARTER) == BARS - 1


def test_early_eos_is_logged_and_redone_with_the_next_unused_seed(tokenizer: DrumTokenizer, tmp_path: Path) -> None:
    generator = ScriptedGenerator(tokenizer, NO_CAPABILITIES, early_seeds=frozenset({11}))

    report = batch(tokenizer, generator).run(tmp_path)

    assert seeds_in(tmp_path / "funk") == [10, 12, 13]
    assert seeds_in(tmp_path / "funk_quantizado") == [10, 12, 13]
    rows = [row for row in read_manifest(report.manifest_file) if row["folder"] == "funk"]
    assert [(row["seed"], row["ended_early"], row["file"]) for row in rows] == [
        ("10", "False", "funk/10.mid"),
        ("11", "True", ""),
        ("12", "False", "funk/12.mid"),
        ("13", "False", "funk/13.mid"),
    ]
    assert report.attempts[SampleFolder.original(Style.FUNK)] == 4


def test_eos_inside_the_last_bar_counts_as_incomplete(tokenizer: DrumTokenizer, tmp_path: Path) -> None:
    generator = ScriptedGenerator(tokenizer, NO_CAPABILITIES, last_bar_eos_seeds=frozenset({10}))

    batch(tokenizer, generator).run(tmp_path)

    assert seeds_in(tmp_path / "rock") == [11, 12, 13]


def test_quantized_variant_keeps_notes_and_pitches_and_moves_onsets_to_the_grid(
    tokenizer: DrumTokenizer, tmp_path: Path
) -> None:
    batch(tokenizer, ScriptedGenerator(tokenizer, NO_CAPABILITIES)).run(tmp_path)

    original = ScoreTick(tmp_path / "soul" / "11.mid").tracks[0].notes
    quantized = ScoreTick(tmp_path / "soul_quantizado" / "11.mid").tracks[0].notes

    assert len(quantized) == len(original)
    assert [note.pitch for note in quantized] == [note.pitch for note in original]
    assert [note.velocity for note in quantized] == [note.velocity for note in original]
    assert all(note.time % SIXTEENTH == 0 for note in quantized)
    assert any(note.time % SIXTEENTH for note in original)
    assert [round(note.time / SIXTEENTH) for note in original] == [note.time // SIXTEENTH for note in quantized]


def test_manifest_records_every_attempt_with_its_parameters(tokenizer: DrumTokenizer, tmp_path: Path) -> None:
    generator = ScriptedGenerator(tokenizer, NO_CAPABILITIES, dangling_pitch=True)

    report = batch(tokenizer, generator, samples_per_style=1, temperature=1.0, top_p=None).run(tmp_path)

    rows = read_manifest(report.manifest_file)
    assert report.manifest_file == tmp_path / "manifest.csv"
    assert list(rows[0]) == MANIFEST_COLUMNS
    assert len(rows) == 2 * len(Style.evaluated())
    original = next(row for row in rows if row["folder"] == "latin")
    quantized = next(row for row in rows if row["folder"] == "latin_quantizado")
    expected_tokens = str(1 + BARS * len(BAR_CONTENT) + BARS - 1 + 1)
    assert original == {
        "system": "markov",
        "folder": "latin",
        "variant": "original",
        "style": "latin",
        "seed": "10",
        "cfg_scale": "",
        "temperature": "1.0",
        "top_k": "",
        "top_p": "",
        "tokens": expected_tokens,
        "ended_early": "False",
        "dropped_notes": "1",
        "file": "latin/10.mid",
    }
    assert quantized | {"folder": "latin", "variant": "original", "file": "latin/10.mid"} == original


def test_unsupported_variants_are_skipped_with_a_warning(tokenizer: DrumTokenizer, tmp_path: Path) -> None:
    generator = ScriptedGenerator(tokenizer, NO_CAPABILITIES)

    report = batch(tokenizer, generator, samples_per_style=1).run(tmp_path)

    assert not list(tmp_path.glob("*_cfg*")) and not list(tmp_path.glob("troca_*")) and not list(tmp_path.glob("virada_*"))
    assert len(report.warnings) == 1
    assert any("troca_rock-funk" in warning for warning in report.warnings)
    assert all(request.cfg_scale is None for request in generator.requests)


def test_supported_variants_get_their_own_plan_and_guidance(tokenizer: DrumTokenizer, tmp_path: Path) -> None:
    generator = ScriptedGenerator(tokenizer, ALL_CAPABILITIES)

    report = batch(
        tokenizer,
        generator,
        samples_per_style=1,
        cfg_scale=1.5,
        cfg_scale_sweep=[],
        style_switches=[StyleSwitch(Style.ROCK, Style.FUNK)],
        fill_styles=[Style.JAZZ],
    ).run(tmp_path)

    assert report.warnings == []
    folders = {path.name for path in tmp_path.iterdir() if path.is_dir()}
    assert {"rock", "rock_quantizado", "troca_rock-funk", "virada_jazz"} <= folders
    assert not list(tmp_path.glob("*_cfg*"))
    assert not (tmp_path / "virada_rock").exists()
    assert {request.cfg_scale for request in generator.requests} == {1.5}
    switch = next(request for request in generator.requests if request.plan[0].style is not request.plan[-1].style)
    assert [bar.style for bar in switch.plan] == [Style.ROCK] * 4 + [Style.FUNK] * 4
    fill = next(request for request in generator.requests if request.plan[-1].piece_type is PieceType.FILL)
    assert [bar.piece_type for bar in fill.plan] == [PieceType.GROOVE] * (BARS - 1) + [PieceType.FILL]
    assert {bar.style for bar in fill.plan} == {Style.JAZZ}
    rows = read_manifest(report.manifest_file)
    assert {row["variant"] for row in rows if row["folder"] == "troca_rock-funk"} == {"troca_rock-funk"}


def test_guidance_sweep_run_generates_only_the_sweep(tokenizer: DrumTokenizer, tmp_path: Path) -> None:
    generator = ScriptedGenerator(tokenizer, ALL_CAPABILITIES)

    report = batch(tokenizer, generator, samples_per_style=1, cfg_scale_sweep=[1.0, 1.5], fill_styles=[Style.JAZZ]).run(
        tmp_path
    )

    folders = {path.name for path in tmp_path.iterdir() if path.is_dir()}
    assert folders == {f"{style.value}_cfg{scale}_t1_p1" for style in Style.evaluated() for scale in ("1", "1.5")}
    rows = read_manifest(report.manifest_file)
    assert {row["cfg_scale"] for row in rows if row["folder"] == "rock_cfg1.5_t1_p1"} == {"1.5"}


def test_sweep_of_d18_writes_twelve_folders_per_style_and_no_final_folders(
    tokenizer: DrumTokenizer, tmp_path: Path
) -> None:
    sweep = ExperimentConfig.from_yaml(Path(__file__).resolve().parents[1] / "configs" / "sweep.yaml").generation
    generator = ScriptedGenerator(tokenizer, ALL_CAPABILITIES)

    batch(
        tokenizer,
        generator,
        samples_per_style=1,
        cfg_scale_sweep=sweep.cfg_scale_sweep,
        temperature_sweep=sweep.temperature_sweep,
        top_p_sweep=sweep.top_p_sweep,
    ).run(tmp_path)

    folders = {path.name for path in tmp_path.iterdir() if path.is_dir()}
    for style in Style.evaluated():
        assert len({folder for folder in folders if folder.startswith(f"{style.value}_cfg")}) == 12
        assert style.value not in folders
        assert f"{style.value}_quantizado" not in folders
    assert len(folders) == 12 * len(Style.evaluated())
    assert not list(tmp_path.glob("troca_*"))


def test_sweep_requests_carry_the_temperature_and_top_p_of_their_point(tokenizer: DrumTokenizer, tmp_path: Path) -> None:
    generator = ScriptedGenerator(tokenizer, ALL_CAPABILITIES)

    report = batch(
        tokenizer,
        generator,
        samples_per_style=1,
        temperature=0.7,
        top_k=12,
        top_p=0.5,
        cfg_scale_sweep=[1.5, 3.0],
        temperature_sweep=[0.9, 1.0],
        top_p_sweep=[0.9, 1.0],
    ).run(tmp_path)

    points = {(request.cfg_scale, request.temperature, request.top_p) for request in generator.requests}
    assert points == {
        (scale, temperature, top_p) for scale in (1.5, 3.0) for temperature in (0.9, 1.0) for top_p in (0.9, None)
    }
    assert {request.top_k for request in generator.requests} == {12}
    for request in generator.requests:
        assert request.cfg_scale is not None
        point = SamplingPoint.of(request.cfg_scale, request.temperature, request.top_p)
        assert (tmp_path / SampleFolder.guided(request.plan[0].style, point).name / f"{request.seed}.mid").is_file()
    rows = {row["folder"]: row for row in read_manifest(report.manifest_file)}
    assert (rows["jazz_cfg3_t0.9_p0.9"]["cfg_scale"], rows["jazz_cfg3_t0.9_p0.9"]["temperature"]) == ("3.0", "0.9")
    assert rows["jazz_cfg3_t0.9_p0.9"]["top_p"] == "0.9"
    assert rows["jazz_cfg3_t0.9_p1"]["top_p"] == ""
    assert rows["jazz_cfg3_t0.9_p1"]["variant"] == "cfg3_t0.9_p1"


def test_empty_temperature_and_top_p_sweeps_use_the_single_values(tokenizer: DrumTokenizer, tmp_path: Path) -> None:
    generator = ScriptedGenerator(tokenizer, ALL_CAPABILITIES)

    batch(tokenizer, generator, samples_per_style=1, temperature=0.95, top_p=0.8, cfg_scale_sweep=[3.0]).run(tmp_path)

    assert {path.name for path in tmp_path.iterdir() if path.is_dir()} == {
        f"{style.value}_cfg3_t0.95_p0.8" for style in Style.evaluated()
    }
    assert {(request.temperature, request.top_p) for request in generator.requests} == {(0.95, 0.8)}


def test_final_run_ignores_temperature_and_top_p_sweeps_without_a_guidance_sweep(
    tokenizer: DrumTokenizer, tmp_path: Path
) -> None:
    generator = ScriptedGenerator(tokenizer, ALL_CAPABILITIES)

    batch(
        tokenizer,
        generator,
        samples_per_style=1,
        temperature=0.9,
        top_p=0.9,
        cfg_scale=1.5,
        temperature_sweep=[0.8, 1.0],
        top_p_sweep=[0.7],
    ).run(tmp_path)

    folders = {path.name for path in tmp_path.iterdir() if path.is_dir()}
    assert folders == {
        name for style in Style.evaluated() for name in (style.value, f"{style.value}_quantizado")
    } | {"troca_rock-funk"}
    assert {(request.cfg_scale, request.temperature, request.top_p) for request in generator.requests} == {(1.5, 0.9, 0.9)}


def test_seeds_are_generated_in_batches_without_extra_attempts(tokenizer: DrumTokenizer, tmp_path: Path) -> None:
    generator = ScriptedGenerator(tokenizer, NO_CAPABILITIES, early_seeds=frozenset({11, 12}))

    report = batch(tokenizer, generator, samples_per_style=5, batch_size=4).run(tmp_path)

    assert seeds_in(tmp_path / "rock") == [10, 13, 14, 15, 16]
    assert generator.batch_sizes[:2] == [4, 3]
    assert report.attempts[SampleFolder.original(Style.ROCK)] == 7


def test_complete_sequence_with_the_wrong_number_of_bars_is_refused(tokenizer: DrumTokenizer, tmp_path: Path) -> None:
    generator = ScriptedGenerator(tokenizer, NO_CAPABILITIES, bars_returned=BARS - 1)

    with pytest.raises(ValueError, match="7 compassos"):
        batch(tokenizer, generator).run(tmp_path)


def test_gives_up_when_almost_every_attempt_ends_early(tokenizer: DrumTokenizer, tmp_path: Path) -> None:
    generator = ScriptedGenerator(tokenizer, NO_CAPABILITIES, early_seeds=frozenset(range(10, 10_000)))

    with pytest.raises(RuntimeError, match="EOS"):
        batch(tokenizer, generator).run(tmp_path)


def test_refuses_to_overwrite_existing_samples_of_the_system(tokenizer: DrumTokenizer, tmp_path: Path) -> None:
    (tmp_path / "rock").mkdir()

    with pytest.raises(FileExistsError):
        batch(tokenizer, ScriptedGenerator(tokenizer, NO_CAPABILITIES)).run(tmp_path)


def test_original_and_quantized_are_not_capabilities(tokenizer: DrumTokenizer, tmp_path: Path) -> None:
    generator = ScriptedGenerator(tokenizer, frozenset({VariantKind.ORIGINAL, VariantKind.QUANTIZED}))

    with pytest.raises(ValueError, match="original"):
        batch(tokenizer, generator).run(tmp_path)


CATALOG_HEADER = "drummer,session,id,style,bpm,beat_type,time_signature,midi_filename,audio_filename,duration,split"
CATALOG_ROWS = [
    ("d1/s1/1", "funk", "beat", "funk_117_beat.mid", "train"),
    ("d1/s1/2", "jazz", "beat", "jazz_120_beat.mid", "train"),
    ("d1/s1/3", "rock", "beat", "funk_117_beat.mid", "train"),
    ("d1/s1/4", "latin", "beat", "jazz_120_beat.mid", "train"),
    ("d1/s1/5", "hiphop", "beat", "funk_117_beat.mid", "train"),
    ("d1/s1/6", "soul", "beat", "jazz_120_beat.mid", "train"),
    ("d1/s2/7", "funk", "beat", "funk_117_beat.mid", "validation"),
    ("d1/s2/8", "jazz", "beat", "jazz_120_beat.mid", "validation"),
]


@pytest.fixture(scope="module")
def markov_run(tmp_path_factory: pytest.TempPathFactory) -> ExperimentRun:
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
    config = ExperimentConfig(
        name="markov",
        paths=paths,
        markov=MarkovSettings(candidate_orders=[2, 4]),
        generation=GenerationSettings(samples_per_style=2),
    )
    Preparer(config).run()
    return MarkovTrainer(config).run()


def test_markov_adapter_passes_the_baseline_sample_through(markov_run: ExperimentRun) -> None:
    config = markov_run.record.config
    tokenizer = DrumTokenizer.from_file(config.paths.tokenizer_file)
    baseline = MarkovBaseline.load(markov_run.path / MarkovTrainer.MODEL_FILE, tokenizer)
    generator = MarkovGenerator(baseline)
    request = GenerationRequest(
        plan=(BarPlan(Style.FUNK, PieceType.GROOVE),) * BARS, seed=4, cfg_scale=None, temperature=1.0, top_k=None, top_p=None
    )

    generated = generator.generate([request])[0]

    expected = baseline.sample(Style.FUNK, seed=4, bars=BARS)
    assert generated.ids == expected.ids
    assert generator.capabilities == NO_CAPABILITIES


@pytest.mark.parametrize(
    "change",
    [
        {"plan": (BarPlan(Style.ROCK, PieceType.GROOVE),) * 4 + (BarPlan(Style.FUNK, PieceType.GROOVE),) * 4},
        {"plan": (BarPlan(Style.ROCK, PieceType.GROOVE),) * 7 + (BarPlan(Style.ROCK, PieceType.FILL),)},
        {"cfg_scale": 3.0},
        {"temperature": 0.9},
        {"top_k": 20},
        {"top_p": 0.9},
    ],
)
def test_markov_adapter_refuses_controls_the_chain_does_not_have(
    markov_run: ExperimentRun, change: dict[str, object]
) -> None:
    config = markov_run.record.config
    tokenizer = DrumTokenizer.from_file(config.paths.tokenizer_file)
    generator = MarkovGenerator(MarkovBaseline.load(markov_run.path / MarkovTrainer.MODEL_FILE, tokenizer))
    request = GenerationRequest(
        plan=(BarPlan(Style.ROCK, PieceType.GROOVE),) * BARS, seed=0, cfg_scale=None, temperature=1.0, top_k=None, top_p=None
    )

    with pytest.raises(ValueError):
        generator.generate([dataclasses.replace(request, **change)])


def test_command_writes_into_the_model_run_by_default(tmp_path: Path) -> None:
    config_file = tmp_path / "markov.yaml"
    config_file.write_text("name: markov\n")

    command = GenerateCommand.parse(["--config", str(config_file), "--system", "markov", "--model", "runs/x"])
    elsewhere = GenerateCommand.parse(
        ["--config", str(config_file), "--system", "markov", "--model", "runs/x", "--run", "runs/y"]
    )

    assert (command.system, command.model, command.run) == (System.MARKOV, Path("runs/x"), Path("runs/x"))
    assert elsewhere.run == Path("runs/y")


def test_command_rejects_systems_that_are_not_generated(tmp_path: Path) -> None:
    config_file = tmp_path / "markov.yaml"
    config_file.write_text("name: markov\n")

    with pytest.raises(SystemExit):
        GenerateCommand.parse(["--config", str(config_file), "--system", "humano", "--model", "runs/x"])


def test_markov_run_generates_samples_that_the_evaluation_reads(markov_run: ExperimentRun) -> None:
    config = markov_run.record.config

    report = Generation(config).run(System.MARKOV, markov_run.path, markov_run.path)

    samples = markov_run.path / "samples" / "markov"
    assert report.manifest_file == samples / "manifest.csv"
    for style in Style.evaluated():
        assert len(list((samples / style.value).glob("*.mid"))) == 2
        assert len(list((samples / f"{style.value}_quantizado").glob("*.mid"))) == 2
    rows = read_manifest(report.manifest_file)
    written = [row for row in rows if row["file"] and row["variant"] == "original"]
    assert len(written) == 2 * len(Style.evaluated())
    assert all(row["ended_early"] == "False" for row in written)

    evaluation = Evaluator(config).run(markov_run.path)

    summary = read_manifest(evaluation.output_dir / "summary.csv")
    assert {(row["system"], row["variant"]) for row in summary} == {("markov", "original"), ("markov", "quantizado")}
    assert {row["style"] for row in summary} == {"funk", "jazz"}


def test_run_with_a_different_tokenizer_is_refused(markov_run: ExperimentRun) -> None:
    config = markov_run.record.config
    other = dataclasses.replace(config, tokenizer=TokenizerSettings(num_velocities=8))

    with pytest.raises(ValueError, match="tokenizer"):
        Generation(other).run(System.MARKOV, markov_run.path, markov_run.path)


def test_missing_output_run_is_refused(markov_run: ExperimentRun, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run de saída precisa existir"):
        Generation(markov_run.record.config).run(System.MARKOV, markov_run.path, tmp_path / "nao_existe")


def test_transformer_run_without_a_checkpoint_is_refused(markov_run: ExperimentRun) -> None:
    with pytest.raises(FileNotFoundError, match="best.pt"):
        Generation(markov_run.record.config).run(System.TRANSFORMER, markov_run.path, markov_run.path)


def test_transformer_adapter_samples_from_the_best_checkpoint(tokenizer: DrumTokenizer, tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = DrumTransformer(SMALL_MODEL, tokenizer.vocab_size, 64, tokenizer.pad_id).eval()
    ModelCheckpoint.of(model, tmp_path / "tokenizer.json", epoch=1).save(tmp_path / "best.pt")
    cpu = ComputeDevice.resolve("cpu")
    requests = [
        GenerationRequest(
            plan=(BarPlan(Style.JAZZ, PieceType.GROOVE),) * 2, seed=seed, cfg_scale=2.0, temperature=1.0, top_k=5, top_p=None
        )
        for seed in (1, 2)
    ]

    generator = TransformerGenerator.load(tmp_path, tokenizer, cpu)
    generated = generator.generate(requests)

    expected = TokenSampler(model, SpecialTokens.of(tokenizer), cpu).sample(requests)
    assert [sequence.ids for sequence in generated] == expected
    assert generator.capabilities == frozenset({VariantKind.GUIDED, VariantKind.STYLE_SWITCH})
