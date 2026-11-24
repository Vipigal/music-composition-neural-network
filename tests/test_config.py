import dataclasses
import re
from datetime import datetime
from pathlib import Path

import pytest

from drumgen.config import (
    ConfigLoader,
    DemoSettings,
    EvaluationSettings,
    ExperimentConfig,
    ExperimentRun,
    GenerationSettings,
    GitRevision,
    MarkovSettings,
    ModelSettings,
    PathSettings,
    PieceType,
    PrepareSettings,
    RenderSettings,
    SampleFolder,
    SamplingPoint,
    Split,
    Style,
    StyleSwitch,
    System,
    TokenizerSettings,
    VariantKind,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("gmd_style", "expected"),
    [
        ("jazz/funk", Style.JAZZ),
        ("funk/groove1", Style.FUNK),
        ("hiphop", Style.HIPHOP),
        ("soul/motown", Style.SOUL),
        ("afrocuban/bembe", Style.OTHER),
        ("neworleans/secondline", Style.OTHER),
    ],
)
def test_style_comes_from_first_segment_of_gmd_column(gmd_style: str, expected: Style) -> None:
    assert Style.from_gmd(gmd_style) is expected


def test_evaluated_styles_are_the_six_of_d3() -> None:
    assert [style.value for style in Style.evaluated()] == [
        "rock",
        "funk",
        "latin",
        "jazz",
        "hiphop",
        "soul",
    ]


def test_style_embedding_has_eight_values() -> None:
    assert len(Style) == 8
    assert sorted(style.index for style in Style) == list(range(8))


@pytest.mark.parametrize(("beat_type", "expected"), [("beat", PieceType.GROOVE), ("fill", PieceType.FILL)])
def test_piece_type_comes_from_gmd_beat_type(beat_type: str, expected: PieceType) -> None:
    assert PieceType.from_gmd(beat_type) is expected


def test_debug_config_loads_with_d2_tokenizer_defaults() -> None:
    config = ExperimentConfig.from_yaml(REPO_ROOT / "configs" / "debug.yaml")

    assert config.name == "debug"
    assert config.tokenizer.use_microtiming is True
    assert config.tokenizer.num_microtiming_bins == 16
    assert config.tokenizer.max_microtiming_shift == 0.125
    assert config.tokenizer.num_velocities == 16
    assert config.paths.raw_dir == REPO_ROOT / "data" / "raw" / "groove"


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("name: x\ntokenizer:\n  num_bins: 3\n")

    with pytest.raises(ValueError, match="num_bins"):
        ExperimentConfig.from_yaml(config_file)


def test_wrong_type_is_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("name: x\ntokenizer:\n  num_velocities: muitas\n")

    with pytest.raises(TypeError, match="num_velocities"):
        ExperimentConfig.from_yaml(config_file)


@pytest.mark.parametrize(
    ("gmd_split", "expected"), [("train", Split.TRAIN), ("validation", Split.VAL), ("test", Split.TEST)]
)
def test_split_comes_from_gmd_split_column(gmd_split: str, expected: Split) -> None:
    assert Split.from_gmd(gmd_split) is expected


def test_debug_config_selects_a_subset_into_its_own_folder() -> None:
    config = ExperimentConfig.from_yaml(REPO_ROOT / "configs" / "debug.yaml")

    assert config.prepare.pieces_per_split is not None
    assert config.paths.processed_dir != PathSettings().processed_dir


def test_full_prepare_has_no_subset() -> None:
    assert ExperimentConfig(name="x").prepare.pieces_per_split is None


def test_optional_int_accepts_null_and_int(tmp_path: Path) -> None:
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("name: x\nprepare:\n  pieces_per_split: 4\n")

    assert ExperimentConfig.from_yaml(config_file).prepare.pieces_per_split == 4


def test_processed_layout_follows_the_contract(tmp_path: Path) -> None:
    paths = PathSettings(processed_dir=tmp_path)

    assert paths.tokenizer_file == tmp_path / "tokenizer.json"
    assert paths.pieces_file == tmp_path / "pieces.csv"
    assert paths.stats_file == tmp_path / "stats.json"
    assert paths.split_file(Split.VAL) == tmp_path / "val.npz"
    assert paths.reference_dir(Split.TEST, Style.JAZZ) == tmp_path / "reference" / "test" / "jazz"


def test_cli_reads_the_config_flag() -> None:
    config = ExperimentConfig.from_cli("teste", ["--config", str(REPO_ROOT / "configs" / "debug.yaml")])

    assert config.name == "debug"


def test_tool_sections_default_to_the_spec() -> None:
    config = ExperimentConfig(name="x")

    assert config.markov.candidate_orders == [1, 2, 3, 4, 6, 8, 12, 16]
    assert config.markov.order is None
    assert config.generation.samples_per_style == 100
    assert config.generation.first_seed == 0
    assert config.generation.temperature == 1.0
    assert config.generation.top_k is None
    assert config.generation.top_p is None
    assert config.generation.cfg_scale == 3.0
    assert config.generation.cfg_scale_sweep == []
    assert config.generation.temperature_sweep == []
    assert config.generation.top_p_sweep == []
    assert config.generation.style_switches == [StyleSwitch(before=Style.ROCK, after=Style.FUNK)]
    assert config.generation.fill_styles == []
    assert config.generation.batch_size == 100
    assert config.render.soundfont == Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")
    assert config.render.sample_rate == 22050
    assert config.evaluation.reference_split is Split.VAL
    assert config.evaluation.output_dir is None
    assert config.demo.final_run is None
    assert config.demo.sweep_run is None
    assert config.demo.cfg_scales == [1.0, 1.5, 3.0]
    assert config.demo.style_switches == [
        StyleSwitch(before=Style.ROCK, after=Style.FUNK),
        StyleSwitch(before=Style.JAZZ, after=Style.HIPHOP),
    ]
    assert config.demo.output_dir == REPO_ROOT


def test_model_and_training_default_to_p1() -> None:
    config = ExperimentConfig(name="x")

    assert config.model == ModelSettings(d_model=256, layers=4, heads=4, feedforward=1024, dropout=0.1)
    assert config.training.batch_size == 32
    assert config.training.learning_rate == 3e-4
    assert config.training.weight_decay == 0.01
    assert config.training.warmup_steps == 300
    assert config.training.gradient_clip == 1.0
    assert config.training.device == "cuda"


def test_model_and_training_are_overridden_from_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("name: x\nmodel:\n  layers: 2\ntraining:\n  max_epochs: 3\n  device: cpu\n")

    config = ExperimentConfig.from_yaml(config_file)

    assert config.model.layers == 2
    assert config.training.max_epochs == 3
    assert config.training.device == "cpu"


def test_tool_sections_are_overridden_from_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(
        "name: x\n"
        "markov:\n  candidate_orders: [2, 3]\n  order: 3\n"
        "generation:\n"
        "  top_k: 32\n  top_p: 0.9\n  cfg_scale_sweep: [1, 2]\n"
        "  temperature_sweep: [0.9, 1]\n  top_p_sweep: [0.9, 1]\n"
        "  style_switches:\n    - {before: jazz, after: soul}\n"
        "  fill_styles: [jazz, funk]\n"
        "render:\n  soundfont: sf/drums.sf2\n"
        "evaluation:\n  reference_split: test\n  output_dir: results\n"
        "demo:\n  final_run: runs/final\n  sweep_run: /x/sweep\n  cfg_scales: [1, 3]\n"
        "  style_switches:\n    - {before: funk, after: latin}\n  output_dir: entrega\n"
    )

    config = ExperimentConfig.from_yaml(config_file)

    assert config.markov.candidate_orders == [2, 3]
    assert config.markov.order == 3
    assert config.generation.top_k == 32
    assert config.generation.top_p == 0.9
    assert config.generation.cfg_scale_sweep == [1.0, 2.0]
    assert config.generation.temperature_sweep == [0.9, 1.0]
    assert config.generation.top_p_sweep == [0.9, 1.0]
    assert config.generation.style_switches == [StyleSwitch(before=Style.JAZZ, after=Style.SOUL)]
    assert config.generation.fill_styles == [Style.JAZZ, Style.FUNK]
    assert config.render.soundfont == REPO_ROOT / "sf" / "drums.sf2"
    assert config.evaluation.reference_split is Split.TEST
    assert config.evaluation.output_dir == REPO_ROOT / "results"
    assert config.demo.final_run == REPO_ROOT / "runs" / "final"
    assert config.demo.sweep_run == Path("/x/sweep")
    assert config.demo.cfg_scales == [1.0, 3.0]
    assert config.demo.style_switches == [StyleSwitch(before=Style.FUNK, after=Style.LATIN)]
    assert config.demo.output_dir == REPO_ROOT / "entrega"


@pytest.mark.parametrize(
    ("body", "where"),
    [
        ("evaluation:\n  reference_split: validation\n", "evaluation.reference_split"),
        ("generation:\n  fill_styles: [jazz, polka]\n", r"generation.fill_styles\[1\]"),
        ("markov:\n  candidate_orders: [2, tres]\n", r"markov.candidate_orders\[1\]"),
        ("markov:\n  candidate_orders: 3\n", "markov.candidate_orders"),
        ("generation:\n  style_switches: [jazz]\n", r"generation.style_switches\[0\]"),
        ("generation:\n  style_switches:\n    - {before: jazz, after: 7}\n", r"style_switches\[0\].after"),
    ],
)
def test_wrong_type_in_tool_sections_names_the_key(tmp_path: Path, body: str, where: str) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("name: x\n" + body)

    with pytest.raises(TypeError, match=where):
        ExperimentConfig.from_yaml(config_file)


def test_unknown_key_inside_a_list_is_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("name: x\ngeneration:\n  style_switches:\n    - {before: jazz, middle: rock}\n")

    with pytest.raises(ValueError, match=r"style_switches\[0\].middle"):
        ExperimentConfig.from_yaml(config_file)


def test_dump_writes_enums_by_value_and_paths_relative_to_the_project(tmp_path: Path) -> None:
    config = ExperimentConfig(
        name="x",
        paths=PathSettings(runs_dir=tmp_path / "runs"),
        generation=GenerationSettings(fill_styles=Style.evaluated()),
    )

    data = ConfigLoader.dump(config)

    paths = data["paths"]
    evaluation = data["evaluation"]
    generation = data["generation"]
    assert isinstance(paths, dict) and isinstance(evaluation, dict) and isinstance(generation, dict)
    assert paths["raw_dir"] == "data/raw/groove"
    assert paths["runs_dir"] == str(tmp_path / "runs")
    assert evaluation["reference_split"] == "val"
    assert generation["style_switches"] == [{"before": "rock", "after": "funk"}]
    assert generation["fill_styles"] == ["rock", "funk", "latin", "jazz", "hiphop", "soul"]


@pytest.mark.parametrize("config_name", ["debug.yaml", "transformer.yaml", "eval.yaml", "sweep.yaml", "demo.yaml"])
def test_run_round_trip_preserves_the_repo_configs(tmp_path: Path, config_name: str) -> None:
    loaded = ExperimentConfig.from_yaml(REPO_ROOT / "configs" / config_name)
    config = dataclasses.replace(loaded, paths=dataclasses.replace(loaded.paths, runs_dir=tmp_path))

    run = ExperimentRun.create(config)

    assert ExperimentRun.open(run.path).record.config == config


def test_run_round_trip_preserves_every_override(tmp_path: Path) -> None:
    config = ExperimentConfig(
        name="tudo",
        seed=7,
        paths=PathSettings(processed_dir=REPO_ROOT / "outro", runs_dir=tmp_path),
        prepare=PrepareSettings(pieces_per_split=3, reference_bars=4),
        markov=MarkovSettings(candidate_orders=[2, 5], order=5),
        generation=GenerationSettings(
            top_k=10,
            top_p=0.8,
            cfg_scale_sweep=[1.0, 4.5],
            temperature_sweep=[0.8],
            top_p_sweep=[0.9, 1.0],
            style_switches=[StyleSwitch(before=Style.HIPHOP, after=Style.LATIN)],
            fill_styles=[Style.SOUL],
        ),
        render=RenderSettings(sample_rate=44100),
        evaluation=EvaluationSettings(reference_split=Split.TEST, output_dir=REPO_ROOT / "results"),
        demo=DemoSettings(
            final_run=tmp_path / "final",
            sweep_run=REPO_ROOT / "runs" / "varredura",
            cfg_scales=[1.0, 2.0],
            style_switches=[StyleSwitch(before=Style.SOUL, after=Style.ROCK)],
            output_dir=tmp_path / "entrega",
        ),
    )

    run = ExperimentRun.create(config)

    assert ExperimentRun.open(run.path).record.config == config


def test_git_revision_reads_the_project_head() -> None:
    revision = GitRevision.of(REPO_ROOT)

    assert re.fullmatch(r"[0-9a-f]{40}", revision.commit)


def test_git_revision_outside_a_repo_is_empty(tmp_path: Path) -> None:
    assert GitRevision.of(tmp_path) == GitRevision(commit="", dirty=False)


def test_run_folder_holds_the_resolved_config_and_the_commit(tmp_path: Path) -> None:
    config = ExperimentConfig(name="debug", paths=PathSettings(runs_dir=tmp_path / "runs"))

    run = ExperimentRun.create(config, now=datetime(2026, 11, 19, 14, 3))

    assert run.path == tmp_path / "runs" / "20261119-1403_debug"
    assert run.config_file.is_file()
    assert run.record.revision == GitRevision.of(REPO_ROOT)
    assert ExperimentRun.open(run.path) == run


def test_run_in_the_same_minute_does_not_overwrite(tmp_path: Path) -> None:
    now = datetime(2026, 11, 19, 14, 3)
    first = ExperimentRun.create(ExperimentConfig(name="a", paths=PathSettings(runs_dir=tmp_path)), now=now)
    original = first.config_file.read_text()

    second = ExperimentRun.create(ExperimentConfig(name="a", seed=9, paths=PathSettings(runs_dir=tmp_path)), now=now)

    assert second.path == tmp_path / "20261119-1403_a_2"
    assert first.config_file.read_text() == original
    assert ExperimentRun.open(second.path).record.config.seed == 9


def test_saving_a_run_keeps_its_revision(tmp_path: Path) -> None:
    run = ExperimentRun.create(ExperimentConfig(name="m", paths=PathSettings(runs_dir=tmp_path)))
    chosen = dataclasses.replace(run.record.config, markov=MarkovSettings(order=4))

    dataclasses.replace(run, record=dataclasses.replace(run.record, config=chosen)).save()

    reopened = ExperimentRun.open(run.path)
    assert reopened.record.config.markov.order == 4
    assert reopened.record.revision == run.record.revision


def test_run_lays_out_samples_per_system(tmp_path: Path) -> None:
    run = ExperimentRun.create(ExperimentConfig(name="m", paths=PathSettings(runs_dir=tmp_path)))

    assert run.samples_dir == run.path / "samples"
    assert run.system_samples_dir(System.MARKOV) == run.path / "samples" / "markov"


def test_run_is_compatible_with_a_config_of_the_same_tokenizer_and_bars(tmp_path: Path) -> None:
    config = ExperimentConfig(name="m", paths=PathSettings(runs_dir=tmp_path))
    run = ExperimentRun.create(config)

    run.check_compatible(dataclasses.replace(config, name="outra", seed=7))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"prepare": PrepareSettings(reference_bars=4)}, "8 compassos, config com 4"),
        ({"tokenizer": TokenizerSettings(num_velocities=8)}, "tokenizer da run difere"),
    ],
)
def test_run_with_other_bars_or_tokenizer_is_incompatible(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    config = ExperimentConfig(name="m", paths=PathSettings(runs_dir=tmp_path))
    run = ExperimentRun.create(config)

    with pytest.raises(ValueError, match=message):
        run.check_compatible(dataclasses.replace(config, **change))


def test_run_file_with_unknown_section_is_rejected(tmp_path: Path) -> None:
    run = ExperimentRun.create(ExperimentConfig(name="m", paths=PathSettings(runs_dir=tmp_path)))
    run.config_file.write_text(run.config_file.read_text() + "extra: 1\n")

    with pytest.raises(ValueError, match="extra"):
        ExperimentRun.open(run.path)


def test_run_file_with_wrong_type_names_the_key(tmp_path: Path) -> None:
    run = ExperimentRun.create(ExperimentConfig(name="m", paths=PathSettings(runs_dir=tmp_path)))
    run.config_file.write_text(run.config_file.read_text().replace("dirty: ", "dirty: talvez #"))

    with pytest.raises(TypeError, match="revision.dirty"):
        ExperimentRun.open(run.path)


@pytest.mark.parametrize(
    ("name", "folder"),
    [
        ("rock", SampleFolder.original(Style.ROCK)),
        ("jazz_quantizado", SampleFolder.quantized(Style.JAZZ)),
        ("funk_cfg1.5_t0.9_p0.9", SampleFolder.guided(Style.FUNK, SamplingPoint.of(1.5, 0.9, 0.9))),
        ("soul_cfg3_t1_p1", SampleFolder.guided(Style.SOUL, SamplingPoint.of(3.0, 1.0, None))),
        ("troca_rock-funk", SampleFolder.switch(StyleSwitch(before=Style.ROCK, after=Style.FUNK))),
        ("virada_hiphop", SampleFolder.fill(Style.HIPHOP)),
    ],
)
def test_folder_name_round_trips(name: str, folder: SampleFolder) -> None:
    assert SampleFolder.parse(name) == folder
    assert folder.name == name


@pytest.mark.parametrize("cfg_scale", [1.0, 1.5, 3.0, 0.25])
@pytest.mark.parametrize("temperature", [0.9, 1.0, 1.25])
@pytest.mark.parametrize("top_p", [0.9, 0.95, None])
def test_guided_folder_name_preserves_the_sampling_point(
    cfg_scale: float, temperature: float, top_p: float | None
) -> None:
    point = SamplingPoint.of(cfg_scale, temperature, top_p)

    parsed = SampleFolder.parse(SampleFolder.guided(Style.JAZZ, point).name).point

    assert parsed == point
    assert parsed is not None
    assert (parsed.cfg_scale, parsed.temperature, parsed.top_p) == (cfg_scale, temperature, top_p)


def test_top_p_of_one_is_the_same_point_as_no_top_p() -> None:
    assert SamplingPoint.of(1.5, 0.9, 1.0) == SamplingPoint.of(1.5, 0.9, None)
    assert SampleFolder.guided(Style.ROCK, SamplingPoint.of(1.5, 0.9, 1.0)).name == "rock_cfg1.5_t0.9_p1"


@pytest.mark.parametrize(
    ("name", "variant", "evaluated"),
    [
        ("rock", "original", True),
        ("rock_quantizado", "quantizado", True),
        ("rock_cfg1.5_t0.9_p1", "cfg1.5_t0.9_p1", True),
        ("troca_rock-funk", "troca_rock-funk", False),
        ("virada_rock", "virada", False),
    ],
)
def test_folder_variant_and_whether_it_has_a_reference(name: str, variant: str, evaluated: bool) -> None:
    folder = SampleFolder.parse(name)

    assert folder.variant == variant
    assert folder.kind.has_reference is evaluated


@pytest.mark.parametrize(
    "name",
    ["polka", "other", "null", "rock_lento", "troca_rock", "rock_cfg", "virada_", "rock_cfg1.5", "rock_cfg1.5_t0.9"],
)
def test_unknown_folder_name_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        SampleFolder.parse(name)


def test_kind_of_each_folder() -> None:
    assert SampleFolder.parse("rock_cfg3_t0.9_p0.9").kind is VariantKind.GUIDED
    assert SampleFolder.parse("rock_cfg3_t0.9_p0.9").point == SamplingPoint(cfg_scale=3.0, temperature=0.9, top_p=0.9)
    assert SampleFolder.parse("rock").point is None
    assert SampleFolder.parse("troca_jazz-soul").style_switch == StyleSwitch(before=Style.JAZZ, after=Style.SOUL)


def test_selection_styles_are_the_ones_with_validation_of_p2() -> None:
    assert Style.selection() == [Style.ROCK, Style.FUNK, Style.LATIN, Style.JAZZ]


def test_sweep_points_are_the_product_of_the_three_sweeps() -> None:
    settings = GenerationSettings(cfg_scale_sweep=[1.0, 3.0], temperature_sweep=[0.9, 1.0], top_p_sweep=[0.9, 1.0])

    assert settings.sweep_points == [
        SamplingPoint.of(cfg_scale, temperature, top_p)
        for cfg_scale in (1.0, 3.0)
        for temperature in (0.9, 1.0)
        for top_p in (0.9, None)
    ]


def test_empty_temperature_and_top_p_sweeps_use_the_single_values() -> None:
    settings = GenerationSettings(temperature=0.95, top_p=0.8, cfg_scale_sweep=[1.5, 3.0])

    assert settings.sweep_points == [SamplingPoint.of(1.5, 0.95, 0.8), SamplingPoint.of(3.0, 0.95, 0.8)]


def test_without_a_guidance_sweep_there_are_no_sweep_points() -> None:
    assert GenerationSettings(temperature_sweep=[0.9, 1.0], top_p_sweep=[0.9]).sweep_points == []


def test_sweep_config_has_the_grid_of_d18() -> None:
    generation = ExperimentConfig.from_yaml(REPO_ROOT / "configs" / "sweep.yaml").generation

    assert generation.samples_per_style == 100
    assert generation.cfg_scale_sweep == [1.0, 1.5, 3.0]
    assert generation.temperature_sweep == [0.9, 1.0]
    assert generation.top_p_sweep == [0.9, 1.0]
    assert len(generation.sweep_points) == 12
