from pathlib import Path

import pytest

from drumgen.config import ExperimentConfig, PathSettings, PieceType, Split, Style

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
