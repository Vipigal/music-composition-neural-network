import json
import shutil
from pathlib import Path

import pytest
from symusic.core import ScoreTick

from drumgen.config import ExperimentConfig, PathSettings, PieceType, PrepareSettings, Split, Style
from drumgen.prepare import Piece, Preparer, TokenizedSplit
from drumgen.tokenizer import DrumTokenizer

FIXTURES = Path(__file__).resolve().parent / "fixtures"
BAR_TICKS = 4 * 480

CATALOG_HEADER = "drummer,session,id,style,bpm,beat_type,time_signature,midi_filename,audio_filename,duration,split"
CATALOG_ROWS = [
    ("d1/s1/1", "jazz/swing", 120, "beat", "4-4", "jazz_120_beat.mid", "train"),
    ("d1/s1/2", "rock", 115, "fill", "4-4", "rock_115_fill.mid", "train"),
    ("d1/s1/3", "afrocuban/bembe", 120, "beat", "4-4", "jazz_120_beat.mid", "train"),
    ("d1/s1/4", "jazz", 120, "beat", "3-4", "jazz_120_beat.mid", "train"),
    ("d1/s2/5", "funk/groove1", 117, "beat", "4-4", "funk_117_beat.mid", "validation"),
    ("d1/s2/6", "funk", 117, "fill", "4-4", "rock_115_fill.mid", "validation"),
    ("d1/s3/7", "jazz", 120, "beat", "4-4", "jazz_120_beat.mid", "test"),
]


def write_catalog(raw_dir: Path) -> None:
    lines = [CATALOG_HEADER]
    for piece_id, style, bpm, beat_type, signature, fixture, split in CATALOG_ROWS:
        midi_path = f"{piece_id}.mid"
        (raw_dir / midi_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIXTURES / fixture, raw_dir / midi_path)
        drummer, session, _ = piece_id.split("/")
        lines.append(
            f"{drummer},{drummer}/{session},{piece_id},{style},{bpm},{beat_type},{signature},"
            f"{midi_path},,30.0,{split}"
        )
    (raw_dir / "info.csv").write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> PathSettings:
    root = tmp_path_factory.mktemp("gmd")
    raw_dir = root / "raw"
    raw_dir.mkdir()
    write_catalog(raw_dir)
    paths = PathSettings(raw_dir=raw_dir, processed_dir=root / "processed")
    Preparer(ExperimentConfig(name="test", paths=paths)).run()
    return paths


def pieces_by_id(paths: PathSettings) -> dict[str, Piece]:
    return {piece.id: piece for piece in Piece.read_csv(paths.pieces_file)}


def test_only_four_four_pieces_are_kept(prepared: PathSettings) -> None:
    assert sorted(pieces_by_id(prepared)) == ["d1/s1/1", "d1/s1/2", "d1/s1/3", "d1/s2/5", "d1/s2/6", "d1/s3/7"]


def test_pieces_carry_style_type_and_split(prepared: PathSettings) -> None:
    pieces = pieces_by_id(prepared)

    assert (pieces["d1/s1/1"].style, pieces["d1/s1/1"].type, pieces["d1/s1/1"].split) == (
        Style.JAZZ,
        PieceType.GROOVE,
        Split.TRAIN,
    )
    assert pieces["d1/s1/3"].style is Style.OTHER
    assert pieces["d1/s1/3"].genre == "afrocuban"
    assert pieces["d1/s2/6"].type is PieceType.FILL
    assert pieces["d1/s2/5"].split is Split.VAL


def test_each_split_file_holds_bos_to_eos_sequences_of_its_pieces(prepared: PathSettings) -> None:
    tokenizer = DrumTokenizer.from_file(prepared.tokenizer_file)
    pieces = pieces_by_id(prepared)

    for split in Split:
        tokenized = TokenizedSplit.load(prepared.split_file(split))
        assert sorted(tokenized.piece_ids) == sorted(p.id for p in pieces.values() if p.split is split)
        for piece_id, sequence in zip(tokenized.piece_ids, tokenized.sequences, strict=True):
            assert sequence[0] == tokenizer.bos_id
            assert sequence[-1] == tokenizer.eos_id
            assert len(sequence) == pieces[piece_id].n_tokens


def test_bar_starts_point_to_bos_or_bar_tokens_one_per_bar(prepared: PathSettings) -> None:
    tokenizer = DrumTokenizer.from_file(prepared.tokenizer_file)
    pieces = pieces_by_id(prepared)
    tokenized = TokenizedSplit.load(prepared.split_file(Split.TRAIN))

    for piece_id, sequence, starts in zip(tokenized.piece_ids, tokenized.sequences, tokenized.bar_starts, strict=True):
        assert starts[0] == 0
        assert all(sequence[index] == tokenizer.bar_id for index in starts[1:])
        assert len(starts) == pieces[piece_id].n_bars


def test_reference_windows_are_eight_bar_grooves_of_val_and_test(prepared: PathSettings) -> None:
    pieces = pieces_by_id(prepared)

    val_funk = sorted(prepared.reference_dir(Split.VAL, Style.FUNK).glob("*.mid"))
    test_jazz = sorted(prepared.reference_dir(Split.TEST, Style.JAZZ).glob("*.mid"))

    assert len(val_funk) == pieces["d1/s2/5"].n_bars // 8
    assert len(test_jazz) == pieces["d1/s3/7"].n_bars // 8
    assert val_funk[0].name == "d1-s2-5_0.mid"
    assert not (prepared.processed_dir / "reference" / "train").exists()


def test_reference_window_spans_eight_bars_from_tick_zero(prepared: PathSettings) -> None:
    window = sorted(prepared.reference_dir(Split.VAL, Style.FUNK).glob("*.mid"))[1]

    score = ScoreTick(window)

    onsets = [note.time for note in score.tracks[0].notes]
    assert score.tracks[0].is_drum is True
    assert min(onsets) >= 0
    assert max(onsets) < 8 * BAR_TICKS
    assert max(onsets) >= 7 * BAR_TICKS


def test_stats_report_vocabulary_lengths_and_counts(prepared: PathSettings) -> None:
    stats = json.loads(prepared.stats_file.read_text())
    pieces = pieces_by_id(prepared)

    assert stats["vocab_size"] == DrumTokenizer.from_file(prepared.tokenizer_file).vocab_size
    assert stats["performances"]["train"]["jazz"]["groove"] == 1
    assert stats["bars"]["val"]["funk"]["groove"] == pieces["d1/s2/5"].n_bars
    assert stats["reference_windows"]["val"]["funk"] == pieces["d1/s2/5"].n_bars // 8
    assert stats["median_bpm"]["jazz"] == 120
    assert stats["tokens_per_bar"] > 0
    assert stats["mean_tokens_per_sequence"] > 0


def test_subset_keeps_at_most_n_pieces_per_split(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_catalog(raw_dir)
    paths = PathSettings(raw_dir=raw_dir, processed_dir=tmp_path / "processed")

    Preparer(ExperimentConfig(name="t", paths=paths, prepare=PrepareSettings(pieces_per_split=1))).run()

    counts = {split: 0 for split in Split}
    for piece in Piece.read_csv(paths.pieces_file):
        counts[piece.split] += 1
    assert counts == {Split.TRAIN: 1, Split.VAL: 1, Split.TEST: 1}
