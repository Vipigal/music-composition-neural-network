from __future__ import annotations

import csv
import json
import random
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
import numpy.typing as npt
from symusic.core import ScoreTick
from tqdm import tqdm

from drumgen.config import ExperimentConfig, PathSettings, PieceType, Split, Style
from drumgen.tokenizer import DrumTokenizer

FOUR_FOUR = "4-4"

PIECE_COLUMNS = ["id", "split", "genre", "style", "type", "drummer", "session", "bpm", "n_bars", "n_tokens"]


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    split: Split
    gmd_style: str
    type: PieceType
    drummer: str
    session: str
    bpm: int
    time_signature: str
    midi_filename: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> CatalogEntry:
        return cls(
            id=row["id"],
            split=Split.from_gmd(row["split"]),
            gmd_style=row["style"],
            type=PieceType.from_gmd(row["beat_type"]),
            drummer=row["drummer"],
            session=row["session"],
            bpm=int(row["bpm"]),
            time_signature=row["time_signature"],
            midi_filename=row["midi_filename"],
        )


@dataclass(frozen=True)
class Piece:
    id: str
    split: Split
    genre: str
    style: Style
    type: PieceType
    drummer: str
    session: str
    bpm: int
    n_bars: int
    n_tokens: int

    @property
    def file_stem(self) -> str:
        return self.id.replace("/", "-")

    @classmethod
    def write_csv(cls, pieces: Sequence[Piece], path: Path) -> None:
        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(PIECE_COLUMNS)
            for piece in pieces:
                writer.writerow(
                    [
                        piece.id,
                        piece.split.value,
                        piece.genre,
                        piece.style.value,
                        piece.type.value,
                        piece.drummer,
                        piece.session,
                        piece.bpm,
                        piece.n_bars,
                        piece.n_tokens,
                    ]
                )

    @classmethod
    def read_csv(cls, path: Path) -> list[Piece]:
        with path.open(newline="") as file:
            return [
                cls(
                    id=row["id"],
                    split=Split(row["split"]),
                    genre=row["genre"],
                    style=Style(row["style"]),
                    type=PieceType(row["type"]),
                    drummer=row["drummer"],
                    session=row["session"],
                    bpm=int(row["bpm"]),
                    n_bars=int(row["n_bars"]),
                    n_tokens=int(row["n_tokens"]),
                )
                for row in csv.DictReader(file)
            ]


@dataclass(frozen=True)
class TokenizedSplit:
    piece_ids: list[str]
    sequences: list[npt.NDArray[np.int32]]
    bar_starts: list[npt.NDArray[np.int32]]

    def save(self, path: Path) -> None:
        tokens, token_offsets = self._pack(self.sequences)
        starts, start_offsets = self._pack(self.bar_starts)
        np.savez_compressed(
            path,
            piece_ids=np.array(self.piece_ids, dtype=str),
            tokens=tokens,
            token_offsets=token_offsets,
            bar_starts=starts,
            bar_offsets=start_offsets,
        )

    @classmethod
    def load(cls, path: Path) -> TokenizedSplit:
        with np.load(path, allow_pickle=False) as archive:
            return cls(
                piece_ids=[str(piece_id) for piece_id in archive["piece_ids"]],
                sequences=cls._unpack(archive["tokens"], archive["token_offsets"]),
                bar_starts=cls._unpack(archive["bar_starts"], archive["bar_offsets"]),
            )

    @staticmethod
    def _pack(arrays: Sequence[npt.NDArray[np.int32]]) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]]:
        offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(array) for array in arrays])
        flat = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.int32)
        return flat.astype(np.int32), offsets

    @staticmethod
    def _unpack(flat: npt.NDArray[np.int32], offsets: npt.NDArray[np.int64]) -> list[npt.NDArray[np.int32]]:
        return [flat[start:end] for start, end in zip(offsets[:-1], offsets[1:], strict=True)]


class Preparer:
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._paths: PathSettings = config.paths
        self._tokenizer = DrumTokenizer.from_settings(config.tokenizer)

    def run(self) -> dict[str, object]:
        self._paths.processed_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(self._paths.tokenizer_file)

        pieces: list[Piece] = []
        sequences: dict[str, list[int]] = {}
        skipped: list[str] = []
        for entry in tqdm(self._catalog_entries(), desc="tokenizando"):
            tokenized = self._tokenize(entry)
            if tokenized is None:
                skipped.append(entry.id)
                continue
            piece, sequence = tokenized
            pieces.append(piece)
            sequences[piece.id] = sequence

        Piece.write_csv(pieces, self._paths.pieces_file)
        for split in Split:
            self._split_of(pieces, sequences, split).save(self._paths.split_file(split))
        windows = self._write_references(pieces, sequences)

        stats = self._stats(pieces, windows, skipped)
        self._paths.stats_file.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
        return stats

    def _catalog_entries(self) -> list[CatalogEntry]:
        with self._paths.catalog_file.open(newline="") as file:
            entries = [CatalogEntry.from_row(row) for row in csv.DictReader(file)]
        entries = sorted((entry for entry in entries if entry.time_signature == FOUR_FOUR), key=lambda e: e.id)
        subset = self._config.prepare.pieces_per_split
        if subset is None:
            return entries
        random.Random(self._config.seed).shuffle(entries)
        kept: list[CatalogEntry] = []
        for split in Split:
            kept += [entry for entry in entries if entry.split is split][:subset]
        return sorted(kept, key=lambda entry: entry.id)

    def _tokenize(self, entry: CatalogEntry) -> tuple[Piece, list[int]] | None:
        ids = self._tokenizer.encode(ScoreTick(self._paths.raw_dir / entry.midi_filename))
        if not ids:
            return None
        sequence = [self._tokenizer.bos_id, *ids, self._tokenizer.eos_id]
        piece = Piece(
            id=entry.id,
            split=entry.split,
            genre=Style.genre_of(entry.gmd_style),
            style=Style.from_gmd(entry.gmd_style),
            type=entry.type,
            drummer=entry.drummer,
            session=entry.session,
            bpm=entry.bpm,
            n_bars=self._tokenizer.count_bars(sequence),
            n_tokens=len(sequence),
        )
        return piece, sequence

    def _split_of(self, pieces: Sequence[Piece], sequences: dict[str, list[int]], split: Split) -> TokenizedSplit:
        selected = [piece for piece in pieces if piece.split is split]
        return TokenizedSplit(
            piece_ids=[piece.id for piece in selected],
            sequences=[np.array(sequences[piece.id], dtype=np.int32) for piece in selected],
            bar_starts=[
                np.array(self._tokenizer.bar_starts(sequences[piece.id]), dtype=np.int32) for piece in selected
            ],
        )

    def _write_references(
        self, pieces: Sequence[Piece], sequences: dict[str, list[int]]
    ) -> dict[str, dict[str, int]]:
        shutil.rmtree(self._paths.reference_root, ignore_errors=True)
        bars = self._config.prepare.reference_bars
        evaluated = Style.evaluated()
        windows = {split.value: {style.value: 0 for style in evaluated} for split in (Split.VAL, Split.TEST)}
        for piece in pieces:
            if piece.split is Split.TRAIN or piece.type is not PieceType.GROOVE or piece.style not in evaluated:
                continue
            sequence = sequences[piece.id]
            starts = self._tokenizer.bar_starts(sequence)
            target_dir = self._paths.reference_dir(piece.split, piece.style)
            target_dir.mkdir(parents=True, exist_ok=True)
            for window in range(len(starts) // bars):
                first = window * bars
                end = starts[first + bars] if first + bars < len(starts) else len(sequence) - 1
                content = sequence[starts[first] + 1 : end]
                score = self._tokenizer.decode(content)
                score.dump_midi(target_dir / f"{piece.file_stem}_{window}.mid")
                windows[piece.split.value][piece.style.value] += 1
        return windows

    def _stats(
        self, pieces: Sequence[Piece], windows: dict[str, dict[str, int]], skipped: Sequence[str]
    ) -> dict[str, object]:
        styles = [style.value for style in Style if style is not Style.NULL]
        types = [piece_type.value for piece_type in PieceType]
        performances = {s.value: {st: {t: 0 for t in types} for st in styles} for s in Split}
        bars = {s.value: {st: {t: 0 for t in types} for st in styles} for s in Split}
        for piece in pieces:
            performances[piece.split.value][piece.style.value][piece.type.value] += 1
            bars[piece.split.value][piece.style.value][piece.type.value] += piece.n_bars

        train_grooves = [p for p in pieces if p.split is Split.TRAIN and p.type is PieceType.GROOVE]
        median_bpm = {
            style.value: median(p.bpm for p in train_grooves if p.style is style)
            for style in Style.evaluated()
            if any(p.style is style for p in train_grooves)
        }
        content_tokens = sum(piece.n_tokens - 2 for piece in pieces)
        total_bars = sum(piece.n_bars for piece in pieces)
        return {
            "vocab_size": self._tokenizer.vocab_size,
            "mean_tokens_per_sequence": content_tokens / len(pieces) if pieces else 0.0,
            "tokens_per_bar": content_tokens / total_bars if total_bars else 0.0,
            "totals": {
                "grooves": sum(1 for p in pieces if p.type is PieceType.GROOVE),
                "fills": sum(1 for p in pieces if p.type is PieceType.FILL),
                "train_groove_bars": sum(p.n_bars for p in train_grooves),
                "train_tokens": sum(p.n_tokens for p in pieces if p.split is Split.TRAIN),
                "skipped": list(skipped),
            },
            "performances": performances,
            "bars": bars,
            "median_bpm": median_bpm,
            "reference_windows": windows,
        }


if __name__ == "__main__":
    Preparer(ExperimentConfig.from_cli("Tokeniza o GMD e grava data/processed")).run()
