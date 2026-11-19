from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from drumgen.config import DataSettings, PathSettings, PieceType, Split, Style
from drumgen.prepare import Piece, TokenizedSplit
from drumgen.tokenizer import DrumTokenizer


@dataclass(frozen=True)
class Window:
    tokens: torch.Tensor
    targets: torch.Tensor
    styles: torch.Tensor
    types: torch.Tensor

    @classmethod
    def from_chunk(cls, chunk: Sequence[int], style: Style, piece_type: PieceType) -> Window:
        tokens = torch.tensor(chunk[:-1], dtype=torch.long)
        return cls(
            tokens=tokens,
            targets=torch.tensor(chunk[1:], dtype=torch.long),
            styles=torch.full_like(tokens, style.index),
            types=torch.full_like(tokens, piece_type.index),
        )


@dataclass(frozen=True)
class WindowBatch:
    tokens: torch.Tensor
    targets: torch.Tensor
    styles: torch.Tensor
    types: torch.Tensor

    @classmethod
    def collate(cls, windows: Sequence[Window], pad_id: int) -> WindowBatch:
        longest = max(len(window.tokens) for window in windows)
        shape = (len(windows), longest)
        tokens = torch.full(shape, pad_id, dtype=torch.long)
        targets = torch.full(shape, pad_id, dtype=torch.long)
        styles = torch.full(shape, Style.NULL.index, dtype=torch.long)
        types = torch.full(shape, PieceType.GROOVE.index, dtype=torch.long)
        for row, window in enumerate(windows):
            length = len(window.tokens)
            tokens[row, :length] = window.tokens
            targets[row, :length] = window.targets
            styles[row, :length] = window.styles
            types[row, :length] = window.types
        return cls(tokens=tokens, targets=targets, styles=styles, types=types)

    def to(self, device: torch.device) -> WindowBatch:
        return WindowBatch(
            tokens=self.tokens.to(device),
            targets=self.targets.to(device),
            styles=self.styles.to(device),
            types=self.types.to(device),
        )


@dataclass(frozen=True)
class Corpus:
    sequences: list[npt.NDArray[np.int32]]
    bar_starts: list[npt.NDArray[np.int32]]
    styles: list[Style]
    types: list[PieceType]

    @classmethod
    def load(cls, paths: PathSettings, split: Split) -> Corpus:
        tokenized = TokenizedSplit.load(paths.split_file(split))
        pieces = {piece.id: piece for piece in Piece.read_csv(paths.pieces_file)}
        return cls(
            sequences=tokenized.sequences,
            bar_starts=tokenized.bar_starts,
            styles=[pieces[piece_id].style for piece_id in tokenized.piece_ids],
            types=[pieces[piece_id].type for piece_id in tokenized.piece_ids],
        )

    @property
    def total_tokens(self) -> int:
        return sum(len(sequence) for sequence in self.sequences)


class TrainingWindows(Dataset[Window]):
    def __init__(self, corpus: Corpus, tokenizer: DrumTokenizer, settings: DataSettings, seed: int) -> None:
        self._corpus = corpus
        self._tokenizer = tokenizer
        self._settings = settings
        self._seed = seed
        self._epoch = 0
        self._probabilities = self._sequence_probabilities(corpus)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return math.ceil(self._corpus.total_tokens / self._settings.context_length)

    def __getitem__(self, index: int) -> Window:
        rng = np.random.default_rng([self._seed, self._epoch, index])
        sequence_index = int(rng.choice(len(self._corpus.sequences), p=self._probabilities))
        start = int(rng.choice(self._corpus.bar_starts[sequence_index]))
        sequence = self._corpus.sequences[sequence_index]
        chunk = sequence[start : start + self._settings.context_length + 1].tolist()
        factor = float(rng.uniform(self._settings.velocity_scale_min, self._settings.velocity_scale_max))
        chunk = self._tokenizer.scale_velocities(chunk, factor)
        style = self._corpus.styles[sequence_index]
        if rng.random() < self._settings.null_style_probability:
            style = Style.NULL
        return Window.from_chunk(chunk, style, self._corpus.types[sequence_index])

    @staticmethod
    def _sequence_probabilities(corpus: Corpus) -> np.ndarray:
        bars_per_style: dict[Style, int] = defaultdict(int)
        for style, starts in zip(corpus.styles, corpus.bar_starts, strict=True):
            bars_per_style[style] += len(starts)
        weights = np.array(
            [
                len(starts) / math.sqrt(bars_per_style[style])
                for style, starts in zip(corpus.styles, corpus.bar_starts, strict=True)
            ]
        )
        return weights / weights.sum()


class EvaluationWindows(Dataset[Window]):
    def __init__(self, corpus: Corpus, tokenizer: DrumTokenizer, settings: DataSettings) -> None:
        self._corpus = corpus
        self._pad_id = tokenizer.pad_id
        self._context = settings.context_length
        self._spans = [
            (sequence_index, start, scored_until)
            for sequence_index in range(len(corpus.sequences))
            for start, scored_until in self._spans_of(sequence_index)
        ]

    def __len__(self) -> int:
        return len(self._spans)

    def __getitem__(self, index: int) -> Window:
        sequence_index, start, scored_until = self._spans[index]
        chunk = self._corpus.sequences[sequence_index][start : start + self._context + 1].tolist()
        window = Window.from_chunk(chunk, self._corpus.styles[sequence_index], self._corpus.types[sequence_index])
        already_scored = max(0, scored_until - start)
        window.targets[:already_scored] = self._pad_id
        return window

    def _spans_of(self, sequence_index: int) -> list[tuple[int, int]]:
        length = len(self._corpus.sequences[sequence_index])
        starts = self._corpus.bar_starts[sequence_index].tolist()
        spans: list[tuple[int, int]] = []
        start, scored_until = 0, 0
        while True:
            spans.append((start, scored_until))
            end = min(start + self._context, length - 1)
            if end >= length - 1:
                return spans
            scored_until = end
            later = [bar for bar in starts if start < bar <= end]
            if not later:
                raise ValueError(f"compasso com mais de {self._context} tokens na sequência {sequence_index}")
            start = later[-1]
