from __future__ import annotations

import csv
import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import numpy.typing as npt

from drumgen.config import ExperimentConfig, ExperimentRun, PieceType, Split, Style
from drumgen.dataset import Corpus
from drumgen.tokenizer import DrumTokenizer

SELECTION_STYLES = [Style.ROCK, Style.FUNK, Style.LATIN, Style.JAZZ]
DISCOUNT_FALLBACK = 0.5

Hashes = npt.NDArray[np.uint64]
Counts = npt.NDArray[np.int64]
Tokens = npt.NDArray[np.int64]
Probabilities = npt.NDArray[np.float64]


class NgramHasher:
    PRIME = np.uint64(0x9E3779B97F4A7C15)
    SEED = np.uint64(0x243F6A8885A308D3)
    MIX_FIRST = np.uint64(0xBF58476D1CE4E5B9)
    MIX_SECOND = np.uint64(0x94D049BB133111EB)

    @classmethod
    def contexts(cls, tokens: Tokens, order: int) -> Hashes:
        length = len(tokens)
        hashes = np.zeros((order + 1, length), dtype=np.uint64)
        hashes[0] = cls.SEED
        for size in range(1, min(order, length - 1) + 1):
            hashes[size, size:] = cls.extend(hashes[size - 1, size:], tokens[: length - size])
        return hashes

    @classmethod
    def extend(cls, hashes: Hashes, tokens: Tokens) -> Hashes:
        mixed = hashes * cls.PRIME + tokens.astype(np.uint64) + np.uint64(1)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * cls.MIX_FIRST
        mixed = (mixed ^ (mixed >> np.uint64(27))) * cls.MIX_SECOND
        return mixed ^ (mixed >> np.uint64(31))

    @staticmethod
    def joined(parts: Sequence[Hashes]) -> Hashes:
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint64)


@dataclass(frozen=True)
class NgramOccurrences:
    contexts: Hashes
    ngrams: Hashes
    extended: Hashes
    at_start: npt.NDArray[np.bool_]

    @classmethod
    def of_level(cls, tokens: Tokens, hashes: Hashes, size: int) -> NgramOccurrences:
        start = max(size, 1)
        contexts = hashes[size, start:]
        ngrams = NgramHasher.extend(contexts, tokens[start:])
        positions = np.arange(start, len(tokens))
        at_start = positions == size
        left = tokens[np.maximum(positions - size - 1, 0)]
        return cls(contexts=contexts, ngrams=ngrams, extended=NgramHasher.extend(ngrams, left), at_start=at_start)

    @classmethod
    def joined(cls, parts: Sequence[NgramOccurrences]) -> NgramOccurrences:
        return cls(
            contexts=NgramHasher.joined([part.contexts for part in parts]),
            ngrams=NgramHasher.joined([part.ngrams for part in parts]),
            extended=NgramHasher.joined([part.extended for part in parts]),
            at_start=np.concatenate([part.at_start for part in parts]) if parts else np.zeros(0, dtype=bool),
        )


@dataclass(frozen=True)
class DiscountedCounts:
    context_totals: Counts
    ngram_counts: Counts
    discount: float

    @classmethod
    def of(cls, contexts: Hashes, ngrams: Hashes) -> DiscountedCounts:
        _, context_totals = np.unique(contexts, return_counts=True)
        _, ngram_counts = np.unique(ngrams, return_counts=True)
        once = int(np.count_nonzero(ngram_counts == 1))
        twice = int(np.count_nonzero(ngram_counts == 2))
        return cls(
            context_totals=context_totals.astype(np.int64),
            ngram_counts=ngram_counts.astype(np.int64),
            discount=once / (once + 2 * twice) if once else DISCOUNT_FALLBACK,
        )


class CountKind(Enum):
    RAW = "raw"
    CONTINUATION = "continuation"

    @classmethod
    def at(cls, size: int, order: int) -> CountKind:
        return cls.RAW if size == order else cls.CONTINUATION


@dataclass(frozen=True)
class NgramLevel:
    context_hashes: Hashes
    context_types: Counts
    ngram_hashes: Hashes
    counts: dict[CountKind, DiscountedCounts]

    @classmethod
    def count(cls, occurrences: NgramOccurrences) -> NgramLevel:
        ngram_hashes, first_seen = np.unique(occurrences.ngrams, return_index=True)
        context_hashes, context_types = np.unique(occurrences.contexts[first_seen], return_counts=True)
        inner = ~occurrences.at_start
        _, distinct_left = np.unique(occurrences.extended[inner], return_index=True)
        continued = np.flatnonzero(inner)[distinct_left]
        continuations = np.concatenate([continued, np.flatnonzero(occurrences.at_start)])
        return cls(
            context_hashes=context_hashes,
            context_types=context_types.astype(np.int64),
            ngram_hashes=ngram_hashes,
            counts={
                CountKind.RAW: DiscountedCounts.of(occurrences.contexts, occurrences.ngrams),
                CountKind.CONTINUATION: DiscountedCounts.of(
                    occurrences.contexts[continuations], occurrences.ngrams[continuations]
                ),
            },
        )

    def context_statistics(self, contexts: Hashes, kind: CountKind) -> tuple[Counts, Counts]:
        return (
            self._lookup(self.context_hashes, self.counts[kind].context_totals, contexts),
            self._lookup(self.context_hashes, self.context_types, contexts),
        )

    def ngram_count(self, ngrams: Hashes, kind: CountKind) -> Counts:
        return self._lookup(self.ngram_hashes, self.counts[kind].ngram_counts, ngrams)

    def arrays(self, prefix: str) -> dict[str, npt.NDArray[np.generic]]:
        arrays: dict[str, npt.NDArray[np.generic]] = {
            f"{prefix}_context_hashes": self.context_hashes,
            f"{prefix}_context_types": self.context_types,
            f"{prefix}_ngram_hashes": self.ngram_hashes,
        }
        for kind, counts in self.counts.items():
            arrays[f"{prefix}_{kind.value}_context_totals"] = counts.context_totals
            arrays[f"{prefix}_{kind.value}_ngram_counts"] = counts.ngram_counts
            arrays[f"{prefix}_{kind.value}_discount"] = np.array(counts.discount)
        return arrays

    @classmethod
    def from_arrays(cls, archive: np.lib.npyio.NpzFile, prefix: str) -> NgramLevel:
        return cls(
            context_hashes=archive[f"{prefix}_context_hashes"],
            context_types=archive[f"{prefix}_context_types"],
            ngram_hashes=archive[f"{prefix}_ngram_hashes"],
            counts={
                kind: DiscountedCounts(
                    context_totals=archive[f"{prefix}_{kind.value}_context_totals"],
                    ngram_counts=archive[f"{prefix}_{kind.value}_ngram_counts"],
                    discount=float(archive[f"{prefix}_{kind.value}_discount"]),
                )
                for kind in CountKind
            },
        )

    @staticmethod
    def _lookup(table: Hashes, values: Counts, keys: Hashes) -> Counts:
        if not len(table):
            return np.zeros(len(keys), dtype=np.int64)
        index = np.minimum(np.searchsorted(table, keys), len(table) - 1)
        return np.where(table[index] == keys, values[index], 0)


@dataclass(frozen=True)
class KneserNeyModel:
    levels: list[NgramLevel]
    base: Probabilities

    @property
    def order(self) -> int:
        return len(self.levels) - 1

    @classmethod
    def fit(cls, sequences: Sequence[npt.NDArray[np.integer]], order: int, base: Probabilities) -> KneserNeyModel:
        occurrences: list[list[NgramOccurrences]] = [[] for _ in range(order + 1)]
        for sequence in sequences:
            tokens = sequence.astype(np.int64)
            hashes = NgramHasher.contexts(tokens, order)
            for size in range(min(order, len(tokens) - 1) + 1):
                occurrences[size].append(NgramOccurrences.of_level(tokens, hashes, size))
        levels = [NgramLevel.count(NgramOccurrences.joined(level)) for level in occurrences]
        return cls(levels=levels, base=base)

    def truncated(self, order: int) -> KneserNeyModel:
        self._check(order)
        return KneserNeyModel(levels=self.levels[: order + 1], base=self.base)

    def token_log_probs(self, sequence: npt.NDArray[np.integer], order: int) -> Probabilities:
        self._check(order)
        tokens = sequence.astype(np.int64)
        hashes = NgramHasher.contexts(tokens, order)
        positions = np.arange(1, len(tokens))
        targets = tokens[1:]
        probabilities = self.base[targets]
        for size in range(order + 1):
            level = self.levels[size]
            kind = CountKind.at(size, order)
            contexts = hashes[size, 1:]
            totals, types = level.context_statistics(contexts, kind)
            ngram_counts = level.ngram_count(NgramHasher.extend(contexts, targets), kind)
            seen = (positions >= size) & (totals > 0)
            discount = level.counts[kind].discount
            interpolated = self._interpolate(ngram_counts, totals, types, discount, probabilities)
            probabilities = np.where(seen, interpolated, probabilities)
        return np.log(probabilities)

    def distribution(self, history: Sequence[int], order: int) -> Probabilities:
        self._check(order)
        window = np.asarray(history, dtype=np.int64)[max(0, len(history) - order) :]
        hashes = NgramHasher.contexts(np.append(window, 0), order)[:, -1]
        candidates = np.arange(len(self.base), dtype=np.int64)
        probabilities = self.base.copy()
        for size in range(min(order, len(window)) + 1):
            level = self.levels[size]
            kind = CountKind.at(size, order)
            context = hashes[size : size + 1]
            totals, types = level.context_statistics(context, kind)
            if totals[0] == 0:
                continue
            ngrams = NgramHasher.extend(np.repeat(context, len(candidates)), candidates)
            ngram_counts = level.ngram_count(ngrams, kind)
            discount = level.counts[kind].discount
            probabilities = self._interpolate(ngram_counts, totals, types, discount, probabilities)
        return probabilities

    @staticmethod
    def _interpolate(
        counts: Counts, totals: Counts, types: Counts, discount: float, lower: Probabilities
    ) -> Probabilities:
        denominator = np.maximum(totals, 1)
        return (np.maximum(counts - discount, 0.0) + discount * types * lower) / denominator

    def _check(self, order: int) -> None:
        if not 0 <= order <= self.order:
            raise ValueError(f"ordem {order} fora do intervalo treinado [0, {self.order}]")


@dataclass(frozen=True)
class MarkovSample:
    ids: list[int]
    ended_early: bool


class MarkovBaseline:
    def __init__(self, tokenizer: DrumTokenizer, chains: dict[Style, KneserNeyModel]) -> None:
        orders = {chain.order for chain in chains.values()}
        if len(orders) != 1:
            raise ValueError(f"as cadeias precisam ter a mesma ordem, vieram {sorted(orders)}")
        self._tokenizer = tokenizer
        self._chains = chains
        self.order = orders.pop()

    @classmethod
    def fit(cls, corpus: Corpus, tokenizer: DrumTokenizer, order: int) -> MarkovBaseline:
        base = cls._base(tokenizer)
        chains = {
            style: KneserNeyModel.fit(cls.grooves(corpus, style), order, base) for style in Style.evaluated()
        }
        return cls(tokenizer, chains)

    @staticmethod
    def grooves(corpus: Corpus, style: Style) -> list[npt.NDArray[np.int32]]:
        return [
            sequence
            for sequence, sequence_style, piece_type in zip(corpus.sequences, corpus.styles, corpus.types, strict=True)
            if sequence_style is style and piece_type is PieceType.GROOVE
        ]

    @property
    def styles(self) -> list[Style]:
        return list(self._chains)

    def chain(self, style: Style) -> KneserNeyModel:
        return self._chains[style]

    def with_order(self, order: int) -> MarkovBaseline:
        chains = {style: chain.truncated(order) for style, chain in self._chains.items()}
        return MarkovBaseline(self._tokenizer, chains)

    def sample(self, style: Style, seed: int, bars: int) -> MarkovSample:
        rng = np.random.default_rng(seed)
        chain = self.chain(style)
        ids = [self._tokenizer.bos_id]
        bar_lines = 0
        while True:
            distribution = chain.distribution(ids, self.order)
            token = int(rng.choice(len(distribution), p=distribution))
            if token == self._tokenizer.bar_id:
                bar_lines += 1
                if bar_lines == bars:
                    return MarkovSample(ids=ids, ended_early=False)
            ids.append(token)
            if token == self._tokenizer.eos_id:
                return MarkovSample(ids=ids, ended_early=True)

    def save(self, path: Path) -> None:
        arrays: dict[str, npt.NDArray[np.generic]] = {
            "order": np.array(self.order),
            "vocab_size": np.array(self._tokenizer.vocab_size),
            "styles": np.array([style.value for style in self._chains], dtype=str),
        }
        for style, chain in self._chains.items():
            for size, level in enumerate(chain.levels):
                arrays |= level.arrays(f"{style.value}_{size}")
        np.savez_compressed(path, allow_pickle=False, **arrays)

    @classmethod
    def load(cls, path: Path, tokenizer: DrumTokenizer) -> MarkovBaseline:
        with np.load(path, allow_pickle=False) as archive:
            vocab_size = int(archive["vocab_size"])
            if vocab_size != tokenizer.vocab_size:
                raise ValueError(f"{path}: modelo com {vocab_size} tokens, tokenizer com {tokenizer.vocab_size}")
            order = int(archive["order"])
            base = cls._base(tokenizer)
            chains = {
                Style(value): KneserNeyModel(
                    levels=[NgramLevel.from_arrays(archive, f"{value}_{size}") for size in range(order + 1)],
                    base=base,
                )
                for value in archive["styles"].tolist()
            }
        return cls(tokenizer, chains)

    @staticmethod
    def _base(tokenizer: DrumTokenizer) -> Probabilities:
        base = np.ones(tokenizer.vocab_size, dtype=np.float64)
        base[[tokenizer.pad_id, tokenizer.bos_id]] = 0.0
        return base / base.sum()


@dataclass(frozen=True)
class ValidationNll:
    orders: list[int]
    styles: list[Style]
    negative_log_likelihoods: dict[tuple[int, Style], float]
    tokens: dict[Style, int]

    @classmethod
    def measure(cls, baseline: MarkovBaseline, corpus: Corpus, orders: Sequence[int]) -> ValidationNll:
        sequences = {style: MarkovBaseline.grooves(corpus, style) for style in SELECTION_STYLES}
        styles = [style for style in SELECTION_STYLES if sequences[style]]
        if not styles:
            names = ", ".join(style.value for style in SELECTION_STYLES)
            raise ValueError(f"a validação não tem grooves de nenhum destes estilos: {names}")
        totals = {
            (order, style): -sum(
                float(baseline.chain(style).token_log_probs(sequence, order).sum()) for sequence in sequences[style]
            )
            for order in orders
            for style in styles
        }
        tokens = {style: sum(len(sequence) - 1 for sequence in sequences[style]) for style in styles}
        return cls(orders=list(orders), styles=styles, negative_log_likelihoods=totals, tokens=tokens)

    def per_token(self, order: int, style: Style) -> float:
        return self.negative_log_likelihoods[(order, style)] / self.tokens[style]

    def pooled(self, order: int) -> float:
        total = sum(self.negative_log_likelihoods[(order, style)] for style in self.styles)
        return total / sum(self.tokens.values())

    def best_order(self) -> int:
        return min(self.orders, key=lambda order: (self.pooled(order), order))

    def write_csv(self, path: Path) -> None:
        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["order", *(style.value for style in self.styles), "pooled"])
            for order in self.orders:
                scores = [self.per_token(order, style) for style in self.styles]
                writer.writerow([order, *(f"{score:.6f}" for score in [*scores, self.pooled(order)])])


class MarkovTrainer:
    NLL_FILE = "markov_nll.csv"
    MODEL_FILE = "markov.npz"

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    def run(self) -> ExperimentRun:
        paths = self._config.paths
        orders = sorted(set(self._config.markov.candidate_orders))
        if not orders:
            raise ValueError("markov.candidate_orders precisa de ao menos uma ordem")
        tokenizer = DrumTokenizer.from_file(paths.tokenizer_file)
        baseline = MarkovBaseline.fit(Corpus.load(paths, Split.TRAIN), tokenizer, orders[-1])
        validation = ValidationNll.measure(baseline, Corpus.load(paths, Split.VAL), orders)
        order = validation.best_order()

        run = ExperimentRun.create(self._config)
        validation.write_csv(run.path / self.NLL_FILE)
        baseline.with_order(order).save(run.path / self.MODEL_FILE)
        config = dataclasses.replace(self._config, markov=dataclasses.replace(self._config.markov, order=order))
        run = dataclasses.replace(run, record=dataclasses.replace(run.record, config=config))
        run.save()
        return run


if __name__ == "__main__":
    finished = MarkovTrainer(
        ExperimentConfig.from_cli("Treina o baseline Markov por estilo e escolhe a ordem pela NLL de validação")
    ).run()
    print(f"ordem escolhida: {finished.record.config.markov.order} · run: {finished.path}")
