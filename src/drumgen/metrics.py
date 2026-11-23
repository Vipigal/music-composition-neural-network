from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import muspy
import numpy as np
import numpy.typing as npt
from symusic.core import ScoreTick

from drumgen.config import Style, TokenizerSettings

BEATS_PER_BAR = 4
SIXTEENTHS_PER_BEAT = 4
SWING_LOW = 0.4
SWING_HIGH = 0.8
WINDOW_END_MARKER = "fim da janela"
MIDI_VELOCITY_MAX = 127
KDE_STEPS_PER_BANDWIDTH = 8
KDE_MIN_GRID_POINTS = 2048
KDE_MAX_GRID_POINTS = 200_000
KDE_CHUNK = 16
KDE_TAIL_BANDWIDTHS = 4.0
DENSITY_FLOOR = 1e-12


class GrooveMetric(Enum):
    EMPTY_BEAT_RATE = "empty_beat_rate"
    GROOVE_CONSISTENCY = "groove_consistency"
    DRUM_PATTERN_CONSISTENCY = "drum_pattern_consistency"
    DENSITY = "density"
    SWING = "swing"
    MICROTIMING_DEVIATION = "microtiming_deviation"


class SetMetric(Enum):
    TIMING_KL = "timing_kl"
    VELOCITY_KL = "velocity_kl"


@dataclass(frozen=True)
class BeatGrid:
    ticks_per_quarter: int

    @property
    def sixteenth(self) -> float:
        return self.ticks_per_quarter / SIXTEENTHS_PER_BEAT

    def sixteenth_indices(self, onsets: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        return np.ceil((onsets - self.sixteenth / 2) / self.sixteenth).astype(np.int64)

    def quantize(self, onsets: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        return np.rint(self.sixteenth_indices(onsets) * self.sixteenth).astype(np.int64)

    def offsets(self, onsets: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        return onsets / self.sixteenth - self.sixteenth_indices(onsets)

    def positions(self, onsets: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        return self.sixteenth_indices(onsets) % SIXTEENTHS_PER_BEAT

    def beat_fractions(self, onsets: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        return (onsets % self.ticks_per_quarter) / self.ticks_per_quarter

    def swing(self, onsets: npt.NDArray[np.int64]) -> float | None:
        fractions = self.beat_fractions(onsets)
        offbeats = fractions[(fractions >= SWING_LOW) & (fractions < SWING_HIGH)]
        return float(np.median(offbeats)) if offbeats.size else None


@dataclass(frozen=True)
class DrumSample:
    score: ScoreTick
    quantized: ScoreTick
    bars: int

    @classmethod
    def load(cls, path: Path, bars: int) -> DrumSample:
        return cls.from_score(ScoreTick(path), bars)

    @classmethod
    def from_score(cls, score: ScoreTick, bars: int) -> DrumSample:
        drums = score.copy()
        for track in drums.tracks:
            track.is_drum = True
        window_ticks = bars * BEATS_PER_BAR * drums.ticks_per_quarter
        late = [note.time for track in drums.tracks for note in track.notes if note.time >= window_ticks]
        if late:
            raise ValueError(f"nota no tick {late[0]}, depois do fim da janela de {bars} compassos ({window_ticks})")
        quantized = drums.copy()
        grid = BeatGrid(drums.ticks_per_quarter)
        for track in quantized.tracks:
            onsets = grid.quantize(np.array([note.time for note in track.notes], dtype=np.int64))
            for note, onset in zip(track.notes, onsets, strict=True):
                note.time = int(onset)
        return cls(score=drums, quantized=quantized, bars=bars)

    @property
    def grid(self) -> BeatGrid:
        return BeatGrid(self.score.ticks_per_quarter)

    @property
    def window_ticks(self) -> int:
        return self.bars * BEATS_PER_BAR * self.score.ticks_per_quarter

    @property
    def onsets(self) -> npt.NDArray[np.int64]:
        return np.array([note.time for track in self.score.tracks for note in track.notes], dtype=np.int64)

    @property
    def velocities(self) -> npt.NDArray[np.float64]:
        return np.array([note.velocity for track in self.score.tracks for note in track.notes], dtype=np.float64)

    def music(self, quantized: bool = False) -> muspy.Music:
        score = self.quantized if quantized else self.score
        last_tick = self.window_ticks - 1
        notes = [
            muspy.Note(time=note.time, pitch=note.pitch, duration=0, velocity=note.velocity)
            for track in score.tracks
            for note in track.notes
        ]
        track = muspy.Track(
            is_drum=True, notes=notes, annotations=[muspy.Annotation(time=last_tick, annotation=WINDOW_END_MARKER)]
        )
        return muspy.Music(resolution=score.ticks_per_quarter, tracks=[track])


@dataclass(frozen=True)
class SampleMetrics:
    values: dict[GrooveMetric, float | None]

    @classmethod
    def measure(cls, sample: DrumSample) -> SampleMetrics:
        onsets = sample.onsets
        grid = sample.grid
        music = sample.music()
        return cls(
            {
                GrooveMetric.EMPTY_BEAT_RATE: cls._defined(muspy.empty_beat_rate(music)),
                GrooveMetric.GROOVE_CONSISTENCY: cls._defined(
                    muspy.groove_consistency(music, BEATS_PER_BAR * music.resolution)
                ),
                GrooveMetric.DRUM_PATTERN_CONSISTENCY: cls._defined(
                    muspy.drum_pattern_consistency(sample.music(quantized=True))
                ),
                GrooveMetric.DENSITY: onsets.size / sample.bars,
                GrooveMetric.SWING: grid.swing(onsets),
                GrooveMetric.MICROTIMING_DEVIATION: (
                    float(np.mean(np.abs(grid.offsets(onsets)))) if onsets.size else None
                ),
            }
        )

    def value(self, metric: GrooveMetric) -> float | None:
        return self.values[metric]

    @staticmethod
    def _defined(value: float) -> float | None:
        return None if math.isnan(value) else float(value)


@dataclass(frozen=True)
class MeasurementResolution:
    timing: float
    velocity: float

    @classmethod
    def of(cls, settings: TokenizerSettings) -> MeasurementResolution:
        return cls(
            timing=2 * settings.max_microtiming_shift * SIXTEENTHS_PER_BEAT / settings.num_microtiming_bins,
            velocity=MIDI_VELOCITY_MAX / settings.num_velocities,
        )


@dataclass(frozen=True)
class Gaussian:
    mean: float
    std: float

    @classmethod
    def fit(cls, values: npt.NDArray[np.float64], resolution: float) -> Gaussian:
        return cls(mean=float(np.mean(values)), std=max(float(np.std(values)), resolution / math.sqrt(12)))

    def kl_divergence(self, other: Gaussian) -> float:
        return (
            math.log(other.std / self.std)
            + (self.std**2 + (self.mean - other.mean) ** 2) / (2 * other.std**2)
            - 0.5
        )

    @classmethod
    def divergence(
        cls, reference: npt.NDArray[np.float64], generated: npt.NDArray[np.float64], resolution: float
    ) -> float | None:
        if not reference.size or not generated.size:
            return None
        return cls.fit(reference, resolution).kl_divergence(cls.fit(generated, resolution))


@dataclass(frozen=True)
class PositionalDivergence:
    per_position: tuple[float | None, ...]

    @classmethod
    def between(
        cls,
        reference: tuple[npt.NDArray[np.float64], ...],
        generated: tuple[npt.NDArray[np.float64], ...],
        resolution: float,
    ) -> PositionalDivergence:
        return cls(
            tuple(Gaussian.divergence(ref, gen, resolution) for ref, gen in zip(reference, generated, strict=True))
        )

    @property
    def positions(self) -> list[int]:
        return [position for position, value in enumerate(self.per_position) if value is not None]

    @property
    def missing_positions(self) -> list[int]:
        return [position for position, value in enumerate(self.per_position) if value is None]

    @property
    def value(self) -> float | None:
        defined = [value for value in self.per_position if value is not None]
        return float(np.mean(defined)) if defined else None


@dataclass(frozen=True)
class PositionalProfile:
    offsets: tuple[npt.NDArray[np.float64], ...]
    velocities: tuple[npt.NDArray[np.float64], ...]

    @classmethod
    def of(cls, samples: Sequence[DrumSample]) -> PositionalProfile:
        offsets: list[npt.NDArray[np.float64]] = [np.zeros(0)]
        velocities: list[npt.NDArray[np.float64]] = [np.zeros(0)]
        positions: list[npt.NDArray[np.int64]] = [np.zeros(0, dtype=np.int64)]
        for sample in samples:
            onsets = sample.onsets
            offsets.append(sample.grid.offsets(onsets))
            velocities.append(sample.velocities)
            positions.append(sample.grid.positions(onsets))
        all_offsets = np.concatenate(offsets)
        all_velocities = np.concatenate(velocities)
        all_positions = np.concatenate(positions)
        groups = range(SIXTEENTHS_PER_BEAT)
        return cls(
            offsets=tuple(all_offsets[all_positions == group] for group in groups),
            velocities=tuple(all_velocities[all_positions == group] for group in groups),
        )

    def divergence(
        self, reference: PositionalProfile, resolution: MeasurementResolution
    ) -> dict[SetMetric, PositionalDivergence]:
        return {
            SetMetric.TIMING_KL: PositionalDivergence.between(reference.offsets, self.offsets, resolution.timing),
            SetMetric.VELOCITY_KL: PositionalDivergence.between(
                reference.velocities, self.velocities, resolution.velocity
            ),
        }


@dataclass(frozen=True)
class KernelDensity:
    values: npt.NDArray[np.float64]
    bandwidth: float
    degenerate: bool

    @classmethod
    def fit(cls, values: npt.NDArray[np.float64], fallback_bandwidth: float) -> KernelDensity:
        bandwidth = cls.scott_bandwidth(values)
        if bandwidth > 0:
            return cls(values=values, bandwidth=bandwidth, degenerate=False)
        return cls(values=values, bandwidth=fallback_bandwidth, degenerate=True)

    @staticmethod
    def scott_bandwidth(values: npt.NDArray[np.float64]) -> float:
        if values.size < 2:
            return 0.0
        return float(np.std(values, ddof=1)) * values.size ** (-1 / 5)

    def density(self, grid: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        kernels = np.zeros_like(grid)
        for start in range(0, self.values.size, KDE_CHUNK):
            chunk = self.values[start : start + KDE_CHUNK]
            distances = (grid[:, np.newaxis] - chunk[np.newaxis, :]) / self.bandwidth
            kernels += np.exp(-0.5 * distances**2).sum(axis=1)
        return kernels / (self.values.size * self.bandwidth * math.sqrt(2 * math.pi))


@dataclass(frozen=True)
class DistributionComparison:
    generated_mean: float
    generated_std: float
    reference_mean: float
    reference_std: float
    mean_distance: float
    overlap_area: float
    kl_divergence: float
    generated_count: int
    reference_count: int
    generated_undefined: int
    reference_undefined: int
    degenerate: bool

    @classmethod
    def of(
        cls, generated: Sequence[float | None], reference: Sequence[float | None]
    ) -> DistributionComparison | None:
        generated_values = cls._defined(generated)
        reference_values = cls._defined(reference)
        if not generated_values.size or not reference_values.size:
            return None
        overlap_area, kl_divergence, degenerate = cls._densities(generated_values, reference_values)
        return cls(
            generated_mean=float(np.mean(generated_values)),
            generated_std=float(np.std(generated_values)),
            reference_mean=float(np.mean(reference_values)),
            reference_std=float(np.std(reference_values)),
            mean_distance=abs(float(np.mean(generated_values)) - float(np.mean(reference_values))),
            overlap_area=overlap_area,
            kl_divergence=kl_divergence,
            generated_count=int(generated_values.size),
            reference_count=int(reference_values.size),
            generated_undefined=len(generated) - int(generated_values.size),
            reference_undefined=len(reference) - int(reference_values.size),
            degenerate=degenerate,
        )

    @staticmethod
    def _defined(values: Sequence[float | None]) -> npt.NDArray[np.float64]:
        return np.array([value for value in values if value is not None], dtype=np.float64)

    @staticmethod
    def _grid(
        pooled: npt.NDArray[np.float64], generated: KernelDensity, reference: KernelDensity
    ) -> npt.NDArray[np.float64]:
        margin = KDE_TAIL_BANDWIDTHS * max(generated.bandwidth, reference.bandwidth)
        low, high = float(pooled.min()) - margin, float(pooled.max()) + margin
        step = min(generated.bandwidth, reference.bandwidth) / KDE_STEPS_PER_BANDWIDTH
        points = math.ceil((high - low) / step) + 1
        return np.linspace(low, high, min(max(points, KDE_MIN_GRID_POINTS), KDE_MAX_GRID_POINTS))

    @classmethod
    def _densities(
        cls,
        generated: npt.NDArray[np.float64], reference: npt.NDArray[np.float64]
    ) -> tuple[float, float, bool]:
        pooled = np.concatenate([generated, reference])
        pooled_bandwidth = KernelDensity.scott_bandwidth(pooled)
        if pooled_bandwidth == 0:
            return 1.0, 0.0, True
        generated_density = KernelDensity.fit(generated, pooled_bandwidth)
        reference_density = KernelDensity.fit(reference, pooled_bandwidth)
        grid = cls._grid(pooled, generated_density, reference_density)
        p = reference_density.density(grid)
        q = generated_density.density(grid)
        p = p / np.trapezoid(p, grid)
        q = q / np.trapezoid(q, grid)
        overlap_area = float(np.trapezoid(np.minimum(p, q), grid))
        kl_divergence = float(np.trapezoid(p * (np.log(p + DENSITY_FLOOR) - np.log(q + DENSITY_FLOOR)), grid))
        return overlap_area, kl_divergence, generated_density.degenerate or reference_density.degenerate


@dataclass(frozen=True)
class StyleMatrix:
    generated_styles: list[Style]
    reference_styles: list[Style]
    metrics: list[GrooveMetric]
    excluded_metrics: list[GrooveMetric]
    distances: npt.NDArray[np.float64]

    @classmethod
    def build(
        cls,
        generated: Mapping[Style, Sequence[SampleMetrics]],
        reference: Mapping[Style, Sequence[SampleMetrics]],
        metrics: Sequence[GrooveMetric] = tuple(GrooveMetric),
    ) -> StyleMatrix:
        pooled = [sample for samples in reference.values() for sample in samples]
        used = [metric for metric in metrics if cls._spread(pooled, metric) > 0]
        if not used:
            raise ValueError("nenhuma métrica varia na referência; não dá para padronizar")
        centers = np.array([cls._mean(pooled, metric, "referência") for metric in used])
        scales = np.array([cls._spread(pooled, metric) for metric in used])
        generated_means = np.array(
            [[cls._mean(samples, metric, style.value) for metric in used] for style, samples in generated.items()]
        )
        reference_means = np.array(
            [[cls._mean(samples, metric, style.value) for metric in used] for style, samples in reference.items()]
        )
        generated_vectors = (generated_means - centers) / scales
        reference_vectors = (reference_means - centers) / scales
        distances = np.linalg.norm(generated_vectors[:, np.newaxis, :] - reference_vectors[np.newaxis, :, :], axis=2)
        return cls(
            generated_styles=list(generated),
            reference_styles=list(reference),
            metrics=used,
            excluded_metrics=[metric for metric in metrics if metric not in used],
            distances=distances.astype(np.float64),
        )

    @property
    def diagonal_rows(self) -> list[Style]:
        return [style for style in self.generated_styles if style in self.reference_styles]

    @property
    def diagonal_wins(self) -> int:
        wins = 0
        for style in self.diagonal_rows:
            row = self.distances[self.generated_styles.index(style)]
            if row[self.reference_styles.index(style)] <= row.min():
                wins += 1
        return wins

    @staticmethod
    def _values(samples: Sequence[SampleMetrics], metric: GrooveMetric) -> npt.NDArray[np.float64]:
        defined = [value for sample in samples if (value := sample.value(metric)) is not None]
        return np.array(defined, dtype=np.float64)

    @classmethod
    def _mean(cls, samples: Sequence[SampleMetrics], metric: GrooveMetric, owner: str) -> float:
        values = cls._values(samples, metric)
        if not values.size:
            raise ValueError(f"{owner}: nenhuma amostra com {metric.value} definido")
        return float(np.mean(values))

    @classmethod
    def _spread(cls, samples: Sequence[SampleMetrics], metric: GrooveMetric) -> float:
        values = cls._values(samples, metric)
        return float(np.std(values)) if values.size else 0.0
