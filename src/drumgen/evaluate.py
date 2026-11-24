from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from tqdm import tqdm

from drumgen.config import (
    ExperimentConfig,
    ExperimentRun,
    PathSettings,
    SampleFolder,
    SamplingPoint,
    Split,
    Style,
    System,
)
from drumgen.metrics import (
    DistributionComparison,
    DrumSample,
    GrooveMetric,
    KernelDensity,
    MeasurementResolution,
    PositionalProfile,
    SampleMetrics,
    SetMetric,
    StyleMatrix,
)
from drumgen.render import ChartPalette

EVAL_DIR = "eval"
MIDI_PATTERN = "*.mid"
LIST_SEPARATOR = ";"
MISSING_OVERLAP = 0.0

PER_SAMPLE_COLUMNS = ["system", "variant", "style", "file", *(metric.value for metric in GrooveMetric)]
SUMMARY_COLUMNS = [
    "system",
    "variant",
    "style",
    "reference_split",
    "metric",
    "count",
    "undefined",
    "reference_count",
    "reference_undefined",
    "mean",
    "std",
    "reference_mean",
    "reference_std",
    "mean_distance",
    "overlap_area",
    "kl_divergence",
    "degenerate",
    "positions",
]
STYLE_MATRIX_COLUMNS = [
    "system",
    "variant",
    "reference_split",
    "generated_style",
    "reference_style",
    "distance",
    "row_minimum",
    "diagonal_wins",
    "diagonal_rows",
    "metrics",
    "excluded_metrics",
]
SWEEP_RANKING_COLUMNS = [
    "system",
    "variant",
    "reference_split",
    "cfg_scale",
    "temperature",
    "top_p",
    "mean_overlap_area",
    *(f"overlap_area_{style.value}" for style in Style.evaluated()),
    "excluded_metrics",
]


@dataclass(frozen=True)
class MeasuredSample:
    file: str
    sample: DrumSample
    metrics: SampleMetrics

    @classmethod
    def measure(cls, path: Path, root: Path, bars: int) -> MeasuredSample:
        try:
            sample = DrumSample.load(path, bars)
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error
        return cls(file=path.relative_to(root).as_posix(), sample=sample, metrics=SampleMetrics.measure(sample))

    @classmethod
    def measure_folder(cls, folder: Path, root: Path, bars: int) -> list[MeasuredSample]:
        return [cls.measure(path, root, bars) for path in sorted(folder.glob(MIDI_PATTERN))]


@dataclass(frozen=True)
class SampleGroup:
    system: System
    folder: SampleFolder
    samples: list[MeasuredSample]

    @property
    def variant(self) -> str:
        return self.folder.variant

    @property
    def style(self) -> Style:
        return self.folder.style

    def values(self, metric: GrooveMetric) -> list[float | None]:
        return [sample.metrics.value(metric) for sample in self.samples]


@dataclass(frozen=True)
class ReferenceSet:
    split: Split
    groups: dict[Style, SampleGroup]

    @classmethod
    def load(cls, paths: PathSettings, split: Split, bars: int) -> ReferenceSet:
        groups: dict[Style, SampleGroup] = {}
        for style in Style.evaluated():
            folder = paths.reference_dir(split, style)
            samples = MeasuredSample.measure_folder(folder, paths.reference_root, bars) if folder.is_dir() else []
            if samples:
                groups[style] = SampleGroup(System.REFERENCE, SampleFolder.original(style), samples)
        return cls(split=split, groups=groups)

    @property
    def styles(self) -> list[Style]:
        return list(self.groups)


@dataclass(frozen=True)
class SampleTree:
    groups: list[SampleGroup]
    warnings: list[str]

    @classmethod
    def load(cls, root: Path, bars: int) -> SampleTree:
        if not root.is_dir():
            raise FileNotFoundError(f"{root}: a run não tem a pasta {ExperimentRun.SAMPLES_DIR}/ de amostras")
        found: list[tuple[System, SampleFolder, Path]] = []
        warnings: list[str] = []
        for system_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            system = cls._system(system_dir)
            for folder_dir in sorted(path for path in system_dir.iterdir() if path.is_dir()):
                folder = SampleFolder.parse(folder_dir.name)
                if folder.kind.has_reference:
                    found.append((system, folder, folder_dir))
                else:
                    warnings.append(
                        f"{system.value}/{folder_dir.name}: sem referência correspondente, fora da avaliação"
                    )
        groups: list[SampleGroup] = []
        for system, folder, folder_dir in tqdm(found, desc="medindo amostras"):
            samples = MeasuredSample.measure_folder(folder_dir, root, bars)
            if samples:
                groups.append(SampleGroup(system, folder, samples))
            else:
                warnings.append(f"{system.value}/{folder_dir.name}: pasta sem {MIDI_PATTERN}, fora da avaliação")
        return cls(groups=groups, warnings=warnings)

    @staticmethod
    def _system(system_dir: Path) -> System:
        try:
            system = System(system_dir.name)
        except ValueError:
            raise ValueError(f"{system_dir}: sistema desconhecido {system_dir.name!r}") from None
        if system is System.REFERENCE:
            raise ValueError(f"{system_dir}: {System.REFERENCE.value} é reservado às janelas de referência")
        return system

    def variants(self) -> list[tuple[System, str]]:
        return list(dict.fromkeys((group.system, group.variant) for group in self.groups))

    def by_style(self, system: System, variant: str) -> dict[Style, SampleGroup]:
        groups = {group.style: group for group in self.groups if (group.system, group.variant) == (system, variant)}
        return {style: groups[style] for style in Style.evaluated() if style in groups}


@dataclass(frozen=True)
class GroupComparison:
    group: SampleGroup
    reference: SampleGroup
    distributions: dict[GrooveMetric, DistributionComparison | None]
    positional: dict[SetMetric, tuple[float | None, int]]

    @classmethod
    def of(cls, group: SampleGroup, reference: SampleGroup, resolution: MeasurementResolution) -> GroupComparison:
        profile = PositionalProfile.of([sample.sample for sample in group.samples])
        reference_profile = PositionalProfile.of([sample.sample for sample in reference.samples])
        divergences = profile.divergence(reference_profile, resolution)
        return cls(
            group=group,
            reference=reference,
            distributions={
                metric: DistributionComparison.of(group.values(metric), reference.values(metric))
                for metric in GrooveMetric
            },
            positional={
                metric: (divergence.value, len(divergence.positions)) for metric, divergence in divergences.items()
            },
        )


@dataclass(frozen=True)
class SweepScore:
    system: System
    variant: str
    point: SamplingPoint
    style_overlap: dict[Style, float]
    excluded: list[tuple[Style, GrooveMetric]]

    @property
    def mean_overlap(self) -> float | None:
        selected = [self.style_overlap[style] for style in Style.selection() if style in self.style_overlap]
        return float(np.mean(selected)) if selected else None

    @classmethod
    def rank(cls, comparisons: Sequence[GroupComparison]) -> list[SweepScore]:
        by_point: dict[tuple[System, SamplingPoint], list[GroupComparison]] = {}
        for comparison in comparisons:
            point = comparison.group.folder.point
            if point is not None:
                by_point.setdefault((comparison.group.system, point), []).append(comparison)
        scores = [cls._score(system, point, grouped) for (system, point), grouped in by_point.items()]
        return sorted(
            scores, key=lambda score: -math.inf if score.mean_overlap is None else score.mean_overlap, reverse=True
        )

    @classmethod
    def _score(cls, system: System, point: SamplingPoint, comparisons: Sequence[GroupComparison]) -> SweepScore:
        style_overlap: dict[Style, float] = {}
        excluded: list[tuple[Style, GrooveMetric]] = []
        for comparison in sorted(comparisons, key=lambda comparison: comparison.group.style.index):
            style = comparison.group.style
            kept: list[float] = []
            for metric, distribution in comparison.distributions.items():
                if cls._reference_is_constant(comparison.reference, metric):
                    excluded.append((style, metric))
                else:
                    kept.append(MISSING_OVERLAP if distribution is None else distribution.overlap_area)
            if kept:
                style_overlap[style] = float(np.mean(kept))
        return cls(
            system=system,
            variant=comparisons[0].group.variant,
            point=point,
            style_overlap=style_overlap,
            excluded=excluded,
        )

    @staticmethod
    def _reference_is_constant(reference: SampleGroup, metric: GrooveMetric) -> bool:
        defined = np.array([value for value in reference.values(metric) if value is not None], dtype=np.float64)
        return KernelDensity.scott_bandwidth(defined) == 0


@dataclass(frozen=True)
class StyleMatrixResult:
    system: System
    variant: str
    matrix: StyleMatrix

    @property
    def wins_label(self) -> str:
        return f"diagonal mínima em {self.matrix.diagonal_wins} de {len(self.matrix.diagonal_rows)} linhas"


class CsvFormat:
    @staticmethod
    def number(value: float | None) -> str:
        return "" if value is None else f"{value:.6f}"

    @staticmethod
    def names(metrics: Sequence[GrooveMetric]) -> str:
        return LIST_SEPARATOR.join(metric.value for metric in metrics)


class EvaluationTables:
    PER_SAMPLE_FILE = "per_sample.csv"
    SUMMARY_FILE = "summary.csv"
    STYLE_MATRIX_FILE = "style_matrix.csv"
    SWEEP_RANKING_FILE = "sweep_ranking.csv"

    def __init__(self, output_dir: Path, split: Split) -> None:
        self._output_dir = output_dir
        self._split = split

    def clear(self) -> None:
        for name in (self.PER_SAMPLE_FILE, self.SUMMARY_FILE, self.STYLE_MATRIX_FILE, self.SWEEP_RANKING_FILE):
            (self._output_dir / name).unlink(missing_ok=True)

    def write_per_sample(self, groups: Sequence[SampleGroup]) -> None:
        with (self._output_dir / self.PER_SAMPLE_FILE).open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(PER_SAMPLE_COLUMNS)
            for group in groups:
                for sample in group.samples:
                    writer.writerow(
                        [
                            group.system.value,
                            group.variant,
                            group.style.value,
                            sample.file,
                            *(CsvFormat.number(sample.metrics.value(metric)) for metric in GrooveMetric),
                        ]
                    )

    def write_summary(self, comparisons: Sequence[GroupComparison]) -> None:
        with (self._output_dir / self.SUMMARY_FILE).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            for comparison in comparisons:
                group = comparison.group
                key: dict[str, object] = {
                    "system": group.system.value,
                    "variant": group.variant,
                    "style": group.style.value,
                    "reference_split": self._split.value,
                }
                for metric, distribution in comparison.distributions.items():
                    writer.writerow(key | {"metric": metric.value} | self._distribution(comparison, metric, distribution))
                for metric, (value, positions) in comparison.positional.items():
                    writer.writerow(
                        key
                        | {
                            "metric": metric.value,
                            "count": len(group.samples),
                            "reference_count": len(comparison.reference.samples),
                            "kl_divergence": CsvFormat.number(value),
                            "positions": positions,
                        }
                    )

    @staticmethod
    def _distribution(
        comparison: GroupComparison, metric: GrooveMetric, distribution: DistributionComparison | None
    ) -> dict[str, object]:
        if distribution is None:
            generated = comparison.group.values(metric)
            reference = comparison.reference.values(metric)
            defined = sum(value is not None for value in generated)
            reference_defined = sum(value is not None for value in reference)
            return {
                "count": defined,
                "undefined": len(generated) - defined,
                "reference_count": reference_defined,
                "reference_undefined": len(reference) - reference_defined,
            }
        return {
            "count": distribution.generated_count,
            "undefined": distribution.generated_undefined,
            "reference_count": distribution.reference_count,
            "reference_undefined": distribution.reference_undefined,
            "mean": CsvFormat.number(distribution.generated_mean),
            "std": CsvFormat.number(distribution.generated_std),
            "reference_mean": CsvFormat.number(distribution.reference_mean),
            "reference_std": CsvFormat.number(distribution.reference_std),
            "mean_distance": CsvFormat.number(distribution.mean_distance),
            "overlap_area": CsvFormat.number(distribution.overlap_area),
            "kl_divergence": CsvFormat.number(distribution.kl_divergence),
            "degenerate": distribution.degenerate,
        }

    def write_style_matrices(self, results: Sequence[StyleMatrixResult]) -> None:
        with (self._output_dir / self.STYLE_MATRIX_FILE).open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(STYLE_MATRIX_COLUMNS)
            for result in results:
                matrix = result.matrix
                for row, generated in enumerate(matrix.generated_styles):
                    distances = matrix.distances[row]
                    for column, reference in enumerate(matrix.reference_styles):
                        writer.writerow(
                            [
                                result.system.value,
                                result.variant,
                                self._split.value,
                                generated.value,
                                reference.value,
                                CsvFormat.number(float(distances[column])),
                                bool(distances[column] <= distances.min()),
                                matrix.diagonal_wins,
                                len(matrix.diagonal_rows),
                                CsvFormat.names(matrix.metrics),
                                CsvFormat.names(matrix.excluded_metrics),
                            ]
                        )

    def write_sweep_ranking(self, ranking: Sequence[SweepScore]) -> None:
        with (self._output_dir / self.SWEEP_RANKING_FILE).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=SWEEP_RANKING_COLUMNS)
            writer.writeheader()
            for score in ranking:
                point = score.point
                writer.writerow(
                    {
                        "system": score.system.value,
                        "variant": score.variant,
                        "reference_split": self._split.value,
                        "cfg_scale": point.cfg_scale,
                        "temperature": point.temperature,
                        "top_p": "" if point.top_p is None else point.top_p,
                        "mean_overlap_area": CsvFormat.number(score.mean_overlap),
                        **{
                            f"overlap_area_{style.value}": CsvFormat.number(score.style_overlap.get(style))
                            for style in Style.evaluated()
                        },
                        "excluded_metrics": LIST_SEPARATOR.join(
                            f"{style.value}:{metric.value}" for style, metric in score.excluded
                        ),
                    }
                )


class EvaluationFigures:
    DISTRIBUTION_PREFIX = "distribution_"
    STYLE_MATRIX_PREFIX = "style_matrix_"
    EXTENSION = ".png"
    DPI = 150
    PANEL_WIDTH = 2.9
    PANEL_HEIGHT = 2.0
    MATRIX_CELL = 0.9
    DENSITY_POINTS = 400
    TAIL_BANDWIDTHS = 3.0
    WASH_ALPHA = 0.1
    LINE_WIDTH = 2.0
    DARK_CELL = 0.55
    REFERENCE_COLOR = ChartPalette.SERIES_1
    GENERATED_COLOR = ChartPalette.SERIES_2

    def __init__(self, output_dir: Path, split: Split) -> None:
        self._output_dir = output_dir
        self._split = split

    def clear(self) -> None:
        for prefix in (self.DISTRIBUTION_PREFIX, self.STYLE_MATRIX_PREFIX):
            for path in self._output_dir.glob(f"{prefix}*{self.EXTENSION}"):
                path.unlink()

    def distribution_file(self, metric: GrooveMetric) -> Path:
        return self._output_dir / f"{self.DISTRIBUTION_PREFIX}{metric.value}{self.EXTENSION}"

    def style_matrix_file(self, system: System, variant: str) -> Path:
        return self._output_dir / f"{self.STYLE_MATRIX_PREFIX}{system.value}_{variant}{self.EXTENSION}"

    def draw_distributions(self, comparisons: Sequence[GroupComparison], metric: GrooveMetric) -> None:
        series = list(dict.fromkeys((comparison.group.system, comparison.group.variant) for comparison in comparisons))
        styles = [style for style in Style.evaluated() if any(c.group.style is style for c in comparisons)]
        by_key = {(c.group.system, c.group.variant, c.group.style): c for c in comparisons}
        everything = self._defined(
            [value for c in comparisons for value in c.group.values(metric)]
            + [value for c in comparisons for value in c.reference.values(metric)]
        )
        figure, axes = pyplot.subplots(
            len(series),
            len(styles),
            figsize=(self.PANEL_WIDTH * len(styles), self.PANEL_HEIGHT * len(series) + 0.9),
            squeeze=False,
            sharex=True,
            layout="constrained",
            facecolor=ChartPalette.SURFACE,
        )
        for row, (system, variant) in enumerate(series):
            for column, style in enumerate(styles):
                axis = axes[row, column]
                self._style_axis(axis)
                comparison = by_key.get((system, variant, style))
                if comparison is None:
                    axis.text(0.5, 0.5, "sem amostras", transform=axis.transAxes, ha="center", color=ChartPalette.INK_MUTED)
                else:
                    self._draw_density(axis, comparison, metric, everything)
                if row == 0:
                    axis.set_title(style.value, color=ChartPalette.INK, fontsize=11)
                if column == 0:
                    axis.set_ylabel(f"{system.value}\n{variant}", color=ChartPalette.INK_SECONDARY)
        for axis in axes[-1, :]:
            axis.set_xlabel(metric.value, color=ChartPalette.INK_SECONDARY)
        handles = [
            Line2D([], [], color=self.REFERENCE_COLOR, linewidth=self.LINE_WIDTH),
            Line2D([], [], color=self.GENERATED_COLOR, linewidth=self.LINE_WIDTH),
        ]
        figure.legend(
            handles,
            [f"referência ({self._split.value})", "gerado"],
            loc="outside upper right",
            frameon=False,
            labelcolor=ChartPalette.INK_SECONDARY,
        )
        figure.suptitle(f"{metric.value}: gerado × referência por estilo", color=ChartPalette.INK, x=0.02, ha="left")
        self._save(figure, self.distribution_file(metric))

    def _draw_density(
        self,
        axis: Axes,
        comparison: GroupComparison,
        metric: GrooveMetric,
        everything: npt.NDArray[np.float64],
    ) -> None:
        generated = self._defined(comparison.group.values(metric))
        reference_values = self._defined(comparison.reference.values(metric))
        if not generated.size or not reference_values.size:
            axis.text(0.5, 0.5, "indefinida", transform=axis.transAxes, ha="center", color=ChartPalette.INK_MUTED)
            return
        pooled = np.concatenate([generated, reference_values])
        bandwidth = KernelDensity.scott_bandwidth(pooled)
        distribution = comparison.distributions[metric]
        if bandwidth == 0:
            axis.axvline(float(pooled[0]), color=ChartPalette.INK_MUTED, linewidth=1)
            self._corner_note(axis, f"constante = {pooled[0]:g}")
            return
        margin = self.TAIL_BANDWIDTHS * bandwidth
        grid = np.linspace(float(everything.min()) - margin, float(everything.max()) + margin, self.DENSITY_POINTS)
        for values, color in ((reference_values, self.REFERENCE_COLOR), (generated, self.GENERATED_COLOR)):
            density = KernelDensity.fit(values, bandwidth).density(grid)
            axis.fill_between(grid, density, color=color, alpha=self.WASH_ALPHA, linewidth=0)
            axis.plot(grid, density, color=color, linewidth=self.LINE_WIDTH, solid_capstyle="round")
        if distribution is not None:
            self._corner_note(axis, f"OA {distribution.overlap_area:.2f}")
        axis.set_ylim(bottom=0)

    def _corner_note(self, axis: Axes, text: str) -> None:
        axis.text(
            0.98, 0.95, text, transform=axis.transAxes, ha="right", va="top", color=ChartPalette.INK_SECONDARY, fontsize=9
        )

    def draw_style_matrix(self, result: StyleMatrixResult) -> None:
        matrix = result.matrix
        rows, columns = matrix.distances.shape
        figure, axis = pyplot.subplots(
            figsize=(self.MATRIX_CELL * columns + 3.2, self.MATRIX_CELL * rows + 2.2),
            layout="constrained",
            facecolor=ChartPalette.SURFACE,
        )
        color_map = LinearSegmentedColormap.from_list("sequencial", ChartPalette.SEQUENTIAL_RAMP)
        norm = Normalize(vmin=0.0, vmax=max(float(matrix.distances.max()), 1e-9))
        image = axis.imshow(matrix.distances, cmap=color_map, norm=norm)
        for row in range(rows):
            minimum = matrix.distances[row].min()
            for column in range(columns):
                value = float(matrix.distances[row, column])
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=ChartPalette.INK_ON_FILL if norm(value) > self.DARK_CELL else ChartPalette.INK,
                    fontweight="bold" if value <= minimum else "normal",
                    fontsize=9,
                )
        axis.set_xticks(range(columns), [style.value for style in matrix.reference_styles], color=ChartPalette.INK_SECONDARY)
        axis.set_yticks(range(rows), [style.value for style in matrix.generated_styles], color=ChartPalette.INK_SECONDARY)
        axis.set_xlabel(f"estilo real (referência {self._split.value})", color=ChartPalette.INK_SECONDARY)
        axis.set_ylabel("estilo gerado", color=ChartPalette.INK_SECONDARY)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        excluded = CsvFormat.names(matrix.excluded_metrics) or "nenhuma"
        axis.set_title(
            f"{result.system.value} · {result.variant}: {result.wins_label}\n"
            f"em negrito, o mínimo da linha · excluídas: {excluded}",
            color=ChartPalette.INK,
            fontsize=10,
            loc="left",
        )
        colorbar = figure.colorbar(image, ax=axis, shrink=0.8)
        colorbar.set_label("distância (desvios-padrão da referência)", color=ChartPalette.INK_SECONDARY)
        colorbar.outline.set_visible(False)
        self._save(figure, self.style_matrix_file(result.system, result.variant))

    def _style_axis(self, axis: Axes) -> None:
        axis.set_facecolor(ChartPalette.SURFACE)
        axis.set_yticks([])
        axis.tick_params(colors=ChartPalette.INK_MUTED, length=0)
        for name, spine in axis.spines.items():
            spine.set_visible(name == "bottom")
            spine.set_color(ChartPalette.BASELINE)
        axis.grid(axis="x", color=ChartPalette.GRIDLINE, linewidth=0.8)
        axis.set_axisbelow(True)

    @staticmethod
    def _defined(values: Sequence[float | None]) -> npt.NDArray[np.float64]:
        return np.array([value for value in values if value is not None], dtype=np.float64)

    def _save(self, figure: Figure, target: Path) -> None:
        figure.savefig(target, dpi=self.DPI, facecolor=ChartPalette.SURFACE)
        pyplot.close(figure)


@dataclass(frozen=True)
class EvaluationReport:
    output_dir: Path
    warnings: list[str] = field(default_factory=list)


class Evaluator:
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._split = config.evaluation.reference_split
        self._bars = config.prepare.reference_bars

    def output_dir(self, run: Path) -> Path:
        return self._config.evaluation.output_dir or run / EVAL_DIR

    def run(self, run: Path) -> EvaluationReport:
        experiment = ExperimentRun.open(run)
        experiment.check_compatible(self._config)
        tree = SampleTree.load(experiment.samples_dir, self._bars)
        reference = ReferenceSet.load(self._config.paths, self._split, self._bars)
        if not reference.styles:
            reference_dir = self._config.paths.reference_root / self._split.value
            raise FileNotFoundError(f"{reference_dir}: sem janelas de referência")
        warnings = list(tree.warnings)
        resolution = MeasurementResolution.of(self._config.tokenizer)
        comparisons: list[GroupComparison] = []
        for group in tree.groups:
            if group.style not in reference.groups:
                warnings.append(
                    f"{group.system.value}/{group.folder.name}: sem janelas de {group.style.value} em "
                    f"reference/{self._split.value}, fora do resumo"
                )
                continue
            comparisons.append(GroupComparison.of(group, reference.groups[group.style], resolution))
        matrices = self._style_matrices(tree, reference, warnings)
        ranking = SweepScore.rank(comparisons)

        output = self.output_dir(run)
        output.mkdir(parents=True, exist_ok=True)
        tables = EvaluationTables(output, self._split)
        figures = EvaluationFigures(output, self._split)
        tables.clear()
        figures.clear()
        tables.write_per_sample([*tree.groups, *reference.groups.values()])
        tables.write_summary(comparisons)
        tables.write_style_matrices(matrices)
        if ranking:
            tables.write_sweep_ranking(ranking)
        if comparisons:
            for metric in GrooveMetric:
                figures.draw_distributions(comparisons, metric)
        for result in matrices:
            figures.draw_style_matrix(result)
        return EvaluationReport(output_dir=output, warnings=warnings)

    def _style_matrices(
        self, tree: SampleTree, reference: ReferenceSet, warnings: list[str]
    ) -> list[StyleMatrixResult]:
        reference_metrics = self._complete(reference.groups, System.REFERENCE.value, warnings)
        results: list[StyleMatrixResult] = []
        for system, variant in tree.variants():
            generated = self._complete(tree.by_style(system, variant), f"{system.value}/{variant}", warnings)
            if generated and reference_metrics:
                matrix = StyleMatrix.build(generated, reference_metrics)
                results.append(StyleMatrixResult(system=system, variant=variant, matrix=matrix))
        return results

    @staticmethod
    def _complete(
        groups: Mapping[Style, SampleGroup], owner: str, warnings: list[str]
    ) -> dict[Style, list[SampleMetrics]]:
        complete: dict[Style, list[SampleMetrics]] = {}
        for style, group in groups.items():
            missing = [metric for metric in GrooveMetric if all(value is None for value in group.values(metric))]
            if missing:
                names = CsvFormat.names(missing)
                warnings.append(f"{owner}: {style.value} sem nenhum valor de {names}, fora da matriz de estilos")
                continue
            complete[style] = [sample.metrics for sample in group.samples]
        return complete


@dataclass(frozen=True)
class EvaluateCommand:
    config: ExperimentConfig
    run: Path

    @classmethod
    def parse(cls, argv: Sequence[str]) -> EvaluateCommand:
        parser = argparse.ArgumentParser(
            description="Avalia as amostras de uma run contra as janelas de referência e grava tabelas e figuras"
        )
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--run", type=Path, required=True)
        arguments = parser.parse_args(list(argv))
        config_file = arguments.config
        run = arguments.run
        if not isinstance(config_file, Path) or not isinstance(run, Path):
            raise TypeError("--config e --run precisam ser caminhos")
        return cls(config=ExperimentConfig.from_yaml(config_file), run=run)


if __name__ == "__main__":
    command = EvaluateCommand.parse(sys.argv[1:])
    report = Evaluator(command.config).run(command.run)
    for warning in report.warnings:
        print(f"aviso: {warning}")
    print(f"avaliação gravada em {report.output_dir}")
