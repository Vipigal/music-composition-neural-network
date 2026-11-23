import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import gaussian_kde
from symusic.core import NoteTick, ScoreTick, TrackTick

from drumgen.config import Style, TokenizerSettings
from drumgen.metrics import (
    BeatGrid,
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
from drumgen.tokenizer import TICKS_PER_QUARTER

FIXTURES = Path(__file__).resolve().parent / "fixtures"
BARS = 8
BAR = 4 * TICKS_PER_QUARTER
SIXTEENTH = TICKS_PER_QUARTER // 4
RESOLUTION = MeasurementResolution.of(TokenizerSettings())


def score_of(onsets: list[int], is_drum: bool = True, duration: int = 10, velocity: int = 90) -> ScoreTick:
    score = ScoreTick(TICKS_PER_QUARTER)
    track = TrackTick("Drums", 0, is_drum)
    for onset in onsets:
        track.notes.append(NoteTick(onset, duration, 42, velocity))
    score.tracks.append(track)
    return score


def fixture_window(name: str) -> DrumSample:
    score = ScoreTick(FIXTURES / name).clip(0, BARS * BAR, False)
    return DrumSample.from_score(score, BARS)


def test_load_marks_every_track_as_drums(tmp_path: Path) -> None:
    path = tmp_path / "piano.mid"
    score_of([0, 480], is_drum=False).dump_midi(path)

    sample = DrumSample.load(path, BARS)

    assert all(track.is_drum for track in sample.score.tracks)
    assert all(track.is_drum for track in sample.quantized.tracks)


def test_quantized_copy_moves_onsets_to_the_nearest_sixteenth() -> None:
    sample = DrumSample.from_score(score_of([7, 120 - 50, 240 + 59, 360 + 60, 1000]), BARS)

    quantized = [note.time for note in sample.quantized.tracks[0].notes]

    assert quantized == [0, 120, 240, 360, 960]
    assert [note.time for note in sample.score.tracks[0].notes] == [7, 70, 299, 420, 1000]


def test_note_after_the_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="janela"):
        DrumSample.from_score(score_of([0, BARS * BAR]), BARS)


def test_swing_is_the_median_beat_fraction_of_the_offbeats() -> None:
    onsets = [beat * TICKS_PER_QUARTER + fraction for beat in range(8) for fraction in (0, 120, 288, 400)]
    onsets += [8 * TICKS_PER_QUARTER + 264, 9 * TICKS_PER_QUARTER + 312]

    metrics = SampleMetrics.measure(DrumSample.from_score(score_of(onsets), BARS))

    assert metrics.value(GrooveMetric.SWING) == pytest.approx(0.6)


def test_beat_grid_swing_ignores_onsets_outside_the_offbeat_band() -> None:
    onsets = np.array([0, 120, 192, 288, 384, 400, 480 + 312], dtype=np.int64)

    assert BeatGrid(TICKS_PER_QUARTER).swing(onsets) == pytest.approx(0.6)


def test_beat_grid_swing_is_undefined_without_offbeats() -> None:
    onsets = np.array([0, 120, 480, 600], dtype=np.int64)

    assert BeatGrid(TICKS_PER_QUARTER).swing(onsets) is None


def test_microtiming_deviation_is_the_mean_distance_to_the_grid_in_sixteenths() -> None:
    onsets = [12, SIXTEENTH - 24, 2 * SIXTEENTH, 3 * SIXTEENTH + 60]

    metrics = SampleMetrics.measure(DrumSample.from_score(score_of(onsets), BARS))

    assert metrics.value(GrooveMetric.MICROTIMING_DEVIATION) == pytest.approx((0.1 + 0.2 + 0.0 + 0.5) / 4)


def test_density_counts_notes_per_bar_of_the_window() -> None:
    onsets = [bar * BAR + step * SIXTEENTH for bar in range(4) for step in range(10)]

    metrics = SampleMetrics.measure(DrumSample.from_score(score_of(onsets), BARS))

    assert metrics.value(GrooveMetric.DENSITY) == pytest.approx(40 / BARS)


def test_empty_beat_rate_counts_the_empty_bars_at_the_end_of_the_window() -> None:
    onsets = [beat * TICKS_PER_QUARTER for beat in range(4)]

    metrics = SampleMetrics.measure(DrumSample.from_score(score_of(onsets), BARS))

    assert metrics.value(GrooveMetric.EMPTY_BEAT_RATE) == pytest.approx(28 / 32)


def test_empty_beat_rate_ignores_the_decoded_note_length() -> None:
    offbeat = TICKS_PER_QUARTER * 3 // 4

    metrics = SampleMetrics.measure(DrumSample.from_score(score_of([offbeat], duration=240), BARS))

    assert metrics.value(GrooveMetric.EMPTY_BEAT_RATE) == pytest.approx(31 / 32)


def test_groove_consistency_spans_the_whole_window() -> None:
    pattern = [0, 250, 480, 1210]
    onsets = [bar * BAR + position for bar in range(4) for position in pattern]

    metrics = SampleMetrics.measure(DrumSample.from_score(score_of(onsets), BARS))

    assert metrics.value(GrooveMetric.GROOVE_CONSISTENCY) == pytest.approx(1 - len(pattern) / (BAR * (BARS - 1)))


def test_drum_pattern_consistency_reads_the_quantized_copy() -> None:
    onsets = [step * SIXTEENTH + (17 if step % 2 else -13) for step in range(1, 32)]

    metrics = SampleMetrics.measure(DrumSample.from_score(score_of(onsets), BARS))

    assert metrics.value(GrooveMetric.DRUM_PATTERN_CONSISTENCY) == pytest.approx(1.0)


def test_empty_window_has_no_nan_and_marks_the_undefined_metrics() -> None:
    metrics = SampleMetrics.measure(DrumSample.from_score(score_of([]), BARS))

    assert metrics.value(GrooveMetric.EMPTY_BEAT_RATE) == 1.0
    assert metrics.value(GrooveMetric.GROOVE_CONSISTENCY) == 1.0
    assert metrics.value(GrooveMetric.DENSITY) == 0.0
    assert metrics.value(GrooveMetric.SWING) is None
    assert metrics.value(GrooveMetric.MICROTIMING_DEVIATION) is None
    assert metrics.value(GrooveMetric.DRUM_PATTERN_CONSISTENCY) is None


@pytest.mark.parametrize("name", ["jazz_120_beat.mid", "funk_117_beat.mid"])
def test_real_eight_bar_windows_have_every_metric_defined(name: str) -> None:
    metrics = SampleMetrics.measure(fixture_window(name))

    for metric in GrooveMetric:
        value = metrics.value(metric)
        assert value is not None and math.isfinite(value), metric


def test_positional_kl_of_a_set_against_itself_is_zero() -> None:
    samples = [fixture_window("jazz_120_beat.mid"), fixture_window("funk_117_beat.mid")]
    profile = PositionalProfile.of(samples)

    divergence = profile.divergence(profile, RESOLUTION)

    assert divergence[SetMetric.TIMING_KL].value == pytest.approx(0.0, abs=1e-12)
    assert divergence[SetMetric.VELOCITY_KL].value == pytest.approx(0.0, abs=1e-12)


def test_timing_kl_fits_one_gaussian_per_sixteenth_position() -> None:
    reference = PositionalProfile.of([DrumSample.from_score(score_of([480 - 12, 480 + 12]), BARS)])
    generated = PositionalProfile.of([DrumSample.from_score(score_of([480, 480 + 24]), BARS)])

    divergence = generated.divergence(reference, RESOLUTION)

    assert divergence[SetMetric.TIMING_KL].value == pytest.approx(0.5)
    assert divergence[SetMetric.VELOCITY_KL].value == pytest.approx(0.0)


def test_resolution_comes_from_the_tokenizer_bins() -> None:
    resolution = MeasurementResolution.of(TokenizerSettings())

    assert resolution.timing == pytest.approx(2 * 0.125 * 4 / 16)
    assert resolution.velocity == pytest.approx(127 / 16)


def test_constant_timing_falls_back_to_the_microtiming_bin_width() -> None:
    reference = PositionalProfile.of([DrumSample.from_score(score_of([480 - 12, 480 + 12]), BARS)])
    on_grid = PositionalProfile.of([DrumSample.from_score(score_of([480, 960]), BARS)])
    floor = 0.0625 / math.sqrt(12)

    timing_kl = on_grid.divergence(reference, RESOLUTION)[SetMetric.TIMING_KL].value

    assert timing_kl == pytest.approx(math.log(floor / 0.1) + 0.1**2 / (2 * floor**2) - 0.5)


def test_velocity_kl_grows_with_the_velocity_gap() -> None:
    reference = PositionalProfile.of([DrumSample.from_score(score_of([0, 480], velocity=80), BARS)])
    close = PositionalProfile.of([DrumSample.from_score(score_of([0, 480], velocity=85), BARS)])
    far = PositionalProfile.of([DrumSample.from_score(score_of([0, 480], velocity=120), BARS)])

    close_kl = close.divergence(reference, RESOLUTION)[SetMetric.VELOCITY_KL].value
    far_kl = far.divergence(reference, RESOLUTION)[SetMetric.VELOCITY_KL].value

    assert close_kl is not None and far_kl is not None
    assert 0 < close_kl < far_kl


def test_quantized_set_has_a_finite_timing_kl_against_the_human_one() -> None:
    human = fixture_window("jazz_120_beat.mid")
    quantized = PositionalProfile.of([DrumSample.from_score(human.quantized, BARS)])

    timing_kl = quantized.divergence(PositionalProfile.of([human]), RESOLUTION)[SetMetric.TIMING_KL].value

    assert timing_kl is not None and math.isfinite(timing_kl) and timing_kl > 0


def test_positional_kl_averages_only_positions_present_in_both_sets() -> None:
    reference = PositionalProfile.of([DrumSample.from_score(score_of([480 - 12, 480 + 12, 120, 125]), BARS)])
    generated = PositionalProfile.of([DrumSample.from_score(score_of([480, 480 + 24]), BARS)])
    disjoint = PositionalProfile.of([DrumSample.from_score(score_of([240, 250]), BARS)])

    shared = generated.divergence(reference, RESOLUTION)[SetMetric.TIMING_KL]
    apart = disjoint.divergence(reference, RESOLUTION)[SetMetric.TIMING_KL]

    assert shared.value == pytest.approx(0.5)
    assert shared.positions == [0]
    assert shared.missing_positions == [1, 2, 3]
    assert apart.value is None
    assert apart.positions == []
    assert apart.missing_positions == [0, 1, 2, 3]


def spread(center: float, count: int = 60, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(value) for value in rng.normal(center, 1.0, count)]


def test_kernel_density_matches_scipy_with_scott_bandwidth() -> None:
    values = np.array(spread(0.0), dtype=np.float64)
    grid = np.linspace(-4, 4, 50)

    density = KernelDensity.fit(values, fallback_bandwidth=1.0).density(grid)

    assert density == pytest.approx(gaussian_kde(values)(grid))


def test_comparison_of_a_set_against_itself_overlaps_fully() -> None:
    values = spread(0.0)

    comparison = DistributionComparison.of(values, values)

    assert comparison is not None
    assert comparison.overlap_area == pytest.approx(1.0, abs=1e-6)
    assert comparison.kl_divergence == pytest.approx(0.0, abs=1e-9)
    assert comparison.mean_distance == 0.0
    assert comparison.generated_mean == comparison.reference_mean
    assert comparison.generated_std == pytest.approx(float(np.std(values)))


def test_overlap_shrinks_and_kl_grows_as_the_sets_drift_apart() -> None:
    reference = spread(0.0)
    near = DistributionComparison.of(spread(0.5, seed=1), reference)
    far = DistributionComparison.of(spread(3.0, seed=1), reference)

    assert near is not None and far is not None
    assert far.overlap_area < near.overlap_area < 1
    assert 0 < near.kl_divergence < far.kl_divergence
    assert far.mean_distance == pytest.approx(far.generated_mean - far.reference_mean)


def test_narrow_reference_against_wide_generated_matches_a_fine_integration() -> None:
    rng = np.random.default_rng(0)
    generated = rng.uniform(0.0, 1.0, 100)
    reference = rng.normal(0.5, 0.001, 50)
    reference_kde, generated_kde = gaussian_kde(reference), gaussian_kde(generated)
    grid = np.linspace(reference.min() - 0.01, reference.max() + 0.01, 200_001)
    p, q = reference_kde(grid), generated_kde(grid)

    comparison = DistributionComparison.of(generated.tolist(), reference.tolist())

    assert comparison is not None
    assert comparison.overlap_area == pytest.approx(np.trapezoid(np.minimum(p, q), grid), rel=1e-2)
    assert comparison.kl_divergence == pytest.approx(np.trapezoid(p * np.log(p / q), grid), rel=1e-2)


def test_undefined_values_are_left_out_and_counted() -> None:
    reference = spread(0.0)
    generated = [*spread(0.0, count=10), None, None]

    comparison = DistributionComparison.of(generated, reference)

    assert comparison is not None
    assert comparison.generated_count == 10
    assert comparison.generated_undefined == 2
    assert comparison.reference_count == 60
    assert comparison.reference_undefined == 0


def test_constant_set_is_compared_without_nan_and_flagged() -> None:
    comparison = DistributionComparison.of([0.0] * 20, spread(0.0))

    assert comparison is not None
    assert comparison.degenerate
    assert 0 < comparison.overlap_area < 1
    assert math.isfinite(comparison.kl_divergence) and comparison.kl_divergence > 0


def test_identical_constant_sets_overlap_fully() -> None:
    comparison = DistributionComparison.of([0.25] * 5, [0.25] * 8)

    assert comparison is not None
    assert comparison.degenerate
    assert comparison.overlap_area == 1.0
    assert comparison.kl_divergence == 0.0


def test_set_without_defined_values_cannot_be_compared() -> None:
    assert DistributionComparison.of([None, None], spread(0.0)) is None


def style_metrics(swing: float, density: float) -> SampleMetrics:
    values: dict[GrooveMetric, float | None] = {metric: 0.0 for metric in GrooveMetric}
    values[GrooveMetric.SWING] = swing
    values[GrooveMetric.DENSITY] = density
    return SampleMetrics(values)


MATRIX_METRICS = [GrooveMetric.SWING, GrooveMetric.DENSITY]
MATRIX_REFERENCE = {
    Style.ROCK: [style_metrics(0.5, 10.0), style_metrics(0.5, 10.0)],
    Style.JAZZ: [style_metrics(0.6, 20.0), style_metrics(0.6, 20.0)],
}


def test_style_matrix_measures_standardized_distances_between_style_means() -> None:
    generated = {
        Style.ROCK: [style_metrics(0.5, 15.0)],
        Style.JAZZ: [style_metrics(0.6, 20.0), style_metrics(0.6, 20.0)],
    }

    matrix = StyleMatrix.build(generated, MATRIX_REFERENCE, MATRIX_METRICS)

    assert matrix.generated_styles == [Style.ROCK, Style.JAZZ]
    assert matrix.reference_styles == [Style.ROCK, Style.JAZZ]
    assert matrix.distances == pytest.approx(np.array([[1.0, math.sqrt(5)], [math.sqrt(8), 0.0]]))
    assert matrix.diagonal_wins == 2


def test_style_matrix_counts_rows_whose_diagonal_is_not_the_minimum() -> None:
    generated = {
        Style.ROCK: [style_metrics(0.6, 20.0)],
        Style.JAZZ: [style_metrics(0.5, 10.0)],
        Style.SOUL: [style_metrics(0.5, 10.0)],
    }

    matrix = StyleMatrix.build(generated, MATRIX_REFERENCE, MATRIX_METRICS)

    assert matrix.diagonal_rows == [Style.ROCK, Style.JAZZ]
    assert matrix.diagonal_wins == 0


def test_style_matrix_ignores_undefined_values_in_the_means() -> None:
    undefined = SampleMetrics({**style_metrics(0.0, 15.0).values, GrooveMetric.SWING: None})
    generated = {Style.ROCK: [style_metrics(0.5, 15.0), undefined]}

    matrix = StyleMatrix.build(generated, MATRIX_REFERENCE, MATRIX_METRICS)

    assert matrix.distances[0, 0] == pytest.approx(1.0)


def test_style_matrix_leaves_out_a_metric_without_spread_in_the_reference() -> None:
    metrics = [GrooveMetric.EMPTY_BEAT_RATE, *MATRIX_METRICS]

    matrix = StyleMatrix.build(MATRIX_REFERENCE, MATRIX_REFERENCE, metrics)

    assert matrix.metrics == MATRIX_METRICS
    assert matrix.excluded_metrics == [GrooveMetric.EMPTY_BEAT_RATE]
    assert matrix.diagonal_wins == 2


def test_style_matrix_needs_one_metric_with_spread() -> None:
    with pytest.raises(ValueError, match="nenhuma métrica"):
        StyleMatrix.build(MATRIX_REFERENCE, MATRIX_REFERENCE, [GrooveMetric.EMPTY_BEAT_RATE])
