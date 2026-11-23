import json
from pathlib import Path

import numpy as np
import pytest
from matplotlib import image
from matplotlib.colors import to_rgb
from scipy.io import wavfile
from symusic.core import NoteTick, ScoreTick, TempoTick, TrackTick

from drumgen.config import ExperimentConfig, PathSettings, RenderSettings, Style
from drumgen.render import (
    RELEASE_TAIL_SECONDS,
    AudioRenderer,
    ChartPalette,
    DrumClip,
    DrumPiece,
    PianoRoll,
    RenderCommand,
    Renderer,
    RenderJob,
    StyleTempos,
)
from drumgen.tokenizer import DRUM_CLASSES

TICKS_PER_QUARTER = 480
SIXTEENTH = TICKS_PER_QUARTER // 4
BAR = TICKS_PER_QUARTER * 4
SAMPLE_RATE = RenderSettings().sample_rate
INT16_FULL_SCALE = 32768
MEDIAN_BPM = {"rock": 105, "funk": 100, "latin": 108.0, "jazz": 124, "hiphop": 92.0, "soul": 104.5}


def drum_score(notes: list[tuple[int, int, int]], is_drum: bool = True) -> ScoreTick:
    score = ScoreTick(TICKS_PER_QUARTER)
    score.tempos.append(TempoTick(0, qpm=120.0))
    track = TrackTick("Drums", 0, is_drum)
    for time, pitch, velocity in notes:
        track.notes.append(NoteTick(time, TICKS_PER_QUARTER // 2, pitch, velocity))
    score.tracks.append(track)
    return score


def eight_bar_groove() -> ScoreTick:
    notes: list[tuple[int, int, int]] = []
    for bar in range(8):
        start = bar * BAR
        notes += [(start, 36, 100), (start + 2 * TICKS_PER_QUARTER, 36, 90)]
        notes += [(start + TICKS_PER_QUARTER, 38, 110), (start + 3 * TICKS_PER_QUARTER, 38, 105)]
        notes += [(start + eighth * TICKS_PER_QUARTER // 2, 42, 70) for eighth in range(8)]
    return drum_score(notes)


def write_stats(processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "stats.json").write_text(json.dumps({"vocab_size": 158, "median_bpm": MEDIAN_BPM}))


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    rate, data = wavfile.read(path)
    return rate, np.asarray(data)


def relative_luminance(color: str | tuple[float, float, float]) -> float:
    channels = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4 for channel in to_rgb(color)
    ]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def onset_seconds(audio: np.ndarray, rate: int, after: float, threshold: float) -> float:
    start = max(0, round(after * rate))
    return (start + int(np.argmax(np.abs(audio[start:].astype(np.float64)) > threshold))) / rate


def test_drum_pieces_are_the_nine_tokenizer_classes() -> None:
    assert {piece.value for piece in DrumPiece} == set(DRUM_CLASSES.values())
    assert len({piece.label for piece in DrumPiece}) == 9


@pytest.mark.parametrize(
    ("relative", "style"),
    [
        ("data/processed/reference/val/jazz/drummer1-session1-3_0.mid", Style.JAZZ),
        ("runs/x/samples/markov/funk/7.mid", Style.FUNK),
        ("runs/x/samples/transformer/latin_quantizado/7.mid", Style.LATIN),
        ("runs/x/samples/transformer/soul_cfg1.5/7.mid", Style.SOUL),
        ("runs/x/samples/transformer/troca_rock-funk/7.mid", Style.ROCK),
        ("runs/x/samples/transformer/virada_hiphop/7.mid", Style.HIPHOP),
    ],
)
def test_style_comes_from_the_sample_folder(relative: str, style: Style) -> None:
    assert RenderJob.of(Path(relative)).style is style


@pytest.mark.parametrize("folder", ["other", "samples", "outro_jazz", "jazzy"])
def test_folder_outside_the_sample_contract_is_rejected(folder: str) -> None:
    with pytest.raises(ValueError, match="estilo"):
        RenderJob.of(Path("runs/x/samples/markov") / folder / "7.mid")


def test_outputs_sit_next_to_the_midi() -> None:
    job = RenderJob.of(Path("samples/markov/jazz/3.mid"))

    assert (job.audio_file, job.piano_roll_file) == (Path("samples/markov/jazz/3.wav"), Path("samples/markov/jazz/3.png"))


def test_title_names_the_system_or_split_the_folder_and_the_file() -> None:
    sample = RenderJob.of(Path("runs/x/samples/markov/jazz_quantizado/3.mid"))
    reference = RenderJob.of(Path("data/processed/reference/val/funk/drummer1-session1-126_2.mid"))

    assert (sample.title, reference.title) == ("markov/jazz_quantizado/3", "val/funk/drummer1-session1-126_2")


def test_collect_walks_directories_and_accepts_single_files(tmp_path: Path) -> None:
    for relative in ["markov/jazz/0.mid", "markov/jazz/1.mid", "markov/funk_quantizado/0.mid", "humano/rock/a.mid"]:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_bytes(b"")
    (tmp_path / "markov/jazz/0.wav").write_bytes(b"")

    jobs = RenderJob.collect([tmp_path / "markov", tmp_path / "humano/rock/a.mid"])

    assert [job.midi_file.relative_to(tmp_path).as_posix() for job in jobs] == [
        "markov/funk_quantizado/0.mid",
        "markov/jazz/0.mid",
        "markov/jazz/1.mid",
        "humano/rock/a.mid",
    ]


def test_missing_input_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RenderJob.collect([tmp_path / "nada"])


def test_tempos_are_the_median_bpm_of_the_prepare_stats(tmp_path: Path) -> None:
    write_stats(tmp_path)

    tempos = StyleTempos.load(tmp_path / "stats.json")

    assert (tempos.of(Style.JAZZ), tempos.of(Style.SOUL)) == (124.0, 104.5)


def test_style_without_median_bpm_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "stats.json").write_text(json.dumps({"median_bpm": {"rock": 105}}))

    with pytest.raises(ValueError, match="jazz"):
        StyleTempos.load(tmp_path / "stats.json").of(Style.JAZZ)


def test_clip_replaces_the_tempo_and_marks_every_track_as_drums() -> None:
    score = drum_score([(0, 36, 100)], is_drum=False)
    score.tempos.append(TempoTick(BAR, qpm=90.0))

    clip = DrumClip.of(score, bpm=124.0, min_bars=8)

    assert [(tempo.time, tempo.qpm) for tempo in clip.score.tempos] == [(0, pytest.approx(124.0, rel=1e-6))]
    assert all(track.is_drum for track in clip.score.tracks)
    assert not score.tracks[0].is_drum


def test_clip_spans_at_least_the_reference_bars() -> None:
    short = DrumClip.of(drum_score([(0, 36, 100)]), bpm=100.0, min_bars=8)
    long = DrumClip.of(drum_score([(9 * BAR + SIXTEENTH, 36, 100)]), bpm=100.0, min_bars=8)

    assert (short.bars, long.bars) == (8, 10)
    assert short.duration_seconds == pytest.approx(8 * 4 * 60 / 100)


@pytest.mark.parametrize("bpm", [92.0, 124.0])
def test_eight_bars_render_to_mono_audio_with_the_bpm_duration(tmp_path: Path, bpm: float) -> None:
    target = tmp_path / "groove.wav"

    AudioRenderer(RenderSettings()).render(DrumClip.of(eight_bar_groove(), bpm=bpm, min_bars=8), target)

    rate, audio = read_wav(target)
    assert rate == SAMPLE_RATE
    assert audio.ndim == 1
    assert audio.dtype == np.int16
    assert len(audio) / rate == pytest.approx(8 * 4 * 60 / bpm + RELEASE_TAIL_SECONDS, abs=1 / rate)
    assert np.abs(audio).max() > 1000


def test_microtiming_of_one_tick_survives_in_the_audio(tmp_path: Path) -> None:
    target = tmp_path / "ticks.wav"
    bpm = 124.0
    seconds_per_tick = 60 / bpm / TICKS_PER_QUARTER
    notes = [(2 * step * TICKS_PER_QUARTER + step, 37, 120) for step in range(16)]
    clip = DrumClip.of(drum_score(notes), bpm=bpm, min_bars=8)

    AudioRenderer(RenderSettings()).render(clip, target)

    rate, audio = read_wav(target)
    errors = [
        onset_seconds(audio, rate, time * seconds_per_tick - 0.02, threshold=300) - time * seconds_per_tick
        for time, _, _ in notes
    ]
    assert max(errors) - min(errors) <= 0.0015


def test_score_without_tracks_renders_silence_with_the_window_duration(tmp_path: Path) -> None:
    target = tmp_path / "empty.wav"
    clip = DrumClip.of(ScoreTick(TICKS_PER_QUARTER), bpm=100.0, min_bars=8)

    AudioRenderer(RenderSettings()).render(clip, target)

    rate, audio = read_wav(target)
    assert [track.is_drum for track in clip.score.tracks] == [True]
    assert len(audio) / rate == pytest.approx(8 * 4 * 60 / 100 + RELEASE_TAIL_SECONDS, abs=1 / rate)
    assert np.abs(audio).max() < 100


def test_onsets_follow_the_bpm(tmp_path: Path) -> None:
    target = tmp_path / "kick.wav"
    clip = DrumClip.of(drum_score([(BAR, 36, 120)]), bpm=100.0, min_bars=8)

    AudioRenderer(RenderSettings()).render(clip, target)

    rate, audio = read_wav(target)
    first_sound = np.argmax(np.abs(audio) > 200) / rate
    assert first_sound == pytest.approx(4 * 60 / 100, abs=0.02)


def test_latest_loudest_crash_rings_into_the_tail_and_ends_below_minus_60_dbfs(tmp_path: Path) -> None:
    target = tmp_path / "crash.wav"
    latest_onset = 8 * BAR - TICKS_PER_QUARTER // 8
    clip = DrumClip.of(drum_score([(latest_onset, 49, 127)]), bpm=124.0, min_bars=8)

    AudioRenderer(RenderSettings()).render(clip, target)

    rate, audio = read_wav(target)
    level = np.abs(audio.astype(np.float64))
    window_end = round(clip.duration_seconds * rate)
    ringing = level[window_end + rate // 2 : window_end + rate].max()
    last_block = level[-rate // 20 :].max()
    assert ringing > level.max() / 20
    assert last_block < INT16_FULL_SCALE / 1000


def test_missing_soundfont_is_reported(tmp_path: Path) -> None:
    renderer = AudioRenderer(RenderSettings(soundfont=tmp_path / "nada.sf2"))

    with pytest.raises(FileNotFoundError, match="soundfont"):
        renderer.render(DrumClip.of(eight_bar_groove(), bpm=100.0, min_bars=8), tmp_path / "x.wav")


def test_piano_roll_keeps_microtiming_off_the_grid() -> None:
    swung = TICKS_PER_QUARTER * 7 // 12
    clip = DrumClip.of(drum_score([(0, 51, 90), (swung, 51, 40), (BAR, 36, 127)]), bpm=124.0, min_bars=8)

    notes = PianoRoll().notes(clip)

    assert [(note.beat, note.piece, note.velocity) for note in notes] == [
        (0.0, DrumPiece.RIDE, 90),
        (pytest.approx(0.5833, abs=1e-4), DrumPiece.RIDE, 40),
        (4.0, DrumPiece.KICK, 127),
    ]


def test_velocity_darkens_a_single_hue_and_the_softest_note_stays_visible() -> None:
    roll = PianoRoll()
    colors = [roll.velocity_color(velocity) for velocity in (1, 40, 80, 127)]

    luminances = [relative_luminance(color) for color in colors]
    surface = relative_luminance(ChartPalette.SURFACE)
    assert luminances == sorted(luminances, reverse=True)
    assert (surface + 0.05) / (luminances[0] + 0.05) >= 2.0
    assert all(color[2] > color[0] for color in colors)


@pytest.mark.parametrize(("beat", "panel"), [(0.0, 0), (15.5, 0), (15.95, 1), (16.0, 1), (40.0, 1)])
def test_early_downbeat_is_drawn_with_the_bar_it_anticipates(beat: float, panel: int) -> None:
    assert PianoRoll().panel_of(beat, panels=2) == panel


def test_piano_roll_draws_one_row_of_bars_per_panel(tmp_path: Path) -> None:
    target = tmp_path / "groove.png"
    roll = PianoRoll()

    roll.draw(DrumClip.of(eight_bar_groove(), bpm=100.0, min_bars=8), "groove · jazz", target)

    height, width = image.imread(target).shape[:2]
    assert roll.panels(8) == 2
    assert width > 1500
    assert height > 400


def test_renderer_writes_audio_and_piano_roll_next_to_each_midi(tmp_path: Path) -> None:
    paths = PathSettings(processed_dir=tmp_path / "processed", runs_dir=tmp_path / "runs")
    write_stats(paths.processed_dir)
    midi = tmp_path / "samples" / "markov" / "jazz" / "0.mid"
    midi.parent.mkdir(parents=True)
    eight_bar_groove().dump_midi(midi)

    rendered = Renderer(ExperimentConfig(name="test", paths=paths)).run([tmp_path / "samples"])

    rate, audio = read_wav(midi.with_suffix(".wav"))
    assert [job.midi_file for job in rendered] == [midi]
    assert len(audio) / rate == pytest.approx(8 * 4 * 60 / 124 + RELEASE_TAIL_SECONDS, abs=1 / rate)
    assert midi.with_suffix(".png").is_file()


def test_command_takes_the_config_and_the_midi_paths(tmp_path: Path) -> None:
    config_file = tmp_path / "render.yaml"
    config_file.write_text("name: render\n")

    command = RenderCommand.parse(["--config", str(config_file), "runs/a/samples", "data/processed/reference/val"])

    assert command.config.name == "render"
    assert command.inputs == [Path("runs/a/samples"), Path("data/processed/reference/val")]
