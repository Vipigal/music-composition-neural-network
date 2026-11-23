from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.io import wavfile
from scipy.signal import resample_poly
from symusic.core import ScoreTick, TempoTick, TextMetaTick, TrackTick
from tqdm import tqdm

from drumgen.config import ExperimentConfig, RenderSettings, SampleFolder, Style
from drumgen.tokenizer import DRUM_CLASSES

BEATS_PER_BAR = 4
SIXTEENTHS_PER_BEAT = 4
RELEASE_TAIL_SECONDS = 3.5
MASTER_GAIN = 0.4
OVERSAMPLING = 4
END_MARKER = "fim"
INT16_MIN = -32768
INT16_MAX = 32767
DRUM_TRACK_NAME = "Drums"


class DrumPiece(Enum):
    KICK = 36
    SNARE = 38
    CLOSED_HIHAT = 42
    OPEN_HIHAT = 46
    LOW_TOM = 43
    MID_TOM = 47
    HIGH_TOM = 50
    CRASH = 49
    RIDE = 51

    @property
    def label(self) -> str:
        return DRUM_PIECE_LABELS[self]

    @property
    def row(self) -> int:
        return list(DrumPiece).index(self)

    @classmethod
    def of(cls, pitch: int) -> DrumPiece | None:
        drum_class = DRUM_CLASSES.get(pitch)
        return None if drum_class is None else cls(drum_class)


DRUM_PIECE_LABELS = {
    DrumPiece.KICK: "bumbo",
    DrumPiece.SNARE: "caixa",
    DrumPiece.CLOSED_HIHAT: "chimbal fechado",
    DrumPiece.OPEN_HIHAT: "chimbal aberto",
    DrumPiece.LOW_TOM: "tom grave",
    DrumPiece.MID_TOM: "tom médio",
    DrumPiece.HIGH_TOM: "tom agudo",
    DrumPiece.CRASH: "prato de ataque",
    DrumPiece.RIDE: "condução",
}


@dataclass(frozen=True)
class StyleTempos:
    bpm: dict[Style, float]

    @classmethod
    def load(cls, stats_file: Path) -> StyleTempos:
        stats = json.loads(stats_file.read_text())
        median_bpm = stats.get("median_bpm") if isinstance(stats, dict) else None
        if not isinstance(median_bpm, dict):
            raise ValueError(f"{stats_file}: sem median_bpm; rode o prepare antes do render")
        by_value = {style.value: style for style in Style.evaluated()}
        return cls(
            bpm={
                by_value[name]: float(bpm)
                for name, bpm in median_bpm.items()
                if name in by_value and isinstance(bpm, int | float)
            }
        )

    def of(self, style: Style) -> float:
        bpm = self.bpm.get(style)
        if bpm is None:
            raise ValueError(f"sem BPM mediano para o estilo {style.value} no stats.json")
        return bpm


@dataclass(frozen=True)
class RenderJob:
    midi_file: Path
    style: Style

    @classmethod
    def of(cls, midi_file: Path) -> RenderJob:
        try:
            folder = SampleFolder.parse(midi_file.parent.name)
        except ValueError as error:
            raise ValueError(f"{midi_file}: {error}") from None
        return cls(midi_file=midi_file, style=folder.style)

    @classmethod
    def collect(cls, inputs: Sequence[Path]) -> list[RenderJob]:
        midi_files: list[Path] = []
        for path in inputs:
            if path.is_dir():
                midi_files += sorted(path.rglob("*.mid"))
            elif path.is_file():
                midi_files.append(path)
            else:
                raise FileNotFoundError(f"{path}: não existe")
        return [cls.of(midi_file) for midi_file in midi_files]

    @property
    def title(self) -> str:
        return "/".join([self.midi_file.parent.parent.name, self.midi_file.parent.name, self.midi_file.stem])

    @property
    def audio_file(self) -> Path:
        return self.midi_file.with_suffix(".wav")

    @property
    def piano_roll_file(self) -> Path:
        return self.midi_file.with_suffix(".png")


@dataclass(frozen=True)
class DrumClip:
    score: ScoreTick
    bpm: float
    bars: int

    @classmethod
    def of(cls, score: ScoreTick, bpm: float, min_bars: int) -> DrumClip:
        clip = score.copy()
        if len(clip.tracks) == 0:
            clip.tracks.append(TrackTick(DRUM_TRACK_NAME, 0, True))
        for track in clip.tracks:
            track.is_drum = True
        clip.tempos.clear()
        clip.tempos.append(TempoTick(0, qpm=bpm))
        bar_ticks = BEATS_PER_BAR * clip.ticks_per_quarter
        onsets = [note.time for track in clip.tracks for note in track.notes]
        bars_with_notes = max(onsets) // bar_ticks + 1 if onsets else 0
        return cls(score=clip, bpm=bpm, bars=max(min_bars, bars_with_notes))

    @property
    def ticks_per_quarter(self) -> int:
        return self.score.ticks_per_quarter

    @property
    def beats(self) -> int:
        return self.bars * BEATS_PER_BAR

    @property
    def duration_seconds(self) -> float:
        return self.beats * 60 / self.bpm

    def ticks_after(self, seconds: float) -> int:
        return self.beats * self.ticks_per_quarter + math.ceil(seconds * self.bpm / 60 * self.ticks_per_quarter)


class AudioRenderer:
    def __init__(self, settings: RenderSettings, tail_seconds: float = RELEASE_TAIL_SECONDS) -> None:
        self._settings = settings
        self._tail_seconds = tail_seconds

    @property
    def synthesis_rate(self) -> int:
        return self._settings.sample_rate * OVERSAMPLING

    def frames(self, clip: DrumClip) -> int:
        return round((clip.duration_seconds + self._tail_seconds) * self._settings.sample_rate)

    def render(self, clip: DrumClip, target: Path) -> None:
        if not self._settings.soundfont.is_file():
            raise FileNotFoundError(f"soundfont não encontrado: {self._settings.soundfont}")
        with tempfile.TemporaryDirectory() as workdir:
            midi_file = Path(workdir) / "clip.mid"
            stereo_file = Path(workdir) / "clip.wav"
            self._with_end_marker(clip).dump_midi(midi_file)
            self._synthesize(midi_file, stereo_file)
            rate, stereo = wavfile.read(stereo_file)
        if rate != self.synthesis_rate:
            raise RuntimeError(f"o FluidSynth gravou {rate} Hz em vez de {self.synthesis_rate} Hz")
        mono = resample_poly(self._downmix(np.asarray(stereo)), 1, OVERSAMPLING)
        target.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(target, self._settings.sample_rate, self._fit(mono, clip))

    def _with_end_marker(self, clip: DrumClip) -> ScoreTick:
        score = clip.score.copy()
        end = clip.ticks_after(self._tail_seconds)
        score.markers.append(TextMetaTick(end, END_MARKER))
        return score

    def _synthesize(self, midi_file: Path, stereo_file: Path) -> None:
        command = [
            "fluidsynth",
            "-ni",
            "-q",
            "-g",
            str(MASTER_GAIN),
            "-r",
            str(self.synthesis_rate),
            "-F",
            str(stereo_file),
            str(self._settings.soundfont),
            str(midi_file),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            raise RuntimeError("fluidsynth não está no PATH") from None
        if result.returncode != 0 or not stereo_file.is_file():
            raise RuntimeError(f"o FluidSynth falhou ({result.returncode}): {result.stderr.strip()}")

    @staticmethod
    def _downmix(audio: npt.NDArray[np.generic]) -> npt.NDArray[np.float64]:
        samples = audio.astype(np.float64)
        return samples.mean(axis=1) if samples.ndim == 2 else samples

    def _fit(self, mono: npt.NDArray[np.floating], clip: DrumClip) -> npt.NDArray[np.int16]:
        frames = self.frames(clip)
        fitted = np.zeros(frames, dtype=np.float64)
        kept = min(frames, len(mono))
        fitted[:kept] = mono[:kept]
        return np.clip(np.round(fitted), INT16_MIN, INT16_MAX).astype(np.int16)


class ChartPalette:
    SURFACE = "#fcfcfb"
    ROW_BAND = "#f4f3f0"
    INK = "#0b0b0b"
    INK_SECONDARY = "#52514e"
    INK_MUTED = "#898781"
    INK_ON_FILL = "#ffffff"
    GRIDLINE = "#e1e0d9"
    BASELINE = "#c3c2b7"
    SERIES_1 = "#2a78d6"
    SERIES_2 = "#eb6834"
    SEQUENTIAL_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    ORDINAL_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


@dataclass(frozen=True)
class RollNote:
    beat: float
    piece: DrumPiece
    velocity: int


class PianoRoll:
    BARS_PER_PANEL = 4
    WIDTH_INCHES = 18.0
    PANEL_HEIGHT_INCHES = 2.6
    DPI = 110
    NOTE_HALF_HEIGHT = 0.38
    NOTE_WIDTH = 2.4
    MAX_VELOCITY = 127

    def __init__(self) -> None:
        self._norm = Normalize(vmin=1, vmax=self.MAX_VELOCITY)
        self._ramp = LinearSegmentedColormap.from_list("velocity", ChartPalette.ORDINAL_RAMP)

    def velocity_color(self, velocity: int) -> tuple[float, float, float]:
        red, green, blue, _ = self._ramp(self._norm(velocity))
        return (red, green, blue)

    def notes(self, clip: DrumClip) -> list[RollNote]:
        notes = [
            RollNote(beat=note.time / clip.ticks_per_quarter, piece=piece, velocity=note.velocity)
            for track in clip.score.tracks
            for note in track.notes
            if (piece := DrumPiece.of(note.pitch)) is not None
        ]
        return sorted(notes, key=lambda note: (note.beat, note.piece.row))

    def panels(self, bars: int) -> int:
        return math.ceil(bars / self.BARS_PER_PANEL)

    def draw(self, clip: DrumClip, title: str, target: Path) -> None:
        panels = self.panels(clip.bars)
        figure, axes = pyplot.subplots(
            panels,
            1,
            figsize=(self.WIDTH_INCHES, self.PANEL_HEIGHT_INCHES * panels + 0.6),
            squeeze=False,
            layout="constrained",
            facecolor=ChartPalette.SURFACE,
        )
        notes = self.notes(clip)
        panel_beats = self.BARS_PER_PANEL * BEATS_PER_BAR
        for panel, axis in enumerate(axes[:, 0]):
            first_beat = panel * panel_beats
            self._draw_grid(axis, first_beat, panel_beats)
            in_panel = [note for note in notes if self.panel_of(note.beat, panels) == panel]
            axis.vlines(
                [note.beat for note in in_panel],
                [note.piece.row - self.NOTE_HALF_HEIGHT for note in in_panel],
                [note.piece.row + self.NOTE_HALF_HEIGHT for note in in_panel],
                colors=[self.velocity_color(note.velocity) for note in in_panel],
                linewidth=self.NOTE_WIDTH,
                capstyle="round",
                zorder=2,
            )
        axes[-1, 0].set_xlabel("compasso.tempo", color=ChartPalette.INK_SECONDARY)
        figure.suptitle(title, color=ChartPalette.INK)
        colorbar = figure.colorbar(ScalarMappable(norm=self._norm, cmap=self._ramp), ax=axes[:, 0], pad=0.01)
        colorbar.set_label("velocity", color=ChartPalette.INK_SECONDARY)
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(colors=ChartPalette.INK_SECONDARY, length=0)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=self.DPI, facecolor=ChartPalette.SURFACE)
        pyplot.close(figure)

    def panel_of(self, beat: float, panels: int) -> int:
        nearest_sixteenth = round(beat * SIXTEENTHS_PER_BEAT) / SIXTEENTHS_PER_BEAT
        return min(int(nearest_sixteenth // (self.BARS_PER_PANEL * BEATS_PER_BAR)), panels - 1)

    def _draw_grid(self, axis: Axes, first_beat: int, panel_beats: int) -> None:
        last_beat = first_beat + panel_beats
        rows = len(DrumPiece)
        axis.set_facecolor(ChartPalette.SURFACE)
        axis.set_xlim(first_beat - 0.25, last_beat + 0.25)
        axis.set_ylim(-0.5, rows - 0.5)
        for row in range(0, rows, 2):
            axis.axhspan(row - 0.5, row + 0.5, color=ChartPalette.ROW_BAND, linewidth=0, zorder=0)
        for step in range(panel_beats * SIXTEENTHS_PER_BEAT + 1):
            beat = first_beat + step / SIXTEENTHS_PER_BEAT
            if step % (BEATS_PER_BAR * SIXTEENTHS_PER_BEAT) == 0:
                axis.axvline(beat, color=ChartPalette.INK_SECONDARY, linewidth=1.2, zorder=1)
            elif step % SIXTEENTHS_PER_BEAT == 0:
                axis.axvline(beat, color=ChartPalette.BASELINE, linewidth=0.8, zorder=1)
            else:
                axis.axvline(beat, color=ChartPalette.GRIDLINE, linewidth=0.6, zorder=1)
        beats = list(range(first_beat, last_beat))
        axis.set_xticks(beats, [f"{beat // BEATS_PER_BAR + 1}.{beat % BEATS_PER_BAR + 1}" for beat in beats])
        axis.set_yticks([piece.row for piece in DrumPiece], [piece.label for piece in DrumPiece])
        axis.tick_params(axis="both", length=0, colors=ChartPalette.INK_SECONDARY)
        for spine in axis.spines.values():
            spine.set_visible(False)


class Renderer:
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._audio = AudioRenderer(config.render)
        self._piano_roll = PianoRoll()

    def run(self, inputs: Sequence[Path]) -> list[RenderJob]:
        tempos = StyleTempos.load(self._config.paths.stats_file)
        jobs = RenderJob.collect(inputs)
        for job in tqdm(jobs, desc="renderizando"):
            bpm = tempos.of(job.style)
            clip = DrumClip.of(ScoreTick(job.midi_file), bpm=bpm, min_bars=self._config.prepare.reference_bars)
            self._audio.render(clip, job.audio_file)
            title = f"{job.title} · {bpm:g} BPM"
            self._piano_roll.draw(clip, title, job.piano_roll_file)
        return jobs


@dataclass(frozen=True)
class RenderCommand:
    config: ExperimentConfig
    inputs: list[Path]

    @classmethod
    def parse(cls, argv: Sequence[str]) -> RenderCommand:
        parser = argparse.ArgumentParser(
            description="Renderiza MIDIs de bateria em WAV mono e piano-roll PNG, gravados ao lado de cada .mid"
        )
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("inputs", type=Path, nargs="+")
        arguments = parser.parse_args(list(argv))
        config_file = arguments.config
        inputs = arguments.inputs
        if not isinstance(config_file, Path) or not isinstance(inputs, list):
            raise TypeError("--config e as entradas precisam ser caminhos")
        return cls(config=ExperimentConfig.from_yaml(config_file), inputs=[Path(path) for path in inputs])


if __name__ == "__main__":
    command = RenderCommand.parse(sys.argv[1:])
    rendered = Renderer(command.config).run(command.inputs)
    print(f"{len(rendered)} MIDIs renderizados")
