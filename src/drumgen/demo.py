from __future__ import annotations

import argparse
import csv
import html
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from matplotlib import image
from symusic.core import ScoreTick
from tqdm import tqdm

from drumgen.config import (
    FULL_TOP_P,
    DemoSettings,
    ExperimentConfig,
    ExperimentRun,
    PathSettings,
    SampleFolder,
    SamplingPoint,
    Split,
    Style,
    System,
)
from drumgen.generate import MANIFEST_FILE
from drumgen.render import AudioRenderer, DrumClip, PianoRoll, StyleTempos
from drumgen.sampling import NEUTRAL_GUIDANCE

HUMAN_SPLIT = Split.VAL
SAMPLES_DIR = "samples"
DEMO_DIR = "demo"
IMAGES_DIR = "img"
PAGE_FILE = "index.html"
MIDI_SUFFIX = ".mid"
AUDIO_SUFFIX = ".wav"
IMAGE_SUFFIX = ".png"
MANIFEST_TRUE = "True"
MANIFEST_FALSE = "False"

STYLE_LABELS = {
    Style.ROCK: "Rock",
    Style.FUNK: "Funk",
    Style.LATIN: "Latin",
    Style.JAZZ: "Jazz",
    Style.HIPHOP: "Hip-hop",
    Style.SOUL: "Soul",
}


@dataclass(frozen=True)
class DemoSources:
    final_run: Path
    sweep_run: Path

    @classmethod
    def of(cls, settings: DemoSettings) -> DemoSources:
        if settings.final_run is None or settings.sweep_run is None:
            raise ValueError("demo.final_run e demo.sweep_run: preencha a run final e a run de varredura na config")
        if settings.output_dir.resolve() in (settings.final_run.resolve(), settings.sweep_run.resolve()):
            raise ValueError(f"demo.output_dir: {settings.output_dir} é uma run de origem; grave a demo fora dela")
        return cls(final_run=settings.final_run, sweep_run=settings.sweep_run)


@dataclass(frozen=True)
class DemoItem:
    system: System
    folder: SampleFolder
    source: Path
    name: str

    @property
    def style(self) -> Style:
        return self.folder.style

    @property
    def title(self) -> str:
        return f"{self.system.value}/{self.folder.name}/{self.name}"


@dataclass(frozen=True)
class ManifestRow:
    folder: str
    seed: int
    file: str
    complete: bool
    cfg_scale: float | None
    temperature: float
    top_p: float | None

    @classmethod
    def parse(cls, row: dict[str, str], where: str) -> ManifestRow:
        ended_early = row["ended_early"]
        if ended_early not in (MANIFEST_TRUE, MANIFEST_FALSE):
            raise ValueError(f"{where}: ended_early {ended_early!r} fora de {MANIFEST_TRUE}/{MANIFEST_FALSE}")
        return cls(
            folder=row["folder"],
            seed=int(row["seed"]),
            file=row["file"],
            complete=ended_early == MANIFEST_FALSE and row["file"] != "",
            cfg_scale=float(row["cfg_scale"]) if row["cfg_scale"] else None,
            temperature=float(row["temperature"]),
            top_p=float(row["top_p"]) if row["top_p"] else None,
        )


class SampleManifest:
    def __init__(self, system: System, system_dir: Path, rows: list[ManifestRow]) -> None:
        self._system = system
        self._dir = system_dir
        self._rows = rows

    @classmethod
    def load(cls, run: Path, system: System, config: ExperimentConfig) -> SampleManifest:
        experiment = ExperimentRun.open(run)
        experiment.check_compatible(config)
        system_dir = experiment.system_samples_dir(system)
        manifest_file = system_dir / MANIFEST_FILE
        if not manifest_file.is_file():
            raise FileNotFoundError(f"{manifest_file}: a run não tem amostras de {system.value}")
        with manifest_file.open(newline="") as file:
            rows = [
                ManifestRow.parse(row, f"{manifest_file}:{line}")
                for line, row in enumerate(csv.DictReader(file), start=2)
            ]
        return cls(system, system_dir, rows)

    def first_complete(self, folder: SampleFolder) -> DemoItem:
        return self._item(folder, self._first_row(folder))

    def complete(self, folder: SampleFolder, seed: int) -> DemoItem:
        for row in self._rows:
            if row.folder == folder.name and row.seed == seed and row.complete:
                return self._item(folder, row)
        raise ValueError(f"{self._system.value}/{folder.name}: a seed {seed} não está completa no manifest de {self._dir}")

    def complete_seeds(self, folder: SampleFolder) -> set[int]:
        return {row.seed for row in self._rows if row.folder == folder.name and row.complete}

    def sampling_of(self, folder: SampleFolder) -> SamplingPoint:
        row = self._first_row(folder)
        cfg_scale = NEUTRAL_GUIDANCE if row.cfg_scale is None else row.cfg_scale
        return SamplingPoint.of(cfg_scale, row.temperature, row.top_p)

    def _first_row(self, folder: SampleFolder) -> ManifestRow:
        for row in self._rows:
            if row.folder == folder.name and row.complete:
                return row
        raise ValueError(f"{self._system.value}/{folder.name}: nenhuma seed completa no manifest de {self._dir}")

    def _item(self, folder: SampleFolder, row: ManifestRow) -> DemoItem:
        return DemoItem(system=self._system, folder=folder, source=self._dir / row.file, name=str(row.seed))


class HumanWindows:
    def __init__(self, paths: PathSettings) -> None:
        self._paths = paths
        split_dir = paths.reference_root / HUMAN_SPLIT.value
        if not split_dir.is_dir():
            raise FileNotFoundError(f"{split_dir}: sem janelas de referência; rode o prepare antes da demo")

    def first(self, style: Style) -> DemoItem | None:
        folder = self._paths.reference_dir(HUMAN_SPLIT, style)
        windows = sorted(folder.glob(f"*{MIDI_SUFFIX}"), key=lambda path: path.name) if folder.is_dir() else []
        if not windows:
            return None
        return DemoItem(system=System.HUMAN, folder=SampleFolder.original(style), source=windows[0], name=windows[0].stem)


@dataclass(frozen=True)
class StyleComparison:
    style: Style
    transformer: DemoItem
    markov: DemoItem
    human: DemoItem | None


@dataclass(frozen=True)
class MicrotimingPair:
    original: DemoItem
    quantized: DemoItem


@dataclass(frozen=True)
class GuidanceLadder:
    style: Style
    seed: int
    steps: list[tuple[float, DemoItem]]


@dataclass(frozen=True)
class DemoSelection:
    sampling: SamplingPoint
    comparisons: list[StyleComparison]
    microtiming: list[MicrotimingPair]
    switches: list[DemoItem]
    guidance: list[GuidanceLadder]

    @property
    def items(self) -> list[DemoItem]:
        items: list[DemoItem] = []
        for comparison in self.comparisons:
            items += [comparison.transformer, comparison.markov]
            if comparison.human is not None:
                items.append(comparison.human)
        for pair in self.microtiming:
            items += [pair.original, pair.quantized]
        items += self.switches
        for ladder in self.guidance:
            items += [item for _, item in ladder.steps]
        return list(dict.fromkeys(items))


class DemoSelector:
    def __init__(self, config: ExperimentConfig, sources: DemoSources) -> None:
        self._config = config
        self._sources = sources

    def select(self) -> DemoSelection:
        config = self._config
        transformer = SampleManifest.load(self._sources.final_run, System.TRANSFORMER, config)
        markov = SampleManifest.load(self._sources.final_run, System.MARKOV, config)
        sweep = SampleManifest.load(self._sources.sweep_run, System.TRANSFORMER, config)
        humans = HumanWindows(config.paths)
        styles = Style.evaluated()
        originals = {style: transformer.first_complete(SampleFolder.original(style)) for style in styles}
        sampling = self._shared_point(transformer, styles)
        comparisons = [
            StyleComparison(
                style=style,
                transformer=originals[style],
                markov=markov.first_complete(SampleFolder.original(style)),
                human=humans.first(style),
            )
            for style in styles
        ]
        microtiming = [
            MicrotimingPair(original, transformer.complete(SampleFolder.quantized(style), int(original.name)))
            for style, original in originals.items()
        ]
        switches = [transformer.first_complete(SampleFolder.switch(switch)) for switch in config.demo.style_switches]
        guidance = [self._ladder(sweep, style, sampling) for style in styles]
        return DemoSelection(sampling, comparisons, microtiming, switches, guidance)

    def _shared_point(self, transformer: SampleManifest, styles: Sequence[Style]) -> SamplingPoint:
        samplings = {style: transformer.sampling_of(SampleFolder.original(style)) for style in styles}
        first = samplings[styles[0]]
        for style, sampling in samplings.items():
            if (sampling.temperature, sampling.top_p) != (first.temperature, first.top_p):
                raise ValueError(
                    f"{self._sources.final_run}: {style.value} amostrou com temperatura {sampling.temperature:g} e "
                    f"top-p {sampling.top_p or FULL_TOP_P:g}, e {styles[0].value} com temperatura "
                    f"{first.temperature:g} e top-p {first.top_p or FULL_TOP_P:g}; a run final usa um ponto só"
                )
        return first

    def _ladder(self, sweep: SampleManifest, style: Style, sampling: SamplingPoint) -> GuidanceLadder:
        folders = [
            (scale, SampleFolder.guided(style, SamplingPoint.of(scale, sampling.temperature, sampling.top_p)))
            for scale in self._config.demo.cfg_scales
        ]
        seeds = set.intersection(*(sweep.complete_seeds(folder) for _, folder in folders))
        if not seeds:
            names = ", ".join(folder.name for _, folder in folders)
            raise ValueError(f"{self._sources.sweep_run}: nenhuma seed completa é a mesma seed em {names}")
        seed = min(seeds)
        return GuidanceLadder(style=style, seed=seed, steps=[(scale, sweep.complete(folder, seed)) for scale, folder in folders])


@dataclass(frozen=True)
class DemoLayout:
    output_dir: Path

    @property
    def samples_dir(self) -> Path:
        return self.output_dir / SAMPLES_DIR

    @property
    def demo_dir(self) -> Path:
        return self.output_dir / DEMO_DIR

    @property
    def images_dir(self) -> Path:
        return self.demo_dir / IMAGES_DIR

    @property
    def page_file(self) -> Path:
        return self.demo_dir / PAGE_FILE

    def sample_dir(self, item: DemoItem) -> Path:
        return self.samples_dir / item.system.value / item.folder.name

    def midi_file(self, item: DemoItem) -> Path:
        return self.sample_dir(item) / f"{item.name}{MIDI_SUFFIX}"

    def audio_file(self, item: DemoItem) -> Path:
        return self.sample_dir(item) / f"{item.name}{AUDIO_SUFFIX}"

    def image_file(self, item: DemoItem) -> Path:
        return self.images_dir / f"{item.system.value}_{item.folder.name}_{item.name}{IMAGE_SUFFIX}"

    def link(self, path: Path) -> str:
        return path.relative_to(self.demo_dir, walk_up=True).as_posix()


class AudioSink(Protocol):
    def render(self, clip: DrumClip, target: Path) -> None: ...


class PianoRollSink(Protocol):
    def draw(self, clip: DrumClip, title: str, target: Path) -> None: ...


class DemoMedia:
    def __init__(
        self, layout: DemoLayout, tempos: StyleTempos, bars: int, audio: AudioSink, piano_roll: PianoRollSink
    ) -> None:
        self._layout = layout
        self._tempos = tempos
        self._bars = bars
        self._audio = audio
        self._piano_roll = piano_roll

    def clear(self, items: Sequence[DemoItem]) -> None:
        folders = {self._layout.sample_dir(item) for item in items}
        stale = [path for folder in folders for suffix in (MIDI_SUFFIX, AUDIO_SUFFIX) for path in folder.glob(f"*{suffix}")]
        stale += self._layout.images_dir.glob(f"*{IMAGE_SUFFIX}")
        for path in stale:
            path.unlink()
        self._layout.page_file.unlink(missing_ok=True)

    def publish(self, item: DemoItem) -> None:
        midi_file = self._layout.midi_file(item)
        midi_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item.source, midi_file)
        bpm = self._tempos.of(item.style)
        clip = DrumClip.of(ScoreTick(midi_file), bpm=bpm, min_bars=self._bars)
        self._audio.render(clip, self._layout.audio_file(item))
        self._piano_roll.draw(clip, f"{item.title} · {bpm:g} BPM", self._layout.image_file(item))


class DemoPage:
    STYLESHEET = """
:root { color-scheme: light dark; --surface: #fcfcfb; --card: #ffffff; --ink: #0b0b0b;
  --ink-secondary: #52514e; --line: #e1e0d9; --accent: #2a78d6; }
@media (prefers-color-scheme: dark) {
  :root { --surface: #161615; --card: #1f1f1d; --ink: #f2f1ed; --ink-secondary: #b5b3ad;
    --line: #34332f; --accent: #6da7ec; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink);
  font: 16px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 1.8rem; margin: 0 0 8px; }
h2 { font-size: 1.35rem; margin: 48px 0 4px; padding-top: 16px; border-top: 1px solid var(--line); }
h3 { font-size: 1.05rem; margin: 24px 0 8px; }
p { max-width: 72ch; color: var(--ink-secondary); }
nav ol { margin: 8px 0 0; padding-left: 24px; }
nav li { margin: 2px 0; }
a { color: var(--accent); }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }
figure { margin: 0; padding: 12px; background: var(--card); border: 1px solid var(--line); border-radius: 8px; }
figcaption { display: flex; justify-content: space-between; gap: 8px; margin-bottom: 8px; }
figcaption span { color: var(--ink-secondary); font-size: 0.9rem; }
audio { width: 100%; }
figure img { display: block; width: 100%; height: auto; margin-top: 8px; border-radius: 4px; }
.missing { display: flex; align-items: center; justify-content: center; min-height: 120px;
  color: var(--ink-secondary); font-style: italic; }
"""

    def __init__(self, layout: DemoLayout, tempos: StyleTempos, bars: int) -> None:
        self._layout = layout
        self._tempos = tempos
        self._bars = bars

    def write(self, selection: DemoSelection) -> Path:
        self._layout.page_file.parent.mkdir(parents=True, exist_ok=True)
        self._layout.page_file.write_text(self.render(selection), encoding="utf-8")
        return self._layout.page_file

    def render(self, selection: DemoSelection) -> str:
        sections = [
            self._comparisons(selection.comparisons),
            self._microtiming(selection.microtiming),
            self._switches(selection.switches),
            self._guidance(selection.guidance, selection.sampling),
        ]
        return "\n".join(
            [
                "<!doctype html>",
                '<html lang="pt-BR">',
                "<head>",
                '<meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">',
                "<title>Bateria condicionada por estilo</title>",
                f"<style>{self.STYLESHEET}</style>",
                "</head>",
                "<body>",
                "<main>",
                self._intro(selection.sampling),
                *sections,
                "</main>",
                "</body>",
                "</html>",
                "",
            ]
        )

    def _intro(self, sampling: SamplingPoint) -> str:
        return f"""<h1>Bateria condicionada por estilo</h1>
<p>Amostras de {self._bars} compassos do Transformer condicionado por estilo (CFG {self._number(sampling.cfg_scale)},
temperatura {self._number(sampling.temperature)}, top-p {self._top_p(sampling.top_p)}) e da cadeia de Markov por estilo,
ao lado de janelas humanas da validação do Groove MIDI Dataset. Em cada pasta, a amostra é a
primeira seed completa na ordem do manifest, sem escolha depois de ouvir. O áudio usa o BPM mediano do estilo no treino e
o soundfont FluidR3_GM. No piano-roll, a cor mostra a velocity e cada nota fica na posição tocada, fora da grade quando há
microtiming.</p>
<nav><ol>
<li><a href="#estilos">Transformer × Markov × humano</a></li>
<li><a href="#microtiming">Com e sem microtiming</a></li>
<li><a href="#troca">Troca de estilo no meio</a></li>
<li><a href="#intensidade">Intensidade do estilo</a></li>
</ol></nav>"""

    def _comparisons(self, comparisons: Sequence[StyleComparison]) -> str:
        blocks = []
        for comparison in comparisons:
            human = (
                self._card(comparison.human, "Humano", f"janela {comparison.human.name}")
                if comparison.human is not None
                else self._missing("Humano", "sem janela de validação")
            )
            cards = [
                self._card(comparison.transformer, "Transformer", f"seed {comparison.transformer.name}"),
                self._card(comparison.markov, "Markov", f"seed {comparison.markov.name}"),
                human,
            ]
            blocks.append(self._style_block(comparison.style, f"estilo-{comparison.style.value}", cards))
        return self._section(
            "estilos",
            "1. Transformer × Markov × humano",
            "Uma amostra de cada sistema por estilo. A humana é a primeira janela, em ordem de nome, da validação; "
            "o soul não tem janela de validação.",
            blocks,
        )

    def _microtiming(self, pairs: Sequence[MicrotimingPair]) -> str:
        blocks = [
            self._style_block(
                pair.original.style,
                f"microtiming-{pair.original.style.value}",
                [
                    self._card(pair.original, "Com microtiming", f"seed {pair.original.name}"),
                    self._card(pair.quantized, "Quantizada", f"seed {pair.quantized.name}"),
                ],
            )
            for pair in pairs
        ]
        return self._section(
            "microtiming",
            "2. Com e sem microtiming",
            "A mesma amostra do Transformer, com os desvios de microtiming e com todos eles zerados (cada nota na "
            "semicolcheia). Peças e velocities são as mesmas.",
            blocks,
        )

    def _switches(self, switches: Sequence[DemoItem]) -> str:
        half = self._bars // 2
        cards = []
        for item in switches:
            after = item.folder.next_style or item.style
            caption = f"{STYLE_LABELS[item.style]} → {STYLE_LABELS[after]}"
            cards.append(self._card(item, caption, f"seed {item.name}"))
        intro = (
            f"O Transformer recebe um estilo nos compassos 1 a {half} e outro nos compassos {half + 1} a {self._bars}. "
            "O áudio inteiro fica no BPM do primeiro estilo."
        )
        return self._section("troca", "3. Troca de estilo no meio", intro, [f'<div class="grid">{"".join(cards)}</div>'])

    def _guidance(self, ladders: Sequence[GuidanceLadder], point: SamplingPoint) -> str:
        scales = "; ".join(self._number(scale) for scale, _ in ladders[0].steps) if ladders else ""
        blocks = [
            self._style_block(
                ladder.style,
                f"intensidade-{ladder.style.value}",
                [self._card(item, f"CFG {self._number(scale)}", f"seed {ladder.seed}") for scale, item in ladder.steps],
            )
            for ladder in ladders
        ]
        intro = (
            f"A mesma seed com escala de CFG {scales}, na temperatura {self._number(point.temperature)} e top-p "
            f"{self._top_p(point.top_p)} da varredura na validação. CFG 1 é a geração condicionada sem reforço; "
            "escalas maiores empurram a amostra para o estilo."
        )
        return self._section("intensidade", "4. Intensidade do estilo", intro, blocks)

    def _section(self, anchor: str, title: str, intro: str, blocks: Sequence[str]) -> str:
        body = "\n".join(blocks)
        return f'<section id="{anchor}">\n<h2>{self._escape(title)}</h2>\n<p>{self._escape(intro)}</p>\n{body}\n</section>'

    def _style_block(self, style: Style, anchor: str, cards: Sequence[str]) -> str:
        heading = f"{STYLE_LABELS[style]} · {self._bpm(style)}"
        return f'<article id="{anchor}">\n<h3>{self._escape(heading)}</h3>\n<div class="grid">{"".join(cards)}</div>\n</article>'

    def _card(self, item: DemoItem, label: str, detail: str) -> str:
        audio = self._escape(self._layout.link(self._layout.audio_file(item)))
        link = self._escape(self._layout.link(self._layout.image_file(item)))
        height, width = image.imread(self._layout.image_file(item)).shape[:2]
        caption = f"{detail} · {self._bpm(item.style)}"
        alt = f"Piano-roll de {item.title}"
        return (
            f"<figure><figcaption><strong>{self._escape(label)}</strong><span>{self._escape(caption)}</span></figcaption>"
            f'<audio controls preload="metadata" src="{audio}"></audio>'
            f'<a href="{link}"><img src="{link}" alt="{self._escape(alt)}" width="{width}" height="{height}" '
            'loading="lazy"></a></figure>'
        )

    def _missing(self, label: str, note: str) -> str:
        return (
            f"<figure><figcaption><strong>{self._escape(label)}</strong></figcaption>"
            f'<div class="missing">{self._escape(note)}</div></figure>'
        )

    def _bpm(self, style: Style) -> str:
        return f"{self._number(self._tempos.of(style))} BPM"

    def _top_p(self, top_p: float | None) -> str:
        return "1 (sem corte)" if top_p is None else self._number(top_p)

    @staticmethod
    def _number(value: float) -> str:
        return f"{value:g}".replace(".", ",")

    @staticmethod
    def _escape(text: str) -> str:
        return html.escape(text, quote=True)


@dataclass(frozen=True)
class DemoReport:
    page_file: Path
    items: list[DemoItem]


class Demo:
    def __init__(
        self, config: ExperimentConfig, audio: AudioSink | None = None, piano_roll: PianoRollSink | None = None
    ) -> None:
        self._config = config
        self._audio = audio or AudioRenderer(config.render)
        self._piano_roll = piano_roll or PianoRoll()

    def run(self) -> DemoReport:
        config = self._config
        sources = DemoSources.of(config.demo)
        selection = DemoSelector(config, sources).select()
        tempos = StyleTempos.load(config.paths.stats_file)
        bars = config.prepare.reference_bars
        layout = DemoLayout(config.demo.output_dir)
        media = DemoMedia(layout, tempos, bars, self._audio, self._piano_roll)
        items = selection.items
        media.clear(items)
        for item in tqdm(items, desc="demo"):
            media.publish(item)
        page_file = DemoPage(layout, tempos, bars).write(selection)
        return DemoReport(page_file=page_file, items=items)


@dataclass(frozen=True)
class DemoCommand:
    config: ExperimentConfig

    @classmethod
    def parse(cls, argv: Sequence[str]) -> DemoCommand:
        parser = argparse.ArgumentParser(
            description="Copia as amostras da demo para samples/, renderiza o áudio e grava demo/index.html"
        )
        parser.add_argument("--config", type=Path, required=True)
        config_file = parser.parse_args(list(argv)).config
        if not isinstance(config_file, Path):
            raise TypeError("--config precisa ser um caminho")
        return cls(config=ExperimentConfig.from_yaml(config_file))


if __name__ == "__main__":
    command = DemoCommand.parse(sys.argv[1:])
    report = Demo(command.config).run()
    print(f"{len(report.items)} amostras publicadas; página em {report.page_file}")
