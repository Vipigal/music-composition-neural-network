import json
import os
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest
from matplotlib import image
from scipy.io import wavfile
from symusic.core import NoteTick, ScoreTick, TrackTick

from drumgen.config import (
    DemoSettings,
    ExperimentConfig,
    ExperimentRun,
    PathSettings,
    PieceType,
    SampleFolder,
    SamplingPoint,
    Split,
    Style,
    StyleSwitch,
    System,
)
from drumgen.demo import (
    Demo,
    DemoCommand,
    DemoItem,
    DemoLayout,
    DemoMedia,
    DemoReport,
    DemoSelection,
    DemoSelector,
    DemoSources,
)
from drumgen.generate import MANIFEST_FILE, GenerationManifest
from drumgen.render import RELEASE_TAIL_SECONDS, AudioRenderer, DrumClip, PianoRoll, StyleTempos
from drumgen.sampling import BarPlan, GenerationRequest
from drumgen.tokenizer import TICKS_PER_QUARTER

REPO_ROOT = Path(__file__).resolve().parents[1]
BARS = 8
MEDIAN_BPM = {"rock": 105, "funk": 100, "latin": 108.0, "jazz": 124, "hiphop": 92.0, "soul": 104.5}
FINAL_CFG = 3.0
POINT_TEMPERATURE = 0.9
POINT_TOP_P = 0.9
CFG_SCALES = [1.0, 1.5, 3.0]
LADDER_COMPLETE_SEEDS = {1.0: [0, 1, 2], 1.5: [1, 2], 3.0: [0, 2]}
VALIDATION_WINDOWS = ["b_0", "a_10", "a_1"]
TEST_WINDOWS = ["0_0", "a_0"]
EXPECTED_ITEMS = 6 + 6 + 5 + 6 + 2 + 18
ROLL_HEIGHT = 6
ROLL_WIDTH = 20


@dataclass
class SilentAudio:
    bpm_of: dict[Path, float] = field(default_factory=dict)

    def render(self, clip: DrumClip, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"RIFF")
        self.bpm_of[target] = clip.bpm


@dataclass
class BlankRoll:
    titles: dict[Path, str] = field(default_factory=dict)

    def draw(self, clip: DrumClip, title: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        image.imsave(target, np.zeros((ROLL_HEIGHT, ROLL_WIDTH, 3)))
        self.titles[target] = title


@dataclass(frozen=True)
class DemoWorld:
    config: ExperimentConfig
    final_run: Path
    sweep_run: Path

    @property
    def output(self) -> Path:
        return self.config.demo.output_dir

    @property
    def test_root(self) -> Path:
        return self.config.paths.reference_root / Split.TEST.value

    def select(self) -> DemoSelection:
        return DemoSelector(self.config, DemoSources.of(self.config.demo)).select()


@dataclass(frozen=True)
class DemoOutcome:
    world: DemoWorld
    report: DemoReport
    page: str
    audio: SilentAudio
    roll: BlankRoll
    test_was_locked: bool


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self.headings: list[str] = []
        self._heading = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in ("src", "href") and value is not None:
                self.links.append((tag, value))
        self._heading = tag == "h2"

    def handle_endtag(self, tag: str) -> None:
        if tag == "h2":
            self._heading = False

    def handle_data(self, data: str) -> None:
        if self._heading and data.strip():
            self.headings.append(data.strip())


def groove(velocity: int) -> ScoreTick:
    score = ScoreTick(TICKS_PER_QUARTER)
    track = TrackTick("Drums", 0, True)
    for beat in range(BARS * 4):
        start = beat * TICKS_PER_QUARTER
        track.notes.append(NoteTick(start, TICKS_PER_QUARTER // 2, 42, velocity))
        track.notes.append(NoteTick(start + 7, TICKS_PER_QUARTER // 2, 36 if beat % 2 == 0 else 38, velocity))
    score.tracks.append(track)
    return score


class ManifestWriter:
    def __init__(self, system_dir: Path, system: System) -> None:
        system_dir.mkdir(parents=True, exist_ok=True)
        self._dir = system_dir
        self._file = (system_dir / MANIFEST_FILE).open("w", newline="")
        self._manifest = GenerationManifest(self._file, system)

    def add(
        self,
        folder: SampleFolder,
        seed: int,
        complete: bool,
        cfg_scale: float | None = None,
        temperature: float = 1.0,
        top_p: float | None = None,
    ) -> None:
        plan = (BarPlan(folder.style, PieceType.GROOVE),) * BARS
        request = GenerationRequest(
            plan=plan, seed=seed, cfg_scale=cfg_scale, temperature=temperature, top_k=None, top_p=top_p
        )
        file = ""
        if complete:
            file = f"{folder.name}/{seed}.mid"
            target = self._dir / file
            target.parent.mkdir(parents=True, exist_ok=True)
            groove(velocity=10 + seed).dump_midi(target)
        self._manifest.record(folder, request, [], not complete, 0, file)

    def close(self) -> None:
        self._file.close()


def write_final_run(run: Path, top_p: float | None = POINT_TOP_P) -> None:
    transformer = ManifestWriter(run / "samples" / "transformer", System.TRANSFORMER)
    sampled = {"cfg_scale": FINAL_CFG, "temperature": POINT_TEMPERATURE, "top_p": top_p}
    for style in Style.evaluated():
        transformer.add(SampleFolder.original(style), 0, complete=False, **sampled)
        for seed in (1, 2):
            transformer.add(SampleFolder.original(style), seed, complete=True, **sampled)
        for seed in (2, 1) if style is Style.LATIN else (1, 2):
            transformer.add(SampleFolder.quantized(style), seed, complete=True, **sampled)
    rock_funk = SampleFolder.switch(StyleSwitch(Style.ROCK, Style.FUNK))
    jazz_hiphop = SampleFolder.switch(StyleSwitch(Style.JAZZ, Style.HIPHOP))
    transformer.add(rock_funk, 0, complete=True, **sampled)
    transformer.add(jazz_hiphop, 0, complete=False, **sampled)
    transformer.add(jazz_hiphop, 1, complete=True, **sampled)
    transformer.close()
    markov = ManifestWriter(run / "samples" / "markov", System.MARKOV)
    for style in Style.evaluated():
        markov.add(SampleFolder.original(style), 0, complete=style is not Style.ROCK)
        markov.add(SampleFolder.quantized(style), 0, complete=style is not Style.ROCK)
        markov.add(SampleFolder.original(style), 1, complete=True)
        markov.add(SampleFolder.quantized(style), 1, complete=True)
    markov.close()


def write_sweep_run(run: Path, top_p: float = POINT_TOP_P) -> None:
    transformer = ManifestWriter(run / "samples" / "transformer", System.TRANSFORMER)
    for style in Style.evaluated():
        for scale in CFG_SCALES:
            for temperature in (POINT_TEMPERATURE, 1.0):
                point = SamplingPoint.of(scale, temperature, top_p)
                complete = LADDER_COMPLETE_SEEDS[scale] if temperature == POINT_TEMPERATURE else [0, 1, 2]
                for seed in range(3):
                    transformer.add(
                        SampleFolder.guided(style, point),
                        seed,
                        complete=seed in complete,
                        cfg_scale=scale,
                        temperature=temperature,
                        top_p=point.top_p,
                    )
    transformer.close()


def write_reference(paths: PathSettings) -> None:
    for style in Style.evaluated():
        if style is Style.SOUL:
            continue
        for split, names in ((Split.VAL, VALIDATION_WINDOWS), (Split.TEST, TEST_WINDOWS)):
            folder = paths.reference_dir(split, style)
            folder.mkdir(parents=True, exist_ok=True)
            for index, name in enumerate(names):
                groove(velocity=90 + index).dump_midi(folder / f"{name}.mid")
    soul_test = paths.reference_dir(Split.TEST, Style.SOUL)
    soul_test.mkdir(parents=True)
    groove(velocity=99).dump_midi(soul_test / "soul_0.mid")


def build_world(root: Path) -> DemoWorld:
    paths = PathSettings(processed_dir=root / "processed", runs_dir=root / "runs")
    paths.processed_dir.mkdir(parents=True)
    paths.stats_file.write_text(json.dumps({"vocab_size": 158, "median_bpm": MEDIAN_BPM}))
    write_reference(paths)
    final_run = ExperimentRun.create(ExperimentConfig(name="final", paths=paths)).path
    sweep_run = ExperimentRun.create(ExperimentConfig(name="sweep", paths=paths)).path
    write_final_run(final_run)
    write_sweep_run(sweep_run)
    settings = DemoSettings(final_run=final_run, sweep_run=sweep_run, output_dir=root / "entrega")
    config = ExperimentConfig(name="demo", paths=paths, demo=settings)
    return DemoWorld(config=config, final_run=final_run, sweep_run=sweep_run)


def outputs(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


@pytest.fixture(scope="module")
def outcome(tmp_path_factory: pytest.TempPathFactory) -> DemoOutcome:
    world = build_world(tmp_path_factory.mktemp("demo"))
    stale_sample = world.output / "samples" / "transformer" / "rock" / "99.wav"
    stale_image = world.output / "demo" / "img" / "antigo.png"
    for stale in (stale_sample, stale_image):
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"old")
    audio = SilentAudio()
    roll = BlankRoll()
    world.test_root.chmod(0o000)
    try:
        locked = not os.access(world.test_root, os.R_OK | os.X_OK)
        report = Demo(world.config, audio=audio, piano_roll=roll).run()
    finally:
        world.test_root.chmod(0o755)
    page = report.page_file.read_text()
    return DemoOutcome(world, report, page, audio, roll, locked)


def test_each_folder_contributes_its_first_complete_seed_in_manifest_order(outcome: DemoOutcome) -> None:
    samples = outcome.world.output / "samples"

    assert outputs(samples / "transformer" / "rock") == ["1.mid", "1.wav"]
    assert outputs(samples / "transformer" / "troca_rock-funk") == ["0.mid", "0.wav"]
    assert outputs(samples / "transformer" / "troca_jazz-hiphop") == ["1.mid", "1.wav"]
    assert outputs(samples / "markov" / "rock") == ["1.mid", "1.wav"]
    assert outputs(samples / "markov" / "funk") == ["0.mid", "0.wav"]
    assert len(outcome.report.items) == EXPECTED_ITEMS


def test_selection_follows_the_manifest_order_rather_than_the_seed_order(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    markov = ManifestWriter(world.final_run / "samples" / "markov", System.MARKOV)
    for style in Style.evaluated():
        for seed in (5, 3):
            markov.add(SampleFolder.original(style), seed, complete=True)
    markov.close()

    selection = world.select()

    assert {comparison.markov.name for comparison in selection.comparisons} == {"5"}


def test_quantized_sample_is_the_same_seed_of_the_original(outcome: DemoOutcome) -> None:
    samples = outcome.world.output / "samples" / "transformer"
    final = outcome.world.final_run / "samples" / "transformer"

    for style in Style.evaluated():
        assert outputs(samples / f"{style.value}_quantizado") == ["1.mid", "1.wav"]
        copied = (samples / f"{style.value}_quantizado" / "1.mid").read_bytes()
        assert copied == (final / f"{style.value}_quantizado" / "1.mid").read_bytes()


def test_style_intensity_uses_the_smallest_seed_complete_in_the_three_guidance_levels(outcome: DemoOutcome) -> None:
    samples = outcome.world.output / "samples" / "transformer"
    guided = sorted(path.name for path in samples.iterdir() if "_cfg" in path.name)

    assert guided == sorted(
        f"{style.value}_cfg{scale:g}_t0.9_p0.9" for style in Style.evaluated() for scale in CFG_SCALES
    )
    for name in guided:
        assert outputs(samples / name) == ["2.mid", "2.wav"]


def test_style_intensity_takes_the_point_of_the_final_run_even_without_top_p(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    write_final_run(world.final_run, top_p=None)
    write_sweep_run(world.sweep_run, top_p=1.0)

    selection = world.select()

    assert selection.sampling == SamplingPoint(cfg_scale=FINAL_CFG, temperature=POINT_TEMPERATURE, top_p=None)
    assert [item.folder.name for _, item in selection.guidance[0].steps] == [
        "rock_cfg1_t0.9_p1",
        "rock_cfg1.5_t0.9_p1",
        "rock_cfg3_t0.9_p1",
    ]
    assert {ladder.seed for ladder in selection.guidance} == {2}


def test_style_intensity_without_a_seed_shared_by_the_three_levels_is_rejected(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    sweep = ManifestWriter(world.sweep_run / "samples" / "transformer", System.TRANSFORMER)
    for style in Style.evaluated():
        for seed, scale in enumerate(CFG_SCALES):
            point = SamplingPoint.of(scale, POINT_TEMPERATURE, POINT_TOP_P)
            sweep.add(SampleFolder.guided(style, point), seed, complete=True, cfg_scale=scale)
    sweep.close()

    with pytest.raises(ValueError, match="mesma seed"):
        world.select()


def test_final_run_must_sample_every_style_at_the_same_point(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    transformer = ManifestWriter(world.final_run / "samples" / "transformer", System.TRANSFORMER)
    for style in Style.evaluated():
        temperature = 1.0 if style is Style.JAZZ else POINT_TEMPERATURE
        transformer.add(SampleFolder.original(style), 0, complete=True, cfg_scale=FINAL_CFG, temperature=temperature)
    transformer.close()

    with pytest.raises(ValueError, match="jazz amostrou com temperatura 1"):
        world.select()


def test_human_is_the_first_validation_window_by_name_and_soul_has_none(outcome: DemoOutcome) -> None:
    human = outcome.world.output / "samples" / "humano"

    assert outputs(human) == sorted(style.value for style in Style.evaluated() if style is not Style.SOUL)
    for style in Style.evaluated():
        if style is not Style.SOUL:
            assert outputs(human / style.value) == ["a_1.mid", "a_1.wav"]
    soul = outcome.page.split('id="estilo-soul"')[1].split("</article>")[0]
    assert "sem janela de validação" in soul
    assert outcome.page.count("sem janela de validação") == 1


def test_demo_runs_with_reference_test_unreadable(outcome: DemoOutcome) -> None:
    validation = outcome.world.config.paths.reference_root / Split.VAL.value
    human_sources = [item.source for item in outcome.report.items if item.system is System.HUMAN]

    assert outcome.test_was_locked
    assert len(human_sources) == 5
    assert all(source.is_relative_to(validation) for source in human_sources)


def test_runs_left_empty_in_the_config_are_reported() -> None:
    with pytest.raises(ValueError, match="final_run"):
        DemoSources.of(DemoSettings())


def test_output_inside_a_source_run_is_refused(tmp_path: Path) -> None:
    settings = DemoSettings(final_run=tmp_path / "a", sweep_run=tmp_path / "b", output_dir=tmp_path / "a")

    with pytest.raises(ValueError, match="run de origem"):
        DemoSources.of(settings)


def test_folder_without_a_complete_seed_is_rejected(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    markov = ManifestWriter(world.final_run / "samples" / "markov", System.MARKOV)
    for style in Style.evaluated():
        markov.add(SampleFolder.original(style), 0, complete=False)
    markov.close()

    with pytest.raises(ValueError, match="markov/rock"):
        world.select()


def test_page_has_the_four_sections_in_order(outcome: DemoOutcome) -> None:
    collector = LinkCollector()
    collector.feed(outcome.page)

    assert collector.headings == [
        "1. Transformer × Markov × humano",
        "2. Com e sem microtiming",
        "3. Troca de estilo no meio",
        "4. Intensidade do estilo",
    ]
    assert '<html lang="pt-BR">' in outcome.page


def test_every_link_is_relative_and_exists_on_disk(outcome: DemoOutcome) -> None:
    collector = LinkCollector()
    collector.feed(outcome.page)
    page_dir = outcome.report.page_file.parent

    audio = [link for tag, link in collector.links if tag == "audio"]
    images = [link for tag, link in collector.links if tag == "img"]
    assert len(audio) == EXPECTED_ITEMS + len(Style.evaluated())
    assert len(set(audio)) == EXPECTED_ITEMS
    assert len(set(images)) == EXPECTED_ITEMS
    files = [link for _, link in collector.links if not link.startswith("#")]
    for link in files:
        assert ":" not in link
        assert not link.startswith("/")
        assert (page_dir / link).resolve().is_file()
    assert all(link.startswith("../samples/") and link.endswith(".wav") for link in audio)
    assert all(link.startswith("img/") for link in images)


def test_players_load_metadata_and_images_reserve_their_size(outcome: DemoOutcome) -> None:
    collector = LinkCollector()
    collector.feed(outcome.page)
    audio = [link for tag, link in collector.links if tag == "audio"]
    images = [link for tag, link in collector.links if tag == "img"]

    assert outcome.page.count('preload="metadata"') == len(audio)
    assert outcome.page.count(f'width="{ROLL_WIDTH}" height="{ROLL_HEIGHT}"') == len(images)


def test_each_item_is_rendered_at_the_bpm_of_its_style(outcome: DemoOutcome) -> None:
    samples = outcome.world.output / "samples"
    bpm_of = outcome.audio.bpm_of

    assert bpm_of[samples / "transformer" / "jazz" / "1.wav"] == 124
    assert bpm_of[samples / "transformer" / "troca_jazz-hiphop" / "1.wav"] == 124
    assert bpm_of[samples / "markov" / "hiphop" / "0.wav"] == 92
    assert bpm_of[samples / "humano" / "latin" / "a_1.wav"] == 108
    assert outcome.roll.titles[outcome.world.output / "demo" / "img" / "transformer_soul_1.png"] == (
        "transformer/soul/1 · 104.5 BPM"
    )
    assert "104,5 BPM" in outcome.page


def test_previous_outputs_are_replaced(outcome: DemoOutcome) -> None:
    assert not (outcome.world.output / "samples" / "transformer" / "rock" / "99.wav").exists()
    assert not (outcome.world.output / "demo" / "img" / "antigo.png").exists()


def test_media_renders_the_audio_and_the_piano_roll_of_an_item(tmp_path: Path) -> None:
    source = tmp_path / "fonte" / "3.mid"
    source.parent.mkdir()
    groove(velocity=90).dump_midi(source)
    layout = DemoLayout(tmp_path / "entrega")
    tempos = StyleTempos(bpm={Style.JAZZ: 124.0})
    config = ExperimentConfig(name="demo")
    media = DemoMedia(layout, tempos, BARS, AudioRenderer(config.render), PianoRoll())
    item = DemoItem(system=System.TRANSFORMER, folder=SampleFolder.original(Style.JAZZ), source=source, name="3")

    media.publish(item)

    rate, audio = wavfile.read(layout.audio_file(item))
    assert layout.midi_file(item).read_bytes() == source.read_bytes()
    assert len(audio) / rate == pytest.approx(BARS * 4 * 60 / 124 + RELEASE_TAIL_SECONDS, abs=1 / rate)
    assert layout.image_file(item).stat().st_size > 10_000


def test_repo_config_points_to_the_final_and_sweep_runs() -> None:
    config = ExperimentConfig.from_yaml(REPO_ROOT / "configs" / "demo.yaml")

    assert config.demo.final_run == REPO_ROOT / "runs" / "20261123-2240_final"
    assert config.demo.sweep_run == REPO_ROOT / "runs" / "20261123-2204_sweep"
    assert config.demo.cfg_scales == CFG_SCALES
    assert config.demo.style_switches == [StyleSwitch(Style.ROCK, Style.FUNK), StyleSwitch(Style.JAZZ, Style.HIPHOP)]
    assert config.demo.output_dir == REPO_ROOT


def test_command_takes_the_config(tmp_path: Path) -> None:
    config_file = tmp_path / "demo.yaml"
    config_file.write_text("name: demo\ndemo:\n  final_run: /x/final\n  sweep_run: /x/sweep\n")

    command = DemoCommand.parse(["--config", str(config_file)])

    assert command.config.name == "demo"
    assert command.config.demo.sweep_run == Path("/x/sweep")
