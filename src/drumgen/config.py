from __future__ import annotations

import argparse
import dataclasses
import itertools
import re
import subprocess
import types
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Self, get_args, get_origin, get_type_hints

import yaml

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Style(Enum):
    ROCK = "rock"
    FUNK = "funk"
    LATIN = "latin"
    JAZZ = "jazz"
    HIPHOP = "hiphop"
    SOUL = "soul"
    OTHER = "other"
    NULL = "null"

    @property
    def index(self) -> int:
        return list(Style).index(self)

    @classmethod
    def evaluated(cls) -> list[Style]:
        return [style for style in cls if style not in (cls.OTHER, cls.NULL)]

    @classmethod
    def selection(cls) -> list[Style]:
        return [cls.ROCK, cls.FUNK, cls.LATIN, cls.JAZZ]

    @staticmethod
    def genre_of(gmd_style: str) -> str:
        return gmd_style.split("/")[0]

    @classmethod
    def from_gmd(cls, gmd_style: str) -> Style:
        by_genre = {style.value: style for style in cls.evaluated()}
        return by_genre.get(cls.genre_of(gmd_style), cls.OTHER)


class PieceType(Enum):
    GROOVE = "groove"
    FILL = "virada"

    @property
    def index(self) -> int:
        return list(PieceType).index(self)

    @classmethod
    def from_gmd(cls, beat_type: str) -> PieceType:
        match beat_type:
            case "beat":
                return cls.GROOVE
            case "fill":
                return cls.FILL
            case _:
                raise ValueError(f"beat_type desconhecido no info.csv: {beat_type!r}")


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @classmethod
    def from_gmd(cls, gmd_split: str) -> Split:
        match gmd_split:
            case "train":
                return cls.TRAIN
            case "validation":
                return cls.VAL
            case "test":
                return cls.TEST
            case _:
                raise ValueError(f"split desconhecido no info.csv: {gmd_split!r}")


class System(Enum):
    TRANSFORMER = "transformer"
    MARKOV = "markov"
    HUMAN = "humano"
    REFERENCE = "referencia"


class VariantKind(Enum):
    ORIGINAL = "original"
    QUANTIZED = "quantizado"
    GUIDED = "cfg"
    STYLE_SWITCH = "troca"
    FILL = "virada"

    @property
    def has_reference(self) -> bool:
        return self in (VariantKind.ORIGINAL, VariantKind.QUANTIZED, VariantKind.GUIDED)


STYLE_NAME = "|".join(style.value for style in Style.evaluated())
NUMBER = r"\d+(?:\.\d+)?"
FULL_TOP_P = 1.0
FOLDER_PATTERNS = {
    VariantKind.STYLE_SWITCH: re.compile(rf"troca_(?P<style>{STYLE_NAME})-(?P<after>{STYLE_NAME})"),
    VariantKind.FILL: re.compile(rf"virada_(?P<style>{STYLE_NAME})"),
    VariantKind.GUIDED: re.compile(
        rf"(?P<style>{STYLE_NAME})_cfg(?P<scale>{NUMBER})_t(?P<temperature>{NUMBER})_p(?P<top_p>{NUMBER})"
    ),
    VariantKind.QUANTIZED: re.compile(rf"(?P<style>{STYLE_NAME})_quantizado"),
    VariantKind.ORIGINAL: re.compile(rf"(?P<style>{STYLE_NAME})"),
}


@dataclass(frozen=True)
class SamplingPoint:
    cfg_scale: float
    temperature: float
    top_p: float | None

    @classmethod
    def of(cls, cfg_scale: float, temperature: float, top_p: float | None) -> SamplingPoint:
        return cls(cfg_scale=cfg_scale, temperature=temperature, top_p=None if top_p == FULL_TOP_P else top_p)

    @property
    def label(self) -> str:
        top_p = FULL_TOP_P if self.top_p is None else self.top_p
        return f"{self.cfg_scale:g}_t{self.temperature:g}_p{top_p:g}"


@dataclass(frozen=True)
class SampleFolder:
    kind: VariantKind
    style: Style
    point: SamplingPoint | None = None
    next_style: Style | None = None

    @classmethod
    def original(cls, style: Style) -> SampleFolder:
        return cls(kind=VariantKind.ORIGINAL, style=style)

    @classmethod
    def quantized(cls, style: Style) -> SampleFolder:
        return cls(kind=VariantKind.QUANTIZED, style=style)

    @classmethod
    def guided(cls, style: Style, point: SamplingPoint) -> SampleFolder:
        return cls(kind=VariantKind.GUIDED, style=style, point=point)

    @classmethod
    def switch(cls, style_switch: StyleSwitch) -> SampleFolder:
        return cls(kind=VariantKind.STYLE_SWITCH, style=style_switch.before, next_style=style_switch.after)

    @classmethod
    def fill(cls, style: Style) -> SampleFolder:
        return cls(kind=VariantKind.FILL, style=style)

    @classmethod
    def parse(cls, name: str) -> SampleFolder:
        for kind, pattern in FOLDER_PATTERNS.items():
            match = pattern.fullmatch(name)
            if match is None:
                continue
            style = Style(match["style"])
            match kind:
                case VariantKind.STYLE_SWITCH:
                    return cls.switch(StyleSwitch(before=style, after=Style(match["after"])))
                case VariantKind.GUIDED:
                    point = SamplingPoint.of(float(match["scale"]), float(match["temperature"]), float(match["top_p"]))
                    return cls.guided(style, point)
                case _:
                    return cls(kind=kind, style=style)
        raise ValueError(
            f"pasta de amostras {name!r} fora do contrato <estilo>[_quantizado|_cfg<s>_t<T>_p<P>], "
            "troca_<a>-<b>, virada_<estilo>"
        )

    @property
    def style_switch(self) -> StyleSwitch | None:
        return None if self.next_style is None else StyleSwitch(before=self.style, after=self.next_style)

    @property
    def variant(self) -> str:
        if self.point is not None:
            return f"{self.kind.value}{self.point.label}"
        if self.next_style is not None:
            return f"{self.kind.value}_{self.style.value}-{self.next_style.value}"
        return self.kind.value

    @property
    def name(self) -> str:
        match self.kind:
            case VariantKind.ORIGINAL:
                return self.style.value
            case VariantKind.QUANTIZED | VariantKind.GUIDED:
                return f"{self.style.value}_{self.variant}"
            case VariantKind.STYLE_SWITCH:
                return self.variant
            case VariantKind.FILL:
                return f"{self.kind.value}_{self.style.value}"


@dataclass(frozen=True)
class PathSettings:
    raw_dir: Path = PROJECT_ROOT / "data" / "raw" / "groove"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    runs_dir: Path = PROJECT_ROOT / "runs"

    @property
    def catalog_file(self) -> Path:
        return self.raw_dir / "info.csv"

    @property
    def tokenizer_file(self) -> Path:
        return self.processed_dir / "tokenizer.json"

    @property
    def pieces_file(self) -> Path:
        return self.processed_dir / "pieces.csv"

    @property
    def stats_file(self) -> Path:
        return self.processed_dir / "stats.json"

    def split_file(self, split: Split) -> Path:
        return self.processed_dir / f"{split.value}.npz"

    @property
    def reference_root(self) -> Path:
        return self.processed_dir / "reference"

    def reference_dir(self, split: Split, style: Style) -> Path:
        return self.reference_root / split.value / style.value


@dataclass(frozen=True)
class TokenizerSettings:
    use_microtiming: bool = True
    num_microtiming_bins: int = 16
    max_microtiming_shift: float = 0.125
    num_velocities: int = 16


@dataclass(frozen=True)
class PrepareSettings:
    pieces_per_split: int | None = None
    reference_bars: int = 8


@dataclass(frozen=True)
class DataSettings:
    context_length: int = 768
    null_style_probability: float = 0.2
    velocity_scale_min: float = 0.85
    velocity_scale_max: float = 1.15


@dataclass(frozen=True)
class ModelSettings:
    d_model: int = 256
    layers: int = 4
    heads: int = 4
    feedforward: int = 1024
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingSettings:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 300
    gradient_clip: float = 1.0
    max_epochs: int = 200
    patience: int = 15
    device: str = "cuda"


@dataclass(frozen=True)
class MarkovSettings:
    candidate_orders: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 8, 12, 16])
    order: int | None = None


@dataclass(frozen=True)
class StyleSwitch:
    before: Style
    after: Style


@dataclass(frozen=True)
class GenerationSettings:
    samples_per_style: int = 100
    first_seed: int = 0
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    cfg_scale: float = 3.0
    cfg_scale_sweep: list[float] = field(default_factory=list)
    temperature_sweep: list[float] = field(default_factory=list)
    top_p_sweep: list[float] = field(default_factory=list)
    style_switches: list[StyleSwitch] = field(default_factory=lambda: [StyleSwitch(Style.ROCK, Style.FUNK)])
    fill_styles: list[Style] = field(default_factory=list)
    batch_size: int = 100

    @property
    def sweep_points(self) -> list[SamplingPoint]:
        temperatures = self.temperature_sweep or [self.temperature]
        top_ps: list[float | None] = [*self.top_p_sweep] or [self.top_p]
        return [
            SamplingPoint.of(cfg_scale, temperature, top_p)
            for cfg_scale, temperature, top_p in itertools.product(self.cfg_scale_sweep, temperatures, top_ps)
        ]


@dataclass(frozen=True)
class RenderSettings:
    soundfont: Path = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")
    sample_rate: int = 22050


@dataclass(frozen=True)
class EvaluationSettings:
    reference_split: Split = Split.VAL
    output_dir: Path | None = None


@dataclass(frozen=True)
class DemoSettings:
    final_run: Path | None = None
    sweep_run: Path | None = None
    cfg_scales: list[float] = field(default_factory=lambda: [1.0, 1.5, 3.0])
    style_switches: list[StyleSwitch] = field(
        default_factory=lambda: [StyleSwitch(Style.ROCK, Style.FUNK), StyleSwitch(Style.JAZZ, Style.HIPHOP)]
    )
    output_dir: Path = PROJECT_ROOT


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int = 0
    paths: PathSettings = field(default_factory=PathSettings)
    tokenizer: TokenizerSettings = field(default_factory=TokenizerSettings)
    prepare: PrepareSettings = field(default_factory=PrepareSettings)
    data: DataSettings = field(default_factory=DataSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    markov:MarkovSettings = field(default_factory=MarkovSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    render: RenderSettings = field(default_factory=RenderSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    demo: DemoSettings = field(default_factory=DemoSettings)

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        with path.open() as file:
            data = yaml.safe_load(file)
        return ConfigLoader.build(cls, data, path.name)

    @classmethod
    def from_cli(cls, description: str, argv: list[str] | None = None) -> Self:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--config", type=Path, required=True)
        config_path = parser.parse_args(argv).config
        if not isinstance(config_path, Path):
            raise TypeError("--config precisa ser um caminho")
        return cls.from_yaml(config_path)


@dataclass(frozen=True)
class GitRevision:
    commit: str
    dirty: bool

    @classmethod
    def of(cls, directory: Path) -> GitRevision:
        head = cls._git(directory, "rev-parse", "HEAD")
        if head is None:
            return cls(commit="", dirty=False)
        status = cls._git(directory, "status", "--porcelain")
        return cls(commit=head, dirty=bool(status))

    @staticmethod
    def _git(directory: Path, *args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(directory), *args], capture_output=True, text=True, check=False
            )
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None


@dataclass(frozen=True)
class RunRecord:
    revision: GitRevision
    config: ExperimentConfig


@dataclass(frozen=True)
class ExperimentRun:
    path: Path
    record: RunRecord

    CONFIG_FILE = "config.yaml"
    SAMPLES_DIR = "samples"

    @property
    def config_file(self) -> Path:
        return self.path / self.CONFIG_FILE

    @property
    def samples_dir(self) -> Path:
        return self.path / self.SAMPLES_DIR

    def system_samples_dir(self, system: System) -> Path:
        return self.samples_dir / system.value

    def check_compatible(self, config: ExperimentConfig) -> None:
        recorded = self.record.config
        bars = config.prepare.reference_bars
        if recorded.prepare.reference_bars != bars:
            raise ValueError(
                f"{self.path}: run de {recorded.prepare.reference_bars} compassos, config com {bars} compassos"
            )
        if recorded.tokenizer != config.tokenizer:
            raise ValueError(f"{self.path}: tokenizer da run difere do tokenizer da config")

    @classmethod
    def create(cls, config: ExperimentConfig, now: datetime | None = None) -> ExperimentRun:
        stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M")
        config.paths.runs_dir.mkdir(parents=True, exist_ok=True)
        path = cls._claim(config.paths.runs_dir, f"{stamp}_{config.name}")
        run = cls(path=path, record=RunRecord(revision=GitRevision.of(PROJECT_ROOT), config=config))
        run.save()
        return run

    @classmethod
    def open(cls, path: Path) -> ExperimentRun:
        data = yaml.safe_load((path / cls.CONFIG_FILE).read_text())
        return cls(path=path, record=ConfigLoader.build(RunRecord, data, cls.CONFIG_FILE))

    def save(self) -> None:
        content = ConfigLoader.dump(self.record)
        self.config_file.write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True))

    @staticmethod
    def _claim(runs_dir: Path, base_name: str) -> Path:
        attempt = 1
        while True:
            path = runs_dir / (base_name if attempt == 1 else f"{base_name}_{attempt}")
            try:
                path.mkdir()
            except FileExistsError:
                attempt += 1
                continue
            return path


class ConfigLoader:
    @classmethod
    def build[T: DataclassInstance](cls, target: type[T], data: object, where: str) -> T:
        if not isinstance(data, dict):
            raise TypeError(f"{where}: esperado um mapeamento, veio {data!r}")
        hints = get_type_hints(target)
        known = {config_field.name for config_field in dataclasses.fields(target)}
        values: dict[str, object] = {}
        for key, value in data.items():
            if key not in known:
                raise ValueError(f"{where}.{key}: chave desconhecida")
            values[key] = cls._coerce(hints[key], value, f"{where}.{key}")
        return target(**values)

    @classmethod
    def dump(cls, instance: DataclassInstance) -> dict[str, object]:
        return {
            config_field.name: cls._plain(getattr(instance, config_field.name))
            for config_field in dataclasses.fields(instance)
        }

    @classmethod
    def _plain(cls, value: object) -> object:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return cls.dump(value)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value.relative_to(PROJECT_ROOT) if value.is_relative_to(PROJECT_ROOT) else value)
        if isinstance(value, list):
            return [cls._plain(item) for item in value]
        return value

    @classmethod
    def _coerce(cls, hint: object, value: object, where: str) -> object:
        if isinstance(hint, types.UnionType):
            return cls._coerce_union(hint, value, where)
        if get_origin(hint) is list:
            return cls._coerce_list(get_args(hint)[0], value, where)
        if isinstance(hint, type) and dataclasses.is_dataclass(hint):
            return cls.build(hint, value, where)
        if isinstance(hint, type) and issubclass(hint, Enum):
            return cls._coerce_enum(hint, value, where)
        if hint is Path and isinstance(value, str):
            path = Path(value)
            return path if path.is_absolute() else PROJECT_ROOT / path
        if hint is float and isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        if hint is int and isinstance(value, bool):
            raise TypeError(f"{where}: esperado int, veio {value!r}")
        if isinstance(hint, type) and isinstance(value, hint):
            return value
        raise TypeError(f"{where}: esperado {hint}, veio {value!r}")

    @classmethod
    def _coerce_list(cls, item_hint: object, value: object, where: str) -> list[object]:
        if not isinstance(value, list):
            raise TypeError(f"{where}: esperado uma lista, veio {value!r}")
        return [cls._coerce(item_hint, item, f"{where}[{index}]") for index, item in enumerate(value)]

    @staticmethod
    def _coerce_enum(hint: type[Enum], value: object, where: str) -> Enum:
        try:
            return hint(value)
        except ValueError:
            accepted = ", ".join(str(member.value) for member in hint)
            raise TypeError(f"{where}: esperado um de {accepted}, veio {value!r}") from None

    @classmethod
    def _coerce_union(cls, hint: types.UnionType, value: object, where: str) -> object:
        options = hint.__args__
        if value is None and type(None) in options:
            return None
        for option in options:
            if option is type(None):
                continue
            try:
                return cls._coerce(option, value, where)
            except TypeError:
                continue
        raise TypeError(f"{where}: esperado {hint}, veio {value!r}")
