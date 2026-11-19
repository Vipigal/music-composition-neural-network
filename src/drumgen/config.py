from __future__ import annotations

import argparse
import dataclasses
import types
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Self, get_type_hints

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
class ExperimentConfig:
    name: str
    seed: int = 0
    paths: PathSettings = field(default_factory=PathSettings)
    tokenizer: TokenizerSettings = field(default_factory=TokenizerSettings)
    prepare: PrepareSettings = field(default_factory=PrepareSettings)
    data: DataSettings = field(default_factory=DataSettings)

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
    def _coerce(cls, hint: object, value: object, where: str) -> object:
        if isinstance(hint, types.UnionType):
            return cls._coerce_union(hint, value, where)
        if isinstance(hint, type) and dataclasses.is_dataclass(hint):
            return cls.build(hint, value, where)
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
