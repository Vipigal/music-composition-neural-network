from __future__ import annotations

import dataclasses
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

    @classmethod
    def from_gmd(cls, gmd_style: str) -> Style:
        genre = gmd_style.split("/")[0]
        by_genre = {style.value: style for style in cls.evaluated()}
        return by_genre.get(genre, cls.OTHER)


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


@dataclass(frozen=True)
class PathSettings:
    raw_dir: Path = PROJECT_ROOT / "data" / "raw" / "groove"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    runs_dir: Path = PROJECT_ROOT / "runs"


@dataclass(frozen=True)
class TokenizerSettings:
    use_microtiming: bool = True
    num_microtiming_bins: int = 16
    max_microtiming_shift: float = 0.125
    num_velocities: int = 16


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    paths: PathSettings = field(default_factory=PathSettings)
    tokenizer: TokenizerSettings = field(default_factory=TokenizerSettings)

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        with path.open() as file:
            data = yaml.safe_load(file)
        return ConfigLoader.build(cls, data, path.name)


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
