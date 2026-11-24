from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

import torch
from tqdm import tqdm

from drumgen.config import (
    ExperimentConfig,
    ExperimentRun,
    GenerationSettings,
    PieceType,
    SampleFolder,
    Style,
    StyleSwitch,
    System,
    VariantKind,
)
from drumgen.markov import MarkovBaseline, MarkovTrainer
from drumgen.model import ComputeDevice, ModelCheckpoint
from drumgen.sampling import BarPlan, GenerationRequest, SpecialTokens, TokenSampler
from drumgen.tokenizer import DrumTokenizer

MANIFEST_FILE = "manifest.csv"
MIDI_SUFFIX = ".mid"
NEUTRAL_TEMPERATURE = 1.0
MAX_ATTEMPTS_PER_SAMPLE = 20
GENERATED_SYSTEMS = [System.TRANSFORMER, System.MARKOV]
CAPABILITIES = frozenset({VariantKind.GUIDED, VariantKind.STYLE_SWITCH, VariantKind.FILL})

MANIFEST_COLUMNS = [
    "system",
    "folder",
    "variant",
    "style",
    "seed",
    "cfg_scale",
    "temperature",
    "top_k",
    "top_p",
    "tokens",
    "ended_early",
    "dropped_notes",
    "file",
]


@dataclass(frozen=True)
class GeneratedSequence:
    ids: list[int]


class DrumGenerator(Protocol):
    @property
    def capabilities(self) -> frozenset[VariantKind]: ...

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GeneratedSequence]: ...


class MarkovGenerator:
    def __init__(self, baseline: MarkovBaseline) -> None:
        self._baseline = baseline

    @classmethod
    def load(cls, model: Path, tokenizer: DrumTokenizer) -> MarkovGenerator:
        return cls(MarkovBaseline.load(model / MarkovTrainer.MODEL_FILE, tokenizer))

    @property
    def capabilities(self) -> frozenset[VariantKind]:
        return frozenset()

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GeneratedSequence]:
        return [self._one(request) for request in requests]

    def _one(self, request: GenerationRequest) -> GeneratedSequence:
        styles = {bar.style for bar in request.plan}
        if len(styles) != 1 or any(bar.piece_type is not PieceType.GROOVE for bar in request.plan):
            raise ValueError("o Markov gera grooves de um estilo só, sem plano por compasso nem virada")
        sampling = (request.cfg_scale, request.temperature, request.top_k, request.top_p)
        if sampling != (None, NEUTRAL_TEMPERATURE, None, None):
            raise ValueError("o Markov amostra direto da cadeia, sem CFG, temperatura, top-k ou top-p")
        return GeneratedSequence(ids=self._baseline.sample(styles.pop(), request.seed, request.bars).ids)


class TransformerGenerator:
    CHECKPOINT_FILE = "best.pt"

    def __init__(self, sampler: TokenSampler) -> None:
        self._sampler = sampler

    @classmethod
    def load(cls, model: Path, tokenizer: DrumTokenizer, device: torch.device) -> TransformerGenerator:
        checkpoint_file = model / cls.CHECKPOINT_FILE
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"{checkpoint_file}: a run do Transformer precisa do {cls.CHECKPOINT_FILE} do treino")
        network = ModelCheckpoint.load(checkpoint_file).build_model().to(device).eval()
        return cls(TokenSampler(network, SpecialTokens.of(tokenizer), device))

    @property
    def capabilities(self) -> frozenset[VariantKind]:
        return frozenset({VariantKind.GUIDED, VariantKind.STYLE_SWITCH})

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GeneratedSequence]:
        return [GeneratedSequence(ids=ids) for ids in self._sampler.sample(requests)]


class GeneratorLoader:
    @staticmethod
    def load(system: System, model: Path, tokenizer: DrumTokenizer, device: torch.device) -> DrumGenerator:
        match system:
            case System.MARKOV:
                return MarkovGenerator.load(model, tokenizer)
            case System.TRANSFORMER:
                return TransformerGenerator.load(model, tokenizer, device)
            case _:
                raise ValueError(f"o sistema {system.value} não tem gerador ligado à geração em lote")


@dataclass(frozen=True)
class GenerationJob:
    folder: SampleFolder
    plan: tuple[BarPlan, ...]
    cfg_scale: float | None

    @classmethod
    def original(cls, style: Style, bars: int, cfg_scale: float | None) -> GenerationJob:
        return cls(SampleFolder.original(style), (BarPlan(style, PieceType.GROOVE),) * bars, cfg_scale)

    @classmethod
    def guided(cls, style: Style, bars: int, cfg_scale: float) -> GenerationJob:
        return cls(SampleFolder.guided(style, cfg_scale), (BarPlan(style, PieceType.GROOVE),) * bars, cfg_scale)

    @classmethod
    def switch(cls, style_switch: StyleSwitch, bars: int, cfg_scale: float | None) -> GenerationJob:
        half = bars // 2
        before = (BarPlan(style_switch.before, PieceType.GROOVE),) * half
        after = (BarPlan(style_switch.after, PieceType.GROOVE),) * (bars - half)
        return cls(SampleFolder.switch(style_switch), before + after, cfg_scale)

    @classmethod
    def fill(cls, style: Style, bars: int, cfg_scale: float | None) -> GenerationJob:
        plan = (BarPlan(style, PieceType.GROOVE),) * (bars - 1) + (BarPlan(style, PieceType.FILL),)
        return cls(SampleFolder.fill(style), plan, cfg_scale)

    def request(self, seed: int, settings: GenerationSettings) -> GenerationRequest:
        return GenerationRequest(
            plan=self.plan,
            seed=seed,
            cfg_scale=self.cfg_scale,
            temperature=settings.temperature,
            top_k=settings.top_k,
            top_p=settings.top_p,
        )


@dataclass(frozen=True)
class GenerationReport:
    manifest_file: Path
    attempts: dict[SampleFolder, int]
    warnings: list[str]


class GenerationManifest:
    def __init__(self, file: TextIO, system: System) -> None:
        self._writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        self._system = system
        self._writer.writeheader()

    def record(
        self, folder: SampleFolder, request: GenerationRequest, ids: Sequence[int], ended_early: bool, dropped: int, file: str
    ) -> None:
        self._writer.writerow(
            {
                "system": self._system.value,
                "folder": folder.name,
                "variant": folder.variant,
                "style": folder.style.value,
                "seed": request.seed,
                "cfg_scale": self._optional(request.cfg_scale),
                "temperature": request.temperature,
                "top_k": self._optional(request.top_k),
                "top_p": self._optional(request.top_p),
                "tokens": len(ids),
                "ended_early": ended_early,
                "dropped_notes": dropped,
                "file": file,
            }
        )

    @staticmethod
    def _optional(value: float | None) -> str:
        return "" if value is None else str(value)


class BatchGenerator:
    def __init__(
        self,
        generator: DrumGenerator,
        tokenizer: DrumTokenizer,
        settings: GenerationSettings,
        bars: int,
        system: System,
    ) -> None:
        self._generator = generator
        self._tokenizer = tokenizer
        self._settings = settings
        self._bars = bars
        self._system = system

    def run(self, output: Path) -> GenerationReport:
        if output.is_dir() and any(output.iterdir()):
            raise FileExistsError(f"{output}: já tem amostras; apague a pasta ou grave em outra run")
        jobs, warnings = self._jobs()
        output.mkdir(parents=True, exist_ok=True)
        manifest_file = output / MANIFEST_FILE
        attempts: dict[SampleFolder, int] = {}
        total = len(jobs) * self._settings.samples_per_style
        with manifest_file.open("w", newline="") as file, tqdm(total=total, desc="gerando") as progress:
            manifest = GenerationManifest(file, self._system)
            for job in jobs:
                attempts[job.folder] = self._run_job(job, output, manifest, progress.update)
        return GenerationReport(manifest_file=manifest_file, attempts=attempts, warnings=warnings)

    def _jobs(self) -> tuple[list[GenerationJob], list[str]]:
        capabilities = self._generator.capabilities
        unknown = capabilities - CAPABILITIES
        if unknown:
            names = ", ".join(sorted(kind.value for kind in unknown))
            raise ValueError(f"o gerador de {self._system.value} declara variantes que não são capacidades: {names}")
        settings = self._settings
        bars = self._bars
        guided = VariantKind.GUIDED in capabilities
        if guided and settings.cfg_scale_sweep:
            sweep = [
                GenerationJob.guided(style, bars, scale) for style in Style.evaluated() for scale in settings.cfg_scale_sweep
            ]
            return sweep, []
        cfg_scale = settings.cfg_scale if guided else None
        candidates: dict[VariantKind, list[GenerationJob]] = {
            VariantKind.STYLE_SWITCH: [GenerationJob.switch(switch, bars, cfg_scale) for switch in settings.style_switches],
            VariantKind.FILL: [GenerationJob.fill(style, bars, cfg_scale) for style in settings.fill_styles],
        }
        jobs = [GenerationJob.original(style, bars, cfg_scale) for style in Style.evaluated()]
        warnings: list[str] = []
        for kind, kind_jobs in candidates.items():
            if kind in capabilities:
                jobs += kind_jobs
            elif kind_jobs:
                names = ", ".join(dict.fromkeys(job.folder.name for job in kind_jobs))
                warnings.append(f"{self._system.value} não suporta a variante {kind.value}; fora da geração: {names}")
        return jobs, warnings

    def _run_job(
        self, job: GenerationJob, output: Path, manifest: GenerationManifest, advance: Callable[[], object]
    ) -> int:
        target = self._settings.samples_per_style
        limit = target * MAX_ATTEMPTS_PER_SAMPLE
        seed = self._settings.first_seed
        written = 0
        attempts = 0
        while written < target:
            if attempts >= limit:
                raise RuntimeError(
                    f"{self._system.value}/{job.folder.name}: {attempts - written} de {attempts} tentativas terminaram "
                    f"em EOS antes de {self._bars} compassos"
                )
            size = min(self._settings.batch_size, target - written, limit - attempts)
            requests = [job.request(seed + offset, self._settings) for offset in range(size)]
            generated = self._generator.generate(requests)
            attempts += size
            seed += size
            for request, sequence in zip(requests, generated, strict=True):
                if self._record(job, request, sequence.ids, output, manifest):
                    written += 1
                    advance()
        return attempts

    def _record(
        self, job: GenerationJob, request: GenerationRequest, ids: Sequence[int], output: Path, manifest: GenerationManifest
    ) -> bool:
        complete = self._tokenizer.eos_id not in ids
        if complete:
            self._check_bars(job.folder, request, ids)
        self._save(job.folder, request, ids, output, manifest, complete)
        if complete and job.folder.kind is VariantKind.ORIGINAL:
            quantized = SampleFolder.quantized(job.folder.style)
            self._save(quantized, request, self._tokenizer.quantize(ids), output, manifest, True)
        return complete

    def _check_bars(self, folder: SampleFolder, request: GenerationRequest, ids: Sequence[int]) -> None:
        bars = self._tokenizer.count_bars(ids)
        if bars != request.bars:
            raise ValueError(
                f"{self._system.value}/{folder.name}: a seed {request.seed} terminou sem EOS com {bars} compassos, "
                f"esperados {request.bars}"
            )

    def _save(
        self,
        folder: SampleFolder,
        request: GenerationRequest,
        ids: Sequence[int],
        output: Path,
        manifest: GenerationManifest,
        complete: bool,
    ) -> None:
        decoded = self._tokenizer.decode_with_dropped_notes(ids)
        file = ""
        if complete:
            target = output / folder.name / f"{request.seed}{MIDI_SUFFIX}"
            target.parent.mkdir(parents=True, exist_ok=True)
            decoded.score.dump_midi(target)
            file = target.relative_to(output).as_posix()
        manifest.record(folder, request, ids, not complete, decoded.dropped_notes, file)


class Generation:
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    def run(self, system: System, model: Path, run: Path) -> GenerationReport:
        if not (run / ExperimentRun.CONFIG_FILE).is_file():
            raise FileNotFoundError(f"{run}: a run de saída precisa existir, com o {ExperimentRun.CONFIG_FILE} dela")
        output = ExperimentRun.open(run)
        for experiment in (ExperimentRun.open(model), output):
            experiment.check_compatible(self._config)
        tokenizer = DrumTokenizer.from_file(self._config.paths.tokenizer_file)
        device = ComputeDevice.resolve(self._config.training.device)
        generator = GeneratorLoader.load(system, model, tokenizer, device)
        batch = BatchGenerator(
            generator, tokenizer, self._config.generation, self._config.prepare.reference_bars, system
        )
        return batch.run(output.system_samples_dir(system))


@dataclass(frozen=True)
class GenerateCommand:
    config: ExperimentConfig
    system: System
    model: Path
    run: Path

    @classmethod
    def parse(cls, argv: Sequence[str]) -> GenerateCommand:
        parser = argparse.ArgumentParser(
            description="Gera as amostras de um sistema por estilo e grava em <run>/samples/<sistema>/ com o manifest"
        )
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--system", required=True, choices=[system.value for system in GENERATED_SYSTEMS])
        parser.add_argument("--model", type=Path, required=True)
        parser.add_argument("--run", type=Path)
        arguments = parser.parse_args(list(argv))
        config_file = arguments.config
        system = arguments.system
        model = arguments.model
        run = arguments.run or model
        if not isinstance(config_file, Path) or not isinstance(model, Path) or not isinstance(run, Path):
            raise TypeError("--config, --model e --run precisam ser caminhos")
        if not isinstance(system, str):
            raise TypeError("--system precisa ser um nome de sistema")
        return cls(config=ExperimentConfig.from_yaml(config_file), system=System(system), model=model, run=run)


if __name__ == "__main__":
    command = GenerateCommand.parse(sys.argv[1:])
    report = Generation(command.config).run(command.system, command.model, command.run)
    for warning in report.warnings:
        print(f"aviso: {warning}")
    for folder, attempts in report.attempts.items():
        print(f"{folder.name}: {attempts} tentativas")
    print(f"manifest gravado em {report.manifest_file}")
