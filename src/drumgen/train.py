from __future__ import annotations

import csv
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch
from matplotlib import pyplot
from torch.nn import functional
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from drumgen.config import ExperimentConfig, ExperimentRun, Split, Style, TrainingSettings
from drumgen.dataset import Corpus, EvaluationWindows, TrainingWindows, Window, WindowBatch
from drumgen.model import ComputeDevice, DrumTransformer, ModelCheckpoint
from drumgen.render import ChartPalette
from drumgen.tokenizer import DrumTokenizer

REPORTED_STYLES = [*Style.evaluated(), Style.OTHER]


# LR: sobe linear durante o warmup e depois cai em cosseno até 0 no último passo.
# factor() devolve o multiplicador do LR base (o LambdaLR usa isso).
@dataclass(frozen=True)
class WarmupCosineSchedule:
    warmup_steps: int
    total_steps: int

    def factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / decay_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))


class TrainingStep:
    def __init__(self, model: DrumTransformer, optimizer: AdamW, scheduler: LambdaLR, gradient_clip: float, device: torch.device) -> None:
        self.model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._gradient_clip = gradient_clip
        self._device = device

    @classmethod
    def create(
        cls, model: DrumTransformer, settings: TrainingSettings, total_steps: int, device: torch.device
    ) -> TrainingStep:
        model.to(device)
        # weight decay só nas matrizes (pesos de linear/embedding); bias e LayerNorm ficam sem
        decay = [p for p in model.parameters() if p.ndim >= 2]
        no_decay = [p for p in model.parameters() if p.ndim < 2]
        optimizer = AdamW(
            [{"params": decay, "weight_decay": settings.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=settings.learning_rate,
        )
        schedule = WarmupCosineSchedule(settings.warmup_steps, total_steps)
        return cls(model, optimizer, LambdaLR(optimizer, schedule.factor), settings.gradient_clip, device)

    @property
    def learning_rate(self) -> float:
        return float(self._scheduler.get_last_lr()[0])

    # Um passo de treino. targets = tokens deslocados de 1 (próximo token), montados no dataset.
    def run(self, batch: WindowBatch) -> float:
        self.model.train()
        batch = batch.to(self._device)
        logits = self.model(batch.tokens, batch.styles, batch.types)
        # achata (B, T, V) -> (B*T, V) e ignora o PAD na loss
        loss = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), batch.targets.reshape(-1), ignore_index=self.model.pad_id
        )
        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # corta gradiente muito grande (evita explodir no começo do treino)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._gradient_clip)
        self._optimizer.step()
        self._scheduler.step()  # o LR muda a cada passo, não a cada época
        return float(loss.detach())


# Loss de validação geral e por estilo. É média por TOKEN (soma das losses / nº de
# tokens não-PAD), não média de médias por batch.
@dataclass(frozen=True)
class ValidationLoss:
    overall: float
    per_style: dict[Style, float]

    @classmethod
    def measure(
        cls, model: DrumTransformer, windows: Dataset[Window], pad_id: int, batch_size: int, device: torch.device
    ) -> ValidationLoss:
        loader = DataLoader(windows, batch_size=batch_size, collate_fn=partial(WindowBatch.collate, pad_id=pad_id))
        totals: dict[Style, float] = defaultdict(float)
        counts: dict[Style, int] = defaultdict(int)
        styles = list(Style)
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch.tokens, batch.styles, batch.types)
                losses = functional.cross_entropy(
                    logits.transpose(1, 2), batch.targets, ignore_index=pad_id, reduction="none"
                )
                scored = batch.targets != pad_id
                row_totals = (losses * scored).sum(dim=1).tolist()
                row_counts = scored.sum(dim=1).tolist()
                # o estilo da janela é o da primeira posição
                for style_index, total, count in zip(batch.styles[:, 0].tolist(), row_totals, row_counts, strict=True):
                    totals[styles[style_index]] += total
                    counts[styles[style_index]] += count
        scored_styles = [style for style in totals if counts[style] > 0]
        return cls(
            overall=sum(totals.values()) / sum(counts.values()),
            per_style={style: totals[style] / counts[style] for style in scored_styles},
        )


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    validation: ValidationLoss
    learning_rate: float
    seconds: float

    def row(self) -> dict[str, str]:
        per_style = {
            f"val_{style.value}": "" if style not in self.validation.per_style else f"{self.validation.per_style[style]:.6f}"
            for style in REPORTED_STYLES
        }
        return {
            "epoch": str(self.epoch),
            "train_loss": f"{self.train_loss:.6f}",
            "val_loss": f"{self.validation.overall:.6f}",
            **per_style,
            "learning_rate": f"{self.learning_rate:.8f}",
            "seconds": f"{self.seconds:.2f}",
        }


class TrainingHistory:
    COLUMNS = [
        "epoch",
        "train_loss",
        "val_loss",
        *(f"val_{style.value}" for style in REPORTED_STYLES),
        "learning_rate",
        "seconds",
    ]

    def __init__(self, path: Path) -> None:
        self._path = path
        self.records: list[EpochRecord] = []
        with path.open("w", newline="") as file:
            csv.DictWriter(file, fieldnames=self.COLUMNS).writeheader()

    def append(self, record: EpochRecord) -> None:
        self.records.append(record)
        with self._path.open("a", newline="") as file:
            csv.DictWriter(file, fieldnames=self.COLUMNS).writerow(record.row())

    def plot(self, path: Path, best_epoch: int) -> None:
        epochs = [record.epoch for record in self.records]
        figure, axes = pyplot.subplots(figsize=(6.0, 3.6), facecolor=ChartPalette.SURFACE)
        axes.set_facecolor(ChartPalette.SURFACE)
        axes.plot(epochs, [record.train_loss for record in self.records], color=ChartPalette.SERIES_1, label="treino")
        axes.plot(
            epochs, [record.validation.overall for record in self.records], color=ChartPalette.SERIES_2, label="validação"
        )
        axes.axvline(best_epoch, color=ChartPalette.BASELINE, linestyle="--", linewidth=1.0, label="melhor época")
        axes.set_xlabel("época", color=ChartPalette.INK_SECONDARY)
        axes.set_ylabel("entropia cruzada por token (nat)", color=ChartPalette.INK_SECONDARY)
        axes.grid(color=ChartPalette.GRIDLINE, linewidth=0.6)
        for spine in ("top", "right"):
            axes.spines[spine].set_visible(False)
        axes.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        pyplot.close(figure)


class TrainingRun:
    HISTORY_FILE = "history.csv"
    CURVES_FILE = "curves.png"
    BEST_FILE = "best.pt"
    LAST_FILE = "last.pt"

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    # Treino completo: monta dados e modelo, roda as épocas com early stopping
    # e salva tudo na pasta do run.
    def run(self) -> ExperimentRun:
        config = self._config
        settings = config.training
        device = ComputeDevice.resolve(settings.device)
        torch.manual_seed(config.seed)
        tokenizer = DrumTokenizer.from_file(config.paths.tokenizer_file)
        training_windows = TrainingWindows(
            Corpus.load(config.paths, Split.TRAIN), tokenizer, config.data, config.seed
        )
        validation_windows = EvaluationWindows(Corpus.load(config.paths, Split.VAL), tokenizer, config.data)
        loader = DataLoader(
            training_windows,
            batch_size=settings.batch_size,
            collate_fn=partial(WindowBatch.collate, pad_id=tokenizer.pad_id),
            num_workers=0,
        )
        model = DrumTransformer(config.model, tokenizer.vocab_size, config.data.context_length, tokenizer.pad_id)
        step = TrainingStep.create(model, settings, settings.max_epochs * len(loader), device)
        run = ExperimentRun.create(config)
        history = TrainingHistory(run.path / self.HISTORY_FILE)
        print(f"run: {run.path} · {model.parameter_count:,} parâmetros · {len(loader)} passos por época")

        # ---- laço de épocas ----
        best_loss = math.inf
        best_epoch = 0
        for epoch in range(1, settings.max_epochs + 1):
            t0 = time.perf_counter()
            # muda a semente das janelas: outro recorte, outra escala de velocity e outro dropout de estilo
            training_windows.set_epoch(epoch)
            train_losses = [step.run(batch) for batch in loader]
            val = ValidationLoss.measure(model, validation_windows, tokenizer.pad_id, settings.batch_size, device)
            record = EpochRecord(
                epoch=epoch,
                train_loss=sum(train_losses) / len(train_losses),
                validation=val,
                learning_rate=step.learning_rate,
                seconds=time.perf_counter() - t0,
            )
            history.append(record)

            # last.pt sempre; best.pt só quando a validação melhora
            ModelCheckpoint.of(model, config.paths.tokenizer_file, epoch).save(run.path / self.LAST_FILE)
            if val.overall < best_loss:
                best_loss, best_epoch = val.overall, epoch
                ModelCheckpoint.of(model, config.paths.tokenizer_file, epoch).save(run.path / self.BEST_FILE)
            print(
                f"época {epoch}: treino {record.train_loss:.4f} · validação {val.overall:.4f} "
                f"· melhor {best_loss:.4f} (época {best_epoch}) · {record.seconds:.1f} s"
            )
            # early stopping: para se ficou `patience` épocas sem melhorar
            if epoch - best_epoch >= settings.patience:
                break
        history.plot(run.path / self.CURVES_FILE, best_epoch)
        return run


if __name__ == "__main__":
    TrainingRun(ExperimentConfig.from_cli("Treina o Transformer condicionado por estilo e tipo")).run()
