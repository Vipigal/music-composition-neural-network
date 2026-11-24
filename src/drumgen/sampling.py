from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from drumgen.config import PieceType, Style
from drumgen.model import AttentionCache, DrumTransformer
from drumgen.tokenizer import DrumTokenizer

NEUTRAL_GUIDANCE = 1.0


class ConditionalModel(Protocol):
    @property
    def context_length(self) -> int: ...

    def __call__(self, tokens: torch.Tensor, styles: torch.Tensor, types: torch.Tensor) -> torch.Tensor: ...


# Plano por compasso: cada compasso tem seu estilo e tipo (ex: rock 1-4, funk 5-8)
@dataclass(frozen=True)
class BarPlan:
    style: Style
    piece_type: PieceType


@dataclass(frozen=True)
class GenerationRequest:
    plan: tuple[BarPlan, ...]
    seed: int
    cfg_scale: float | None
    temperature: float
    top_k: int | None
    top_p: float | None

    @property
    def bars(self) -> int:
        return len(self.plan)

    @property
    def shared_settings(self) -> tuple[int, float | None, float, int | None, float | None]:
        return (self.bars, self.cfg_scale, self.temperature, self.top_k, self.top_p)

    @property
    def guided(self) -> bool:
        return self.cfg_scale is not None and self.cfg_scale != NEUTRAL_GUIDANCE


@dataclass(frozen=True)
class SpecialTokens:
    pad: int
    bos: int
    eos: int
    bar: int

    @classmethod
    def of(cls, tokenizer: DrumTokenizer) -> SpecialTokens:
        return cls(pad=tokenizer.pad_id, bos=tokenizer.bos_id, eos=tokenizer.eos_id, bar=tokenizer.bar_id)


@dataclass(frozen=True)
class LogitFilter:
    temperature: float
    top_k: int | None
    top_p: float | None

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        # temperatura: < 1 deixa a distribuição mais "pontuda", > 1 mais espalhada
        logits = logits / self.temperature

        # top-k: zera (manda pra -inf) tudo que não está entre os k maiores
        if self.top_k is not None and self.top_k < logits.shape[-1]:
            kth = torch.topk(logits, self.top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, -torch.inf)

        # top-p (nucleus): fica só com os tokens mais prováveis até somar p de probabilidade
        if self.top_p is not None:
            sorted_logits, idx = torch.sort(logits, dim=-1, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            # massa acumulada ANTES de cada token; o token que cruza p ainda entra
            mass_before = torch.cumsum(probs, dim=-1) - probs
            outside = mass_before >= self.top_p
            outside[..., 0] = False  # o mais provável nunca sai
            # volta a máscara pra ordem original do vocabulário
            logits = logits.masked_fill(outside.scatter(-1, idx, outside), -torch.inf)
        return logits


# CFG: parte do logit sem estilo e anda na direção do condicionado.
# s = 1 -> igual ao condicionado; s > 1 -> exagera o estilo.
class ClassifierFreeGuidance:
    @staticmethod
    def combine(conditional: torch.Tensor, unconditional: torch.Tensor, scale: float) -> torch.Tensor:
        return unconditional + scale * (conditional - unconditional)


# Sorteio manual em vez de torch.multinomial: cada amostra usa um uniforme do seu
# próprio gerador (sua seed). Assim a mesma seed dá a mesma amostra sozinha ou num lote.
class CategoricalDraw:
    @staticmethod
    def draw(probabilities: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
        cumulative = torch.cumsum(probabilities, dim=-1)
        targets = (uniforms.to(cumulative) * cumulative[:, -1]).unsqueeze(-1)
        return torch.searchsorted(cumulative, targets, right=True).squeeze(-1)


class TokenSampler:
    MAX_TOKENS_PER_BAR = 256

    def __init__(self, model: ConditionalModel, special: SpecialTokens, device: torch.device) -> None:
        self._model = model
        self._cached_model = model if isinstance(model, DrumTransformer) else None
        self._special = special
        self._device = device

    def sample(self, requests: Sequence[GenerationRequest]) -> list[list[int]]:
        if not requests:
            return []
        if len({request.shared_settings for request in requests}) != 1:
            raise ValueError("um lote precisa de compassos, CFG, temperatura, top-k e top-p iguais")
        with torch.inference_mode():
            return self._sample(requests)

    # Laço principal da geração (autoregressiva, um token por vez, em lote).
    def _sample(self, requests: Sequence[GenerationRequest]) -> list[list[int]]:
        first = requests[0]
        n_bars = first.bars
        limit = n_bars * self.MAX_TOKENS_PER_BAR  # trava de segurança caso o modelo nunca feche os compassos
        logit_filter = LogitFilter(first.temperature, first.top_k, first.top_p)
        generators = [torch.Generator().manual_seed(req.seed) for req in requests]

        # ids = saída de cada amostra (lista python); tokens/styles/types = tensores só das ativas
        ids = [[self._special.bos] for _ in requests]
        bars_seen = [0] * len(requests)  # quantos Bar_None cada amostra já gerou
        active = list(range(len(requests)))
        tokens = torch.full((len(requests), 1), self._special.bos, dtype=torch.long, device=self._device)
        styles = self._column([req.plan[0].style.index for req in requests])
        types = self._column([req.plan[0].piece_type.index for req in requests])

        cfg_scale = first.cfg_scale if first.guided else None
        cache = None
        if self._cached_model is not None:
            # com CFG o lote dobra (metade com estilo, metade com NULL), então o cache também
            cache = self._cached_model.empty_cache(len(requests) * (1 if cfg_scale is None else 2), self._device)

        while active:
            # 1) logits do próximo token -> filtros -> probabilidades -> sorteio
            logits = self._next_logits(tokens, styles, types, cfg_scale, cache)
            probs = torch.softmax(logit_filter.apply(logits), dim=-1)
            uniforms = torch.stack([torch.rand((), generator=generators[row]) for row in active])
            drawn = CategoricalDraw.draw(probs, uniforms).tolist()

            # 2) decide, amostra por amostra, se continua ou para
            kept: list[int] = []
            appended: list[int] = []
            next_styles: list[int] = []
            next_types: list[int] = []
            for i, row in enumerate(active):
                token = int(drawn[i])
                if token == self._special.bar:
                    bars_seen[row] += 1
                # o Bar_None de número n_bars marca o fim e NÃO entra na saída
                if bars_seen[row] == n_bars:
                    continue
                ids[row].append(token)
                if token == self._special.eos:
                    continue  # EOS antes da hora: para (quem chama decide se refaz)
                if len(ids[row]) == limit:
                    ids[row].append(self._special.eos)
                    continue
                # o compasso atual define estilo/tipo do próximo token (plano por compasso)
                bar = requests[row].plan[bars_seen[row]]
                kept.append(i)
                appended.append(token)
                next_styles.append(bar.style.index)
                next_types.append(bar.piece_type.index)

            # 3) tira do lote (e do cache) as amostras que terminaram
            if len(kept) < len(active):
                keep_idx = torch.tensor(kept, dtype=torch.long, device=self._device)
                tokens, styles, types = tokens[keep_idx], styles[keep_idx], types[keep_idx]
                if cache is not None:
                    # com CFG as linhas NULL ficam na segunda metade, deslocadas de len(active)
                    cache.keep_rows(keep_idx if cfg_scale is None else torch.cat([keep_idx, keep_idx + len(active)]))
                active = [active[i] for i in kept]

            # 4) anexa o token sorteado como nova coluna e segue
            if active:
                tokens = torch.cat([tokens, self._column(appended)], dim=1)
                styles = torch.cat([styles, self._column(next_styles)], dim=1)
                types = torch.cat([types, self._column(next_types)], dim=1)
        return ids

    def _column(self, values: Sequence[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long, device=self._device).unsqueeze(1)

    def _next_logits(
        self,
        tokens: torch.Tensor,
        styles: torch.Tensor,
        types: torch.Tensor,
        cfg_scale: float | None,
        cache: AttentionCache | None,
    ) -> torch.Tensor:
        # CFG: duplica o lote, a segunda metade com estilo NULL. Assim os dois forwards
        # (condicionado e incondicional) saem numa chamada só.
        if cfg_scale is not None:
            null = torch.full_like(styles, Style.NULL.index)
            tokens, styles, types = torch.cat([tokens, tokens]), torch.cat([styles, null]), torch.cat([types, types])

        if self._cached_model is not None and cache is not None and tokens.shape[1] <= cache.capacity:
            # caminho rápido: passa só o que ainda não está no cache (o último token)
            start = cache.length
            logits = self._cached_model.forward_cached(tokens[:, start:], styles[:, start:], types[:, start:], cache)
        else:
            # passou do contexto (só na demo longa): janela deslizante com os últimos N tokens, sem cache
            ctx = self._model.context_length
            logits = self._model(tokens[:, -ctx:], styles[:, -ctx:], types[:, -ctx:])

        last = logits[:, -1].float()  # só interessa a última posição
        if cfg_scale is None:
            return last
        half = last.shape[0] // 2
        return ClassifierFreeGuidance.combine(last[:half], last[half:], cfg_scale)
