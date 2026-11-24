from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from drumgen.config import ModelSettings, PieceType, Style


class ComputeDevice:
    @staticmethod
    def resolve(name: str) -> torch.device:
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("training.device pede cuda, mas não há GPU visível; use device: cpu na config")
        return device


# Cache de K/V usado só na geração: guarda as chaves e valores de cada camada
# pra não recalcular a atenção do prefixo inteiro a cada token novo.
@dataclass
class AttentionCache:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    length: int = 0

    @property
    def capacity(self) -> int:
        return self.keys[0].shape[2]

    # quando uma amostra do lote termina, a linha dela sai do cache também
    def keep_rows(self, rows: torch.Tensor) -> None:
        self.keys = [keys[rows] for keys in self.keys]
        self.values = [values[rows] for values in self.values]


class DrumTransformer(nn.Module):
    def __init__(self, settings: ModelSettings, vocab_size: int, context_length: int, pad_id: int) -> None:
        super().__init__()
        self.settings = settings
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.pad_id = pad_id

        # 4 embeddings que são somados na entrada (ver _embed).
        # style tem 8 valores: 6 estilos + other + NULL (o NULL é o "sem estilo" do CFG)
        self.token_embedding = nn.Embedding(vocab_size, settings.d_model)
        self.position_embedding = nn.Embedding(context_length, settings.d_model)
        self.style_embedding = nn.Embedding(len(Style), settings.d_model)
        self.type_embedding = nn.Embedding(len(PieceType), settings.d_model)
        self.dropout = nn.Dropout(settings.dropout)

        # É um "decoder-only" estilo GPT: usa o TransformerEncoder do torch com máscara causal.
        # norm_first=True -> pre-LN, que treina mais estável que o post-LN original.
        layer = nn.TransformerEncoderLayer(
            d_model=settings.d_model,
            nhead=settings.heads,
            dim_feedforward=settings.feedforward,
            dropout=settings.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.body = nn.TransformerEncoder(layer, num_layers=settings.layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(settings.d_model)
        self.head = nn.Linear(settings.d_model, vocab_size)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    # forward "normal", usado no treino: recebe a sequência toda e devolve os logits
    # de todas as posições de uma vez. Saída: (batch, T, vocab)
    def forward(self, tokens: torch.Tensor, styles: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        T = tokens.shape[1]
        if T > self.context_length:
            raise ValueError(f"sequência de {T} tokens passa do contexto de {self.context_length}")
        x = self._embed(tokens, styles, types, start=0)
        # máscara causal: a posição t só enxerga 0..t (senão o modelo "cola" vendo o futuro)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=tokens.device)
        x = self.body(self.dropout(x), mask=mask, is_causal=True)
        return self.head(self.final_norm(x))

    def empty_cache(self, rows: int, device: torch.device) -> AttentionCache:
        heads = self.settings.heads
        shape = (rows, heads, self.context_length, self.settings.d_model // heads)
        layers = range(self.settings.layers)
        return AttentionCache(
            keys=[torch.zeros(shape, device=device) for _ in layers],
            values=[torch.zeros(shape, device=device) for _ in layers],
        )

    # Versão incremental do forward, só pra inferência. Processa só os tokens novos
    # (normalmente 1) e reaproveita K/V do cache. Tem que dar o mesmo resultado do
    # forward() acima -- por isso refaz a camada "na mão" com os mesmos pesos.
    def forward_cached(
        self, tokens: torch.Tensor, styles: torch.Tensor, types: torch.Tensor, cache: AttentionCache
    ) -> torch.Tensor:
        if self.training:
            raise RuntimeError("a decodificação com cache é só para inferência; chame eval() antes")
        start = cache.length
        end = start + tokens.shape[1]
        if end > cache.capacity:
            raise ValueError(f"o cache comporta {cache.capacity} tokens; pedido até a posição {end}")
        x = self._embed(tokens, styles, types, start=start)
        for i, layer in enumerate(self.body.layers):
            if not isinstance(layer, nn.TransformerEncoderLayer):
                raise TypeError("o corpo do modelo precisa ser de TransformerEncoderLayer")
            # pre-LN: x = x + attn(norm1(x)); x = x + ffn(norm2(x))
            x = x + self._cached_attention(layer, x, cache, i, start, end)
            x = x + layer.linear2(functional.gelu(layer.linear1(layer.norm2(x))))
        cache.length = end
        return self.head(self.final_norm(x))

    # Entrada de cada posição = token + posição + estilo + tipo (groove/virada).
    # styles e types têm um valor por token, então dá pra trocar o estilo no meio
    # da sequência (é assim que funciona o plano por compasso da geração).
    def _embed(self, tokens: torch.Tensor, styles: torch.Tensor, types: torch.Tensor, start: int) -> torch.Tensor:
        pos = torch.arange(start, start + tokens.shape[1], device=tokens.device)
        return (
            self.token_embedding(tokens)
            + self.position_embedding(pos)
            + self.style_embedding(styles)
            + self.type_embedding(types)
        )

    def _cached_attention(
        self, layer: nn.TransformerEncoderLayer, hidden: torch.Tensor, cache: AttentionCache, index: int, start: int, end: int
    ) -> torch.Tensor:
        attn = layer.self_attn
        B, T, C = hidden.shape
        n_heads = self.settings.heads
        # mesma projeção Q/K/V que o MultiheadAttention faz por dentro
        qkv = functional.linear(layer.norm1(hidden), attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = (part.view(B, T, n_heads, C // n_heads).transpose(1, 2) for part in qkv.chunk(3, dim=-1))

        # escreve K e V dos tokens novos no cache, nas posições start..end
        cache.keys[index][:, :, start:end] = k
        cache.values[index][:, :, start:end] = v

        # os tokens novos olham pra tudo que já está no cache + eles mesmos (causal)
        visible = torch.ones((T, end), dtype=torch.bool, device=hidden.device).tril(diagonal=start)
        out = functional.scaled_dot_product_attention(
            q, cache.keys[index][:, :, :end], cache.values[index][:, :, :end], attn_mask=visible
        )
        return attn.out_proj(out.transpose(1, 2).reshape(B, T, C))


# O que vai pro best.pt / last.pt: pesos + o necessário pra reconstruir o modelo
# sem precisar da config original.
@dataclass(frozen=True)
class ModelCheckpoint:
    settings: ModelSettings
    vocab_size: int
    context_length: int
    pad_id: int
    styles: list[str]
    tokenizer_file: Path
    epoch: int
    state: dict[str, torch.Tensor]

    @classmethod
    def of(cls, model: DrumTransformer, tokenizer_file: Path, epoch: int) -> ModelCheckpoint:
        return cls(
            settings=model.settings,
            vocab_size=model.vocab_size,
            context_length=model.context_length,
            pad_id=model.pad_id,
            styles=[style.value for style in Style],
            tokenizer_file=tokenizer_file,
            epoch=epoch,
            state={name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
        )

    def save(self, path: Path) -> None:
        torch.save(
            {
                "settings": dataclasses.asdict(self.settings),
                "vocab_size": self.vocab_size,
                "context_length": self.context_length,
                "pad_id": self.pad_id,
                "styles": self.styles,
                "tokenizer_file": str(self.tokenizer_file),
                "epoch": self.epoch,
                "state": self.state,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> ModelCheckpoint:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(data, dict):
            raise TypeError(f"{path}: checkpoint fora do formato esperado")
        styles = [style.value for style in Style]
        if data["styles"] != styles:
            raise ValueError(f"{path}: estilos do checkpoint {data['styles']} diferem dos atuais {styles}")
        return cls(
            settings=ModelSettings(**data["settings"]),
            vocab_size=int(data["vocab_size"]),
            context_length=int(data["context_length"]),
            pad_id=int(data["pad_id"]),
            styles=list(data["styles"]),
            tokenizer_file=Path(data["tokenizer_file"]),
            epoch=int(data["epoch"]),
            state=data["state"],
        )

    def build_model(self) -> DrumTransformer:
        model = DrumTransformer(self.settings, self.vocab_size, self.context_length, self.pad_id)
        model.load_state_dict(self.state)
        return model
