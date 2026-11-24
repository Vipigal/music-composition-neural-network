import math
from collections.abc import Callable

import pytest
import torch

from drumgen.config import ModelSettings, PieceType, Style
from drumgen.model import DrumTransformer
from drumgen.sampling import (
    BarPlan,
    CategoricalDraw,
    ConditionalModel,
    ClassifierFreeGuidance,
    GenerationRequest,
    LogitFilter,
    SpecialTokens,
    TokenSampler,
)

VOCAB = 12
SPECIAL = SpecialTokens(pad=0, bos=1, eos=2, bar=3)
NOTE = 5
OTHER_NOTE = 6


class ScriptedModel:
    def __init__(self, context_length: int, choose: Callable[[list[int]], int]) -> None:
        self.context_length = context_length
        self._choose = choose
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(self, tokens: torch.Tensor, styles: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        self.calls.append((tokens.clone(), styles.clone()))
        logits = torch.zeros((*tokens.shape, VOCAB))
        for row in range(tokens.shape[0]):
            logits[row, -1, self._choose(tokens[row].tolist())] = 100.0
        return logits


class UniformModel:
    context_length = 64

    def __call__(self, tokens: torch.Tensor, styles: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((*tokens.shape, VOCAB))
        logits[..., SPECIAL.pad] = -math.inf
        logits[..., SPECIAL.bos] = -math.inf
        logits[..., SPECIAL.eos] = -math.inf
        return logits


def request(
    bars: int = 8,
    seed: int = 0,
    cfg_scale: float | None = None,
    plan: tuple[BarPlan, ...] | None = None,
    temperature: float = 1.0,
) -> GenerationRequest:
    return GenerationRequest(
        plan=plan or (BarPlan(Style.ROCK, PieceType.GROOVE),) * bars,
        seed=seed,
        cfg_scale=cfg_scale,
        temperature=temperature,
        top_k=None,
        top_p=None,
    )


def sampler(model: ConditionalModel) -> TokenSampler:
    return TokenSampler(model, SPECIAL, torch.device("cpu"))


def test_temperature_divides_the_logits() -> None:
    logits = torch.tensor([[2.0, 4.0, -1.0]])

    assert torch.equal(LogitFilter(temperature=2.0, top_k=None, top_p=None).apply(logits), logits / 2.0)


def test_top_k_keeps_only_the_k_largest_logits() -> None:
    logits = torch.tensor([[0.1, 3.0, 2.0, -1.0]])

    filtered = LogitFilter(temperature=1.0, top_k=2, top_p=None).apply(logits)

    assert torch.isfinite(filtered).tolist() == [[False, True, True, False]]


def test_top_p_keeps_the_smallest_set_reaching_p() -> None:
    logits = torch.log(torch.tensor([[0.15, 0.5, 0.05, 0.3]]))

    filtered = LogitFilter(temperature=1.0, top_k=None, top_p=0.7).apply(logits)

    assert torch.isfinite(filtered).tolist() == [[False, True, False, True]]


def test_top_p_always_keeps_the_most_likely_token() -> None:
    logits = torch.log(torch.tensor([[0.9, 0.05, 0.05]]))

    filtered = LogitFilter(temperature=1.0, top_k=None, top_p=0.1).apply(logits)

    assert torch.isfinite(filtered).tolist() == [[True, False, False]]


def test_draw_never_picks_a_token_with_zero_probability() -> None:
    probabilities = torch.tensor([[0.6, 0.0, 0.4, 0.0]]).repeat(1000, 1)
    uniforms = torch.linspace(0.0, 0.9999, 1000)

    drawn = CategoricalDraw.draw(probabilities, uniforms)

    assert set(drawn.tolist()) == {0, 2}
    assert (drawn == 0).sum().item() == 600


def test_guidance_moves_away_from_the_unconditional_logits() -> None:
    conditional = torch.tensor([2.0, 0.0])
    unconditional = torch.tensor([1.0, 1.0])

    combined = ClassifierFreeGuidance.combine(conditional, unconditional, scale=3.0)

    assert combined.tolist() == [4.0, -2.0]


def test_stops_at_the_last_bar_line_and_drops_it() -> None:
    model = ScriptedModel(64, lambda tokens: SPECIAL.bar)

    ids = sampler(model).sample([request(bars=8)])[0]

    assert ids == [SPECIAL.bos] + [SPECIAL.bar] * 7


def test_eos_ends_the_sequence_and_is_kept() -> None:
    model = ScriptedModel(64, lambda tokens: SPECIAL.eos if len(tokens) == 4 else NOTE)

    ids = sampler(model).sample([request()])[0]

    assert ids == [SPECIAL.bos, NOTE, NOTE, NOTE, SPECIAL.eos]


def test_sequence_without_bar_lines_is_cut_and_marked_with_eos() -> None:
    model = ScriptedModel(10_000, lambda tokens: NOTE)

    ids = sampler(model).sample([request(bars=2)])[0]

    assert ids[-1] == SPECIAL.eos
    assert len(ids) == 2 * TokenSampler.MAX_TOKENS_PER_BAR + 1


def test_each_bar_is_conditioned_on_its_planned_style() -> None:
    plan = (BarPlan(Style.ROCK, PieceType.GROOVE),) * 2 + (BarPlan(Style.FUNK, PieceType.GROOVE),) * 2
    model = ScriptedModel(64, lambda tokens: SPECIAL.bar if tokens[-1] == NOTE else NOTE)

    ids = sampler(model).sample([request(plan=plan)])[0]

    tokens, styles = model.calls[-1]
    assert ids == [SPECIAL.bos, NOTE, SPECIAL.bar, NOTE, SPECIAL.bar, NOTE, SPECIAL.bar, NOTE]
    assert styles[0].tolist() == [Style.ROCK.index] * 4 + [Style.FUNK.index] * 4
    assert tokens[0].tolist() == ids


def test_context_slides_to_keep_the_last_tokens() -> None:
    model = ScriptedModel(8, lambda tokens: SPECIAL.bar if len(model.calls) % 10 == 0 else NOTE)

    ids = sampler(model).sample([request(bars=3)])[0]

    assert max(tokens.shape[1] for tokens, _ in model.calls) == 8
    last_tokens, _ = model.calls[-1]
    assert last_tokens[0].tolist() == ids[-8:]


def test_same_seed_gives_the_same_sequence_alone_or_in_a_batch() -> None:
    model = UniformModel()

    alone = sampler(model).sample([request(seed=5)])[0]
    batch = sampler(model).sample([request(seed=4), request(seed=5), request(seed=6)])

    assert batch[1] == alone
    assert batch[0] != alone


def test_guidance_runs_an_unconditional_pass_with_the_null_style() -> None:
    model = ScriptedModel(64, lambda tokens: SPECIAL.bar)

    sampler(model).sample([request(bars=2, cfg_scale=3.0)])

    _, styles = model.calls[0]
    assert styles[:, 0].tolist() == [Style.ROCK.index, Style.NULL.index]


def test_guidance_scale_one_equals_sampling_without_guidance() -> None:
    torch.manual_seed(0)
    settings = ModelSettings(d_model=32, layers=2, heads=2, feedforward=64, dropout=0.0)
    model = DrumTransformer(settings, vocab_size=VOCAB, context_length=64, pad_id=SPECIAL.pad).eval()

    plain = sampler(model).sample([request(bars=2, seed=3)])
    guided = sampler(model).sample([request(bars=2, seed=3, cfg_scale=1.0)])

    assert guided == plain


def test_batch_with_different_sampling_settings_is_refused() -> None:
    with pytest.raises(ValueError, match="temperatura"):
        sampler(UniformModel()).sample([request(), request(temperature=0.5)])


class Uncached:
    def __init__(self, model: DrumTransformer) -> None:
        self._model = model
        self.context_length = model.context_length

    def __call__(self, tokens: torch.Tensor, styles: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        return self._model(tokens, styles, types)


@pytest.mark.parametrize("cfg_scale", [None, 2.0])
def test_cached_sampling_matches_full_recomputation(cfg_scale: float | None) -> None:
    torch.manual_seed(0)
    settings = ModelSettings(d_model=32, layers=2, heads=2, feedforward=64, dropout=0.0)
    model = DrumTransformer(settings, vocab_size=VOCAB, context_length=64, pad_id=SPECIAL.pad).eval()
    requests = [request(bars=2, seed=seed, cfg_scale=cfg_scale) for seed in range(4)]

    cached = sampler(model).sample(requests)
    full = sampler(Uncached(model)).sample(requests)

    assert cached == full


def test_sampling_past_the_context_falls_back_to_the_sliding_window() -> None:
    torch.manual_seed(0)
    settings = ModelSettings(d_model=32, layers=2, heads=2, feedforward=64, dropout=0.0)
    model = DrumTransformer(settings, vocab_size=VOCAB, context_length=6, pad_id=SPECIAL.pad).eval()
    requests = [request(bars=3, seed=seed) for seed in range(3)]

    assert sampler(model).sample(requests) == sampler(Uncached(model)).sample(requests)


class RecordingUniformModel(UniformModel):
    def __init__(self) -> None:
        self.rows: list[int] = []

    def __call__(self, tokens: torch.Tensor, styles: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        self.rows.append(tokens.shape[0])
        return super().__call__(tokens, styles, types)


def test_finished_sequences_leave_the_batch() -> None:
    model = RecordingUniformModel()

    sampler(model).sample([request(bars=4, seed=seed) for seed in range(3)])

    assert model.rows[0] == 3
    assert model.rows[-1] == 1
    assert model.rows == sorted(model.rows, reverse=True)
