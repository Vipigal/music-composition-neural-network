from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from miditok import PerTok, TokenizerConfig, TokSequence
from symusic.core import NoteTick, ScoreTick, TrackTick

from drumgen.config import TokenizerSettings

TICKS_PER_QUARTER = 480

DRUM_CLASSES: dict[int, int] = {
    36: 36,
    38: 38,
    40: 38,
    37: 38,
    42: 42,
    22: 42,
    44: 42,
    46: 46,
    26: 46,
    43: 43,
    58: 43,
    47: 47,
    45: 47,
    50: 50,
    48: 50,
    49: 49,
    52: 49,
    55: 49,
    57: 49,
    51: 51,
    53: 51,
    59: 51,
}

SPECIAL_TOKENS = ["PAD", "BOS", "EOS"]


class DrumTokenizer:
    def __init__(self, pertok: PerTok) -> None:
        self._pertok = pertok
        vocab = pertok.vocab
        if not isinstance(vocab, dict):
            raise TypeError("o PerTok de bateria precisa de um vocabulário único")
        self._ids = vocab
        self._tokens = {token_id: token for token, token_id in vocab.items()}
        self._microtiming_ids = {
            token_id for token, token_id in vocab.items() if token.startswith("MicroTiming_")
        }
        self._special_ids = {self.pad_id, self.bos_id, self.eos_id}
        self._velocity_ids = {
            int(token.removeprefix("Velocity_")): token_id
            for token, token_id in vocab.items()
            if token.startswith("Velocity_")
        }
        self._velocity_of = {token_id: velocity for velocity, token_id in self._velocity_ids.items()}

    @classmethod
    def from_settings(cls, settings: TokenizerSettings) -> DrumTokenizer:
        config = TokenizerConfig(
            beat_res={(0, 4): 4},
            num_velocities=settings.num_velocities,
            special_tokens=SPECIAL_TOKENS,
            use_velocities=True,
            use_note_duration_programs=[],
            use_tempos=False,
            use_time_signatures=False,
            use_programs=False,
            drums_pitch_range=(min(DRUM_CLASSES.values()), max(DRUM_CLASSES.values())),
            use_microtiming=settings.use_microtiming,
            max_microtiming_shift=settings.max_microtiming_shift,
            num_microtiming_bins=settings.num_microtiming_bins,
            ticks_per_quarter=TICKS_PER_QUARTER,
            use_position_toks=True,
        )
        return cls(PerTok(config))

    @classmethod
    def from_file(cls, path: Path) -> DrumTokenizer:
        return cls(PerTok(params=path))

    def save(self, path: Path) -> None:
        self._pertok.save(path)

    @property
    def vocab_size(self) -> int:
        return len(self._ids)

    @property
    def pad_id(self) -> int:
        return self._ids["PAD_None"]

    @property
    def bos_id(self) -> int:
        return self._ids["BOS_None"]

    @property
    def eos_id(self) -> int:
        return self._ids["EOS_None"]

    @property
    def bar_id(self) -> int:
        return self._ids["Bar_None"]

    def token_of(self, token_id: int) -> str:
        return self._tokens[token_id]

    def id_of(self, token: str) -> int:
        return self._ids[token]

    def map_drums(self, score: ScoreTick) -> ScoreTick:
        mapped = ScoreTick(score.ticks_per_quarter)
        track = TrackTick("Drums", 0, True)
        for source in score.tracks:
            for note in source.notes:
                drum_class = DRUM_CLASSES.get(note.pitch)
                if drum_class is not None:
                    track.notes.append(NoteTick(note.time, note.duration, drum_class, note.velocity))
        track.notes.sort(key=lambda note: (note.time, note.pitch))
        mapped.tracks.append(track)
        return mapped

    def encode(self, score: ScoreTick) -> list[int]:
        sequences = self._pertok.encode(self.map_drums(score))
        if isinstance(sequences, list):
            if not sequences:
                return []
            sequences = sequences[0]
        return [token_id for token_id in sequences.ids if isinstance(token_id, int)]

    def decode(self, ids: Sequence[int]) -> ScoreTick:
        sequence = TokSequence(ids=[token_id for token_id in ids if token_id not in self._special_ids])
        score = self._pertok.decode([sequence], programs=[(0, True)])
        for track in score.tracks:
            for note in track.notes:
                if note.time < 0:
                    note.time = 0
        return score

    def quantize(self, ids: Sequence[int]) -> list[int]:
        on_grid = self._ids.get("MicroTiming_0")
        if on_grid is None:
            return list(ids)
        return [on_grid if token_id in self._microtiming_ids else token_id for token_id in ids]

    def count_bars(self, ids: Sequence[int]) -> int:
        content = [token_id for token_id in ids if token_id not in self._special_ids]
        if not content:
            return 0
        return content.count(self.bar_id) + 1

    def bar_starts(self, sequence: Sequence[int]) -> list[int]:
        return [
            index
            for index, token_id in enumerate(sequence)
            if token_id == self.bar_id or (index == 0 and token_id == self.bos_id)
        ]

    def scale_velocities(self, ids: Sequence[int], factor: float) -> list[int]:
        levels = sorted(self._velocity_ids)
        rescaled = {
            token_id: self._velocity_ids[min(levels, key=lambda level: abs(level - velocity * factor))]
            for token_id, velocity in self._velocity_of.items()
        }
        return [rescaled.get(token_id, token_id) for token_id in ids]
