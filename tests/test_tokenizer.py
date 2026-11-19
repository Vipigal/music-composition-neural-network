from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from symusic.core import NoteTick, ScoreTick, TrackTick

from drumgen.config import TokenizerSettings
from drumgen.tokenizer import DrumTokenizer

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE_BPMS = {"jazz_120_beat.mid": 120, "funk_117_beat.mid": 117, "rock_115_fill.mid": 115}
TICKS_PER_QUARTER = 480
SIXTEENTH = TICKS_PER_QUARTER // 4
BAR = TICKS_PER_QUARTER * 4


@pytest.fixture(scope="module")
def tokenizer() -> DrumTokenizer:
    return DrumTokenizer.from_settings(TokenizerSettings())


def drum_score(notes: list[tuple[int, int]]) -> ScoreTick:
    score = ScoreTick(TICKS_PER_QUARTER)
    track = TrackTick("Drums", 0, True)
    for time, pitch in notes:
        track.notes.append(NoteTick(time, SIXTEENTH, pitch, 80))
    score.tracks.append(track)
    return score


def onsets_by_pitch(score: ScoreTick) -> dict[int, list[int]]:
    onsets: dict[int, list[int]] = defaultdict(list)
    for note in score.tracks[0].notes:
        onsets[note.pitch].append(note.time)
    return {pitch: sorted(times) for pitch, times in onsets.items()}


def test_gmd_pitches_collapse_into_the_nine_groovae_classes(tokenizer: DrumTokenizer) -> None:
    score = drum_score([(0, 40), (0, 22), (SIXTEENTH, 59), (SIXTEENTH, 37), (2 * SIXTEENTH, 58)])

    mapped = tokenizer.map_drums(score)

    assert [note.pitch for note in mapped.tracks[0].notes] == [38, 42, 38, 51, 43]


def test_pitches_outside_the_gmd_kit_are_dropped(tokenizer: DrumTokenizer) -> None:
    score = drum_score([(0, 36), (0, 81)])

    mapped = tokenizer.map_drums(score)

    assert [note.pitch for note in mapped.tracks[0].notes] == [36]


@pytest.mark.parametrize("fixture", sorted(FIXTURE_BPMS))
def test_round_trip_keeps_every_note(tokenizer: DrumTokenizer, fixture: str) -> None:
    original = tokenizer.map_drums(ScoreTick(FIXTURES / fixture))

    decoded = tokenizer.decode(tokenizer.encode(original))

    assert {pitch: len(times) for pitch, times in onsets_by_pitch(decoded).items()} == {
        pitch: len(times) for pitch, times in onsets_by_pitch(original).items()
    }


def round_trip_errors_in_ticks(tokenizer: DrumTokenizer, fixture: str) -> list[int]:
    original = tokenizer.map_drums(ScoreTick(FIXTURES / fixture))
    decoded = tokenizer.decode(tokenizer.encode(original))
    expected = onsets_by_pitch(original)
    return [
        abs(got - want)
        for pitch, times in onsets_by_pitch(decoded).items()
        for got, want in zip(times, expected[pitch], strict=True)
    ]


def test_round_trip_median_onset_error_is_at_most_3_ms(tokenizer: DrumTokenizer) -> None:
    errors_ms = [
        error * 60_000 / bpm / TICKS_PER_QUARTER
        for fixture, bpm in FIXTURE_BPMS.items()
        for error in round_trip_errors_in_ticks(tokenizer, fixture)
    ]

    assert np.median(errors_ms) <= 3.0


@pytest.mark.parametrize("fixture", sorted(FIXTURE_BPMS))
def test_round_trip_moves_no_note_more_than_half_a_microtiming_bin(
    tokenizer: DrumTokenizer, fixture: str
) -> None:
    half_bin = 4

    assert max(round_trip_errors_in_ticks(tokenizer, fixture)) <= half_bin


def test_decoded_track_is_a_drum_track(tokenizer: DrumTokenizer) -> None:
    decoded = tokenizer.decode(tokenizer.encode(drum_score([(0, 36), (SIXTEENTH, 42)])))

    assert len(decoded.tracks) == 1
    assert decoded.tracks[0].is_drum is True


def test_decode_ignores_special_tokens(tokenizer: DrumTokenizer) -> None:
    ids = tokenizer.encode(drum_score([(0, 36), (SIXTEENTH, 42)]))

    decoded = tokenizer.decode([tokenizer.bos_id, *ids, tokenizer.eos_id, tokenizer.pad_id])

    assert [note.pitch for note in decoded.tracks[0].notes] == [36, 42]


def test_quantize_only_rewrites_microtiming_tokens(tokenizer: DrumTokenizer) -> None:
    ids = tokenizer.encode(ScoreTick(FIXTURES / "jazz_120_beat.mid"))

    quantized = tokenizer.quantize(ids)

    changed = [
        (tokenizer.token_of(before), tokenizer.token_of(after))
        for before, after in zip(ids, quantized, strict=True)
        if before != after
    ]
    assert changed
    assert all(before.startswith("MicroTiming_") and after == "MicroTiming_0" for before, after in changed)


def test_quantized_onsets_fall_on_the_sixteenth_grid(tokenizer: DrumTokenizer) -> None:
    ids = tokenizer.encode(ScoreTick(FIXTURES / "jazz_120_beat.mid"))

    decoded = tokenizer.decode(tokenizer.quantize(ids))

    assert all(note.time % SIXTEENTH == 0 for note in decoded.tracks[0].notes)


def test_count_bars_counts_the_first_bar_without_bar_token(tokenizer: DrumTokenizer) -> None:
    ids = tokenizer.encode(drum_score([(0, 36), (BAR, 38), (2 * BAR + SIXTEENTH, 42)]))

    assert tokenizer.count_bars(ids) == 3


def test_count_bars_of_a_one_bar_fill(tokenizer: DrumTokenizer) -> None:
    ids = tokenizer.encode(ScoreTick(FIXTURES / "rock_115_fill.mid"))

    assert tokenizer.count_bars(ids) == 1


def test_disabling_microtiming_removes_microtiming_tokens() -> None:
    tokenizer = DrumTokenizer.from_settings(TokenizerSettings(use_microtiming=False))

    ids = tokenizer.encode(ScoreTick(FIXTURES / "funk_117_beat.mid"))

    assert not any(tokenizer.token_of(token_id).startswith("MicroTiming_") for token_id in ids)
