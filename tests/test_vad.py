from app.streaming.audio import FRAME_BYTES, split_frames, silence
from app.streaming.vad import EnergyVAD

LOUD = (3000).to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2)
QUIET = bytes(FRAME_BYTES)


def run(vad, frames):
    return [vad.feed(f) for f in frames]


def test_silence_produces_no_events():
    assert set(run(EnergyVAD(), [QUIET] * 50)) == {None}


def test_short_click_is_ignored():
    vad = EnergyVAD(min_speech_ms=200)
    events = run(vad, [LOUD] * 5 + [QUIET] * 40)  # 100 ms of noise
    assert "speech_start" not in events


def test_speech_then_silence_gives_start_and_end():
    vad = EnergyVAD(min_speech_ms=200, end_silence_ms=700)
    events = run(vad, [LOUD] * 20 + [QUIET] * 40)
    assert events.count("speech_start") == 1
    assert events.count("speech_end") == 1
    assert events.index("speech_start") < events.index("speech_end")


def test_brief_pause_does_not_end_utterance():
    vad = EnergyVAD(end_silence_ms=700)
    events = run(vad, [LOUD] * 20 + [QUIET] * 10 + [LOUD] * 20)  # 200 ms pause
    assert "speech_end" not in events


def test_split_frames_pads_last_frame():
    frames = split_frames(silence(50))  # 800 bytes -> 3 frames, last padded
    assert len(frames) == 3
    assert all(len(f) == FRAME_BYTES for f in frames)
