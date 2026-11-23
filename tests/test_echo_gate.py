"""The agent must not hear its own voice leaking back down the phone line as the caller cutting in."""

import numpy as np

from app.streaming.echo_gate import FRAME_BYTES, EchoGate

AGENT_LEVEL = 3000.0


def tone(level: float, frames: int = 1) -> bytes:
    """Speech-like audio whose 20 ms frames have roughly this RMS."""
    t = np.arange(frames * FRAME_BYTES // 2)
    return (np.sin(t * 0.3) * level * np.sqrt(2)).astype("<i2").tobytes()


def gate_hearing_agent() -> EchoGate:
    gate = EchoGate()
    gate.agent_audio(tone(AGENT_LEVEL, frames=50))
    return gate


def test_caller_audio_passes_untouched_when_the_agent_is_quiet():
    gate = gate_hearing_agent()
    frame = tone(300)
    assert gate.filter(frame, agent_playing=False, now=0.0) == frame


def test_faint_echo_of_the_agent_becomes_silence_while_it_plays():
    gate = gate_hearing_agent()
    echo = tone(AGENT_LEVEL * 0.15)
    assert gate.filter(echo, agent_playing=True, now=0.0) == bytes(len(echo))
    assert gate.blocked_frames == 1


def test_caller_talking_over_the_agent_gets_through_and_stays_open():
    gate = gate_hearing_agent()
    loud = tone(AGENT_LEVEL * 0.8)
    gate.filter(loud, agent_playing=True, now=0.00)
    assert gate.filter(loud, agent_playing=True, now=0.02) == loud
    soft = tone(AGENT_LEVEL * 0.1)  # quieter syllables of the same interruption
    assert gate.filter(soft, agent_playing=True, now=0.5) == soft


def test_gate_closes_again_once_the_caller_has_stopped():
    gate = gate_hearing_agent()
    loud = tone(AGENT_LEVEL * 0.8)
    gate.filter(loud, agent_playing=True, now=0.00)
    gate.filter(loud, agent_playing=True, now=0.02)
    echo = tone(AGENT_LEVEL * 0.15)
    assert gate.filter(echo, agent_playing=True, now=5.0) == bytes(len(echo))


def test_a_single_click_does_not_count_as_the_caller_cutting_in():
    gate = gate_hearing_agent()
    gate.filter(tone(AGENT_LEVEL), agent_playing=True, now=0.00)
    echo = tone(AGENT_LEVEL * 0.15)
    assert gate.filter(echo, agent_playing=True, now=0.02) == bytes(len(echo))


def test_gate_follows_this_calls_agent_voice_level():
    quiet_agent = EchoGate()
    quiet_agent.agent_audio(tone(1000, frames=200))
    frame = tone(700)
    quiet_agent.filter(frame, agent_playing=True, now=0.00)
    assert quiet_agent.filter(frame, agent_playing=True, now=0.02) == frame  # loud for this agent

    loud_agent = EchoGate()
    loud_agent.agent_audio(tone(6000, frames=200))
    loud_agent.filter(frame, agent_playing=True, now=0.00)
    assert loud_agent.filter(frame, agent_playing=True, now=0.02) == bytes(len(frame))  # echo for this one


def test_pauses_in_the_agent_audio_do_not_lower_its_level():
    gate = EchoGate()
    gate.agent_audio(tone(AGENT_LEVEL, frames=50) + bytes(FRAME_BYTES * 200))
    echo = tone(AGENT_LEVEL * 0.15)
    assert gate.filter(echo, agent_playing=True, now=0.0) == bytes(len(echo))
