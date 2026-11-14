"""Microsoft Edge TTS adapter — fast, free, native Telugu & Hindi Neural voices.

Latency: ~400ms first audio via ffmpeg pipe streaming.
Voices: te-IN-ShrutiNeural (Telugu female), te-IN-MohanNeural (Telugu male),
        hi-IN-SwaraNeural (Hindi female), en-IN-NeerjaNeural (English Indian female).
Output: 8kHz 16-bit mono PCM via ffmpeg pipe (MP3 → PCM as it arrives).
"""

import asyncio
import re
from contextlib import suppress

# Voice map: language → {gender → voice_id}
_VOICE_MAP = {
    "te": {"female": "te-IN-ShrutiNeural", "male": "te-IN-MohanNeural"},
    "hi": {"female": "hi-IN-SwaraNeural", "male": "hi-IN-MadhurNeural"},
    "en": {"female": "en-IN-NeerjaNeural", "male": "en-IN-PrabhatNeural"},
}

_DEFAULT_RATE = "+0%"
_DEFAULT_VOLUME = "+0%"


def _add_natural_breaks(text: str) -> str:
    """Inject SSML <break> tags so the voice pauses naturally — like a real person.

    edge_tts wraps text in <prosody> SSML internally; <break/> is valid inside it
    and passes through to the Microsoft Neural TTS engine.

    Pause lengths chosen to match natural Indian conversational speech rhythm:
    - Comma after opener (అర్థమైంది,) → 180ms soft pause
    - Mid-sentence comma → 150ms
    - Devanagari danda (।) → 380ms sentence-end pause (Hindi)
    - Ellipsis (...) → 650ms thinking/reflecting pause (most human-sounding)
    - Em-dash (—) → 280ms dramatic pause
    """
    # Thinking/reflecting pause: "..." or unicode ellipsis
    text = re.sub(r'\.\.\.', '<break time="650ms"/>', text)
    text = re.sub(r'…', '<break time="650ms"/>', text)

    # Devanagari danda sentence end (Hindi)
    text = re.sub(r'।\s*', '।<break time="380ms"/> ', text)

    # Comma after short opener (1–4 words before comma) → longer breathing pause
    # e.g. "అర్థమైంది అండి, ..." → 200ms
    text = re.sub(r'^([\w\s,అ-ఽఀ-౿A-Za-zऀ-ॿ]{1,30}),\s', r'\1, <break time="200ms"/> ', text)

    # Remaining commas → soft pause
    text = re.sub(r',(?!\s*<break)', ', <break time="150ms"/> ', text)

    # Em-dash pause
    text = re.sub(r'\s*—\s*', ' <break time="280ms"/> ', text)

    return text

TELEPHONY_SAMPLE_RATE = 8000
_PCM_CHUNK = 3200  # 200ms at 8kHz 16-bit mono


class EdgeTTS:
    """Async TTS using Microsoft Edge Neural voices. 8kHz 16-bit mono PCM output.

    Uses ffmpeg as a streaming MP3→PCM pipe so the first audio chunk arrives
    ~400ms after synthesis starts instead of waiting for the full MP3 (~1.4s).
    """

    def __init__(self, voice: str | None = None, rate: str = _DEFAULT_RATE,
                 volume: str = _DEFAULT_VOLUME, gender: str = "female"):
        self._voice = voice
        self._rate = rate
        self._volume = volume
        self._gender = gender

    def _pick_voice(self, language: str) -> str:
        if self._voice:
            return self._voice
        voices = _VOICE_MAP.get(language, _VOICE_MAP["te"])
        return voices.get(self._gender, voices["female"])

    async def stream_synthesize(self, text: str, language: str, voice_tier: str):
        """Stream 8kHz PCM via ffmpeg pipe — first audio in ~400ms instead of ~1.4s."""
        import edge_tts
        import time

        voice = self._pick_voice(language)
        t0 = time.time()
        # NOTE: edge_tts escapes all text via xml.sax.saxutils.escape() internally,
        # so raw SSML tags cannot be injected. Instead we rely on the Neural TTS
        # prosody model: "..." creates a natural thinking pause (~500-700ms),
        # commas create ~150ms pauses, and "?" creates a natural question upturn.
        # The LLM prompt instructs the model to use these punctuation cues naturally.
        communicate = edge_tts.Communicate(text, voice, rate=self._rate, volume=self._volume)

        # Open an ffmpeg subprocess: MP3 on stdin → 8kHz 16-bit PCM on stdout.
        # -f mp3: tell ffmpeg input format upfront → skips format probing (~300ms saved)
        # -probesize 32 -analyzeduration 0: minimal startup analysis
        # -fflags +nobuffer: flush PCM to stdout as soon as it's decoded
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-probesize", "32", "-analyzeduration", "0",
            "-f", "mp3", "-i", "pipe:0",
            "-ar", str(TELEPHONY_SAMPLE_RATE), "-ac", "1",
            "-fflags", "+nobuffer",
            "-f", "s16le", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )

        async def _feed_mp3() -> None:
            """Push edge-tts MP3 chunks into ffmpeg stdin."""
            try:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        if proc.stdin and not proc.stdin.is_closing():
                            proc.stdin.write(chunk["data"])
            except Exception as exc:
                print(f"[EdgeTTS] feed error: {exc}")
            finally:
                with suppress(Exception):
                    proc.stdin.close()

        feed_task = asyncio.create_task(_feed_mp3())
        first = True

        try:
            while True:
                pcm = await proc.stdout.read(_PCM_CHUNK)
                if not pcm:
                    break
                if first:
                    print(f"[EdgeTTS] first PCM at {time.time()-t0:.2f}s voice={voice}")
                    first = False
                yield pcm
        finally:
            with suppress(Exception):
                proc.kill()
            await feed_task
            with suppress(Exception):
                await proc.wait()

        print(f"[EdgeTTS] done in {time.time()-t0:.2f}s text_len={len(text)} voice={voice}")

    async def synthesize(self, text: str, language: str, voice_tier: str) -> bytes:
        """Collect full audio for pre-caching (filler phrases etc.)."""
        chunks: list[bytes] = []
        async for pcm in self.stream_synthesize(text, language, voice_tier):
            chunks.append(pcm)
        return b"".join(chunks)
