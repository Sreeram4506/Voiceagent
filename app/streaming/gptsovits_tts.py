"""GPT-SoVITS TTS adapter — calls local GPT-SoVITS HTTP API for Telugu voice synthesis.

Outputs 8 kHz 16-bit mono PCM for the telephony stream.
Requires GPT-SoVITS API server running on localhost:9880.

Start the API server with:
  cd /Users/koushikeslavath/Downloads/GPT-SoVITS
  PYTHONPATH=.:GPT_SoVITS python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
"""

import io
import os

import numpy as np

GPTSOVITS_API_URL = os.environ.get("GPTSOVITS_API_URL", "http://127.0.0.1:9880")

# Default reference audio from OpenSLR Telugu dataset (speaker tem_00682)
DEFAULT_REF_AUDIO = os.environ.get(
    "GPTSOVITS_REF_AUDIO",
    "/Users/koushikeslavath/Downloads/telugu_voice_data/speaker_00682/tem_00682_01850053901.wav",
)
DEFAULT_REF_TEXT = os.environ.get(
    "GPTSOVITS_REF_TEXT",
    "ఏమయ్య ఓ సంఘపోల్లు ఎక్కడ చూస్తే ఏమీ లేదు",
)

TELEPHONY_SAMPLE_RATE = 8000
GPTSOVITS_SAMPLE_RATE = 32000  # GPT-SoVITS v2 outputs 32kHz


def _wav_to_8k_pcm(wav_bytes: bytes) -> bytes:
    """Convert WAV bytes (any sample rate) to 8kHz 16-bit mono PCM.

    GPT-SoVITS output is often very quiet (-60 to -80 dBFS). Cap gain boost at
    +30 dB to avoid clipping distortion, then resample with scipy for quality.
    """
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    buf = io.BytesIO(wav_bytes)
    audio, sr = sf.read(buf, dtype="float32")

    if audio.size == 0:
        return b""

    # Ensure mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to 8kHz
    g = gcd(int(sr), TELEPHONY_SAMPLE_RATE)
    audio = resample_poly(audio, TELEPHONY_SAMPLE_RATE // g, int(sr) // g)

    # Normalize: target -20 dBFS but cap boost at +30 dB to prevent clipping
    peak = float(np.abs(audio).max())
    if peak < 1e-6:
        return b""  # truly silent

    target_peak = 10 ** (-20.0 / 20)              # -20 dBFS
    gain = min(target_peak / peak, 10 ** (30 / 20))  # max +30 dB
    audio = np.clip(audio * gain, -1.0, 1.0)

    return (audio * 32767).astype(np.int16).tobytes()


class GptSoVITSTTS:
    """Async TTS adapter using GPT-SoVITS HTTP API. Outputs 8kHz 16-bit mono PCM."""

    def __init__(
        self,
        api_url: str = GPTSOVITS_API_URL,
        ref_audio_path: str = DEFAULT_REF_AUDIO,
        ref_text: str = DEFAULT_REF_TEXT,
    ):
        self.api_url = api_url.rstrip("/")
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text

    async def synthesize(self, text: str, language: str, voice_tier: str) -> bytes:
        """Generate speech and return 8kHz 16-bit mono PCM bytes."""
        import asyncio
        import time

        loop = asyncio.get_event_loop()

        def _call_api():
            import httpx

            t0 = time.time()
            payload = {
                "text": text,
                "text_lang": "all_zh",   # GPT-SoVITS treats Telugu as CJK-style (no native support)
                "ref_audio_path": self.ref_audio_path,
                "prompt_text": self.ref_text,
                "prompt_lang": "all_zh",
                "text_split_method": "cut5",
                "batch_size": 1,
                "media_type": "wav",
                "streaming_mode": False,
            }
            resp = httpx.post(
                f"{self.api_url}/tts",
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            wav_bytes = resp.content
            elapsed = time.time() - t0
            print(f"[GPT-SoVITS] synthesis={elapsed:.2f}s text_len={len(text)}")
            return wav_bytes

        wav_bytes = await loop.run_in_executor(None, _call_api)
        return _wav_to_8k_pcm(wav_bytes)
