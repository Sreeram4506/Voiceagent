"""Coqui XTTS v2 — zero-shot voice cloning TTS for phone calls.

Usage: place a 10–30 second WAV of the target voice at voice_profiles/reference.wav
       (or any path set in XTTS_REFERENCE_WAV env var). The first call loads the
       model (~1.8 GB, ~10s) and caches speaker conditioning latents; every call
       after that takes ~1–3s on Apple Silicon MPS.

Languages: Hindi (hi) is natively supported. Telugu (te) uses the Hindi phonology
           engine — most sounds map correctly, some retroflex consonants differ.

Output: 8 kHz 16-bit mono PCM yielded in 200 ms chunks.
"""

import asyncio
import os
import threading
import time
from contextlib import suppress
from pathlib import Path

# Must be set before importing TTS to prevent TF/NumPy 2 incompatibility
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("COQUI_TOS_AGREED", "1")

import numpy as np

TELEPHONY_SAMPLE_RATE = 8000
_XTTS_SAMPLE_RATE = 24000  # XTTS v2 native output rate
_PCM_CHUNK = 3200           # 200 ms at 8 kHz 16-bit mono

_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"

# XTTS only natively supports these; Telugu/Kannada best-effort via Hindi
_LANG_MAP = {
    "hi": "hi",
    "en": "en",
    "te": "hi",   # Hindi engine, Telugu Unicode text — most phonemes map
    "kn": "hi",
    "ta": "hi",
    "ml": "hi",
}

DEFAULT_REFERENCE_WAV = os.getenv(
    "XTTS_REFERENCE_WAV",
    str(Path(__file__).parents[2] / "voice_profiles" / "telugu_reference.wav"),
)

_model = None
_model_lock = threading.Lock()

# Cache: reference_wav_path → (gpt_cond_latent, speaker_embedding)
_latent_cache: dict[str, tuple] = {}
_latent_lock = threading.Lock()


def _load_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        import torch
        from TTS.api import TTS
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        t0 = time.time()
        print(f"[XTTS] Loading model on {device} (first time ~10s)…")
        tts = TTS(_MODEL_ID).to(device)
        print(f"[XTTS] Model ready in {time.time()-t0:.1f}s on {device}")
        _model = tts
        return _model


def _get_latents(reference_wav: str):
    """Cache conditioning latents so we don't recompute on every call."""
    with _latent_lock:
        if reference_wav in _latent_cache:
            return _latent_cache[reference_wav]
    tts = _load_model()
    t0 = time.time()
    print(f"[XTTS] Computing speaker latents for {reference_wav}…")
    gpt_cond_latent, speaker_embedding = tts.synthesizer.tts_model.get_conditioning_latents(
        audio_path=[reference_wav]
    )
    print(f"[XTTS] Latents ready in {time.time()-t0:.1f}s")
    with _latent_lock:
        _latent_cache[reference_wav] = (gpt_cond_latent, speaker_embedding)
    return gpt_cond_latent, speaker_embedding


def _chunk_to_pcm(chunk) -> bytes:
    """Convert a float32 numpy array at 24 kHz to int16 bytes at 8 kHz."""
    import torch, torchaudio
    if hasattr(chunk, "numpy"):
        arr = chunk.cpu().numpy()
    else:
        arr = np.array(chunk, dtype=np.float32)
    tensor = torch.from_numpy(arr).unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, _XTTS_SAMPLE_RATE, TELEPHONY_SAMPLE_RATE)
    pcm = (resampled.squeeze(0).numpy() * 32767).clip(-32768, 32767).astype(np.int16)
    return pcm.tobytes()


class XTTS:
    """Zero-shot voice cloning TTS using Coqui XTTS v2.

    Set reference_wav to a 10–30 second WAV of the voice you want to clone.
    The model is lazy-loaded on first use and kept in memory.
    """

    def __init__(self, reference_wav: str | None = None):
        self._reference_wav = reference_wav or DEFAULT_REFERENCE_WAV

    async def stream_synthesize(self, text: str, language: str, voice_tier: str):
        """Stream 8 kHz PCM via XTTS v2 streaming inference.

        Yields PCM chunks as they come out of the model.
        First chunk typically arrives in 1–3 s on Apple Silicon MPS.
        """
        ref = self._reference_wav
        if not Path(ref).exists():
            raise FileNotFoundError(
                f"[XTTS] Reference WAV not found: {ref}\n"
                "Record 10–30 seconds of the target voice and save it there."
            )

        xtts_lang = _LANG_MAP.get(language, "hi")
        t0 = time.time()
        first = True

        # Blocking work runs in a thread pool so the event loop stays free.
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _run_inference():
            try:
                tts = _load_model()
                gpt_cond_latent, speaker_embedding = _get_latents(ref)
                for chunk in tts.synthesizer.tts_model.inference_stream(
                    text,
                    xtts_lang,
                    gpt_cond_latent,
                    speaker_embedding,
                ):
                    pcm = _chunk_to_pcm(chunk)
                    asyncio.run_coroutine_threadsafe(queue.put(pcm), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)  # sentinel

        thread = threading.Thread(target=_run_inference, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            if first:
                print(f"[XTTS] first PCM at {time.time()-t0:.2f}s lang={xtts_lang} ref={Path(ref).name}")
                first = False
            # Yield in _PCM_CHUNK-sized frames for smooth playback
            buf = item
            while len(buf) >= _PCM_CHUNK:
                yield buf[:_PCM_CHUNK]
                buf = buf[_PCM_CHUNK:]
            if buf:
                yield buf

        print(f"[XTTS] done in {time.time()-t0:.2f}s")

    async def synthesize(self, text: str, language: str, voice_tier: str) -> bytes:
        chunks: list[bytes] = []
        async for pcm in self.stream_synthesize(text, language, voice_tier):
            chunks.append(pcm)
        return b"".join(chunks)


def warmup(reference_wav: str | None = None):
    """Call at server startup to pre-load model + compute latents.

    Runs in a background thread so it doesn't block the server start.
    """
    ref = reference_wav or DEFAULT_REFERENCE_WAV
    if not Path(ref).exists():
        print(f"[XTTS] Warmup skipped — reference WAV not found at {ref}")
        print(f"[XTTS] To use voice cloning: record 10–30s WAV, save as {ref}")
        return

    def _do_warmup():
        try:
            _get_latents(ref)
            print("[XTTS] Warmup complete — voice cloning ready")
        except Exception as e:
            print(f"[XTTS] Warmup error: {e}")

    threading.Thread(target=_do_warmup, daemon=True).start()
