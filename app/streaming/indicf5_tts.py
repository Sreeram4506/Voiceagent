"""IndicF5 TTS — local Telugu TTS using AI4Bharat's IndicF5 model.

Outputs 8 kHz 16-bit mono PCM for the telephony stream.
Model: ai4bharat/IndicF5 (gated on HuggingFace, requires HF_TOKEN).
"""

import io
import os
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

_model = None
_model_lock = threading.Lock()
_model_error: str | None = None

INDICF5_SAMPLE_RATE = 24000
TELEPHONY_SAMPLE_RATE = 8000

REF_AUDIO_DIR = Path(__file__).resolve().parent.parent.parent / "voice_profiles"


def _ensure_ref_dir():
    REF_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


NFE_STEPS = 16  # fewer ODE steps = faster inference (32 default, 16 balances quality/speed)


class _IndicF5Wrapper:
    """Wraps the F5-TTS model + vocoder into a simple callable."""

    def __init__(self, ema_model, vocoder, device):
        self.ema_model = ema_model
        self.vocoder = vocoder
        self.device = device
        self._ref_cache: dict[str, tuple] = {}

    def _get_ref(self, ref_audio_path, ref_text):
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text
        key = ref_audio_path
        if key not in self._ref_cache:
            self._ref_cache[key] = preprocess_ref_audio_text(
                ref_audio_path, ref_text, device=self.device
            )
        return self._ref_cache[key]

    def __call__(self, text, ref_audio_path, ref_text):
        import time
        from f5_tts.infer.utils_infer import infer_process

        ref_audio, ref_text_processed = self._get_ref(ref_audio_path, ref_text)
        t0 = time.time()
        audio, sr, _ = infer_process(
            ref_audio, ref_text_processed, text,
            self.ema_model, self.vocoder,
            mel_spec_type="vocos",
            nfe_step=NFE_STEPS,
            device=self.device,
        )
        elapsed = time.time() - t0
        dur = len(audio) / 24000
        print(f"[IndicF5] gen={elapsed:.2f}s audio={dur:.2f}s RTF={elapsed/dur:.2f}x nfe={NFE_STEPS}")
        return audio


def _load_model():
    global _model, _model_error
    with _model_lock:
        if _model is not None:
            return _model
        if _model_error:
            return None
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            from f5_tts.infer.utils_infer import load_vocoder, load_model
            from f5_tts.model import DiT

            hf_token = os.environ.get("HF_TOKEN", "")
            if not hf_token:
                _model_error = "HF_TOKEN not set. Go to Voice Tuning page to configure."
                print(f"[IndicF5] {_model_error}")
                return None

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"[IndicF5] Using device: {device}")

            print("[IndicF5] Downloading vocab from ai4bharat/IndicF5 ...")
            vocab_path = hf_hub_download("ai4bharat/IndicF5", "checkpoints/vocab.txt", token=hf_token)

            print("[IndicF5] Downloading model weights ...")
            weights_path = hf_hub_download("ai4bharat/IndicF5", "model.safetensors", token=hf_token)

            print("[IndicF5] Loading vocoder ...")
            vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

            print("[IndicF5] Building model architecture ...")
            ema_model = load_model(
                DiT,
                dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                mel_spec_type="vocos",
                vocab_file=vocab_path,
                device=device,
            )

            print("[IndicF5] Loading weights from safetensors ...")
            state_dict = load_file(weights_path, device=str(device))
            clean_dict = {
                k.replace("ema_model.", ""): v
                for k, v in state_dict.items()
                if k not in ["initted", "step"]
            }
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                clean_dict.pop(key, None)

            ema_model.load_state_dict(clean_dict, strict=False)
            ema_model = ema_model.eval().to(device)
            del state_dict, clean_dict

            _model = _IndicF5Wrapper(ema_model, vocoder, device)
            print("[IndicF5] Model loaded successfully!")

            profile = get_active_profile()
            if profile:
                _model._get_ref(profile["audio_path"], profile["transcript"])
                print("[IndicF5] Reference audio pre-cached")

            return _model
        except Exception as e:
            _model_error = str(e)
            print(f"[IndicF5] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return None


def reset_model():
    """Reset model state so it can be reloaded (e.g. after token change)."""
    global _model, _model_error
    with _model_lock:
        _model = None
        _model_error = None


def get_model_status() -> dict:
    """Return current model status for the UI."""
    if _model is not None:
        return {"status": "loaded", "message": "IndicF5 model loaded and ready"}
    if _model_error:
        return {"status": "error", "message": _model_error}
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        return {"status": "no_token", "message": "HF_TOKEN not configured"}
    return {"status": "not_loaded", "message": "Model not loaded yet"}


def get_voice_profiles() -> list[dict]:
    """List saved voice profiles (reference audio + transcript pairs)."""
    _ensure_ref_dir()
    profiles = []
    for wav in sorted(REF_AUDIO_DIR.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        transcript = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
        profiles.append({
            "id": wav.stem,
            "name": wav.stem.replace("_", " ").title(),
            "audio_file": wav.name,
            "transcript": transcript,
            "active": _is_active_profile(wav.stem),
        })
    return profiles


def _is_active_profile(profile_id: str) -> bool:
    active_file = REF_AUDIO_DIR / ".active"
    if active_file.exists():
        return active_file.read_text().strip() == profile_id
    return False


def set_active_profile(profile_id: str):
    _ensure_ref_dir()
    active_file = REF_AUDIO_DIR / ".active"
    active_file.write_text(profile_id)


def get_active_profile() -> dict | None:
    _ensure_ref_dir()
    active_file = REF_AUDIO_DIR / ".active"
    if not active_file.exists():
        return None
    pid = active_file.read_text().strip()
    wav = REF_AUDIO_DIR / f"{pid}.wav"
    txt = REF_AUDIO_DIR / f"{pid}.txt"
    if not wav.exists():
        return None
    return {
        "id": pid,
        "audio_path": str(wav),
        "transcript": txt.read_text(encoding="utf-8").strip() if txt.exists() else "",
    }


def save_voice_profile(profile_id: str, audio_bytes: bytes, transcript: str):
    _ensure_ref_dir()
    wav_path = REF_AUDIO_DIR / f"{profile_id}.wav"
    txt_path = REF_AUDIO_DIR / f"{profile_id}.txt"
    wav_path.write_bytes(audio_bytes)
    txt_path.write_text(transcript, encoding="utf-8")


def delete_voice_profile(profile_id: str):
    wav_path = REF_AUDIO_DIR / f"{profile_id}.wav"
    txt_path = REF_AUDIO_DIR / f"{profile_id}.txt"
    wav_path.unlink(missing_ok=True)
    txt_path.unlink(missing_ok=True)
    active_file = REF_AUDIO_DIR / ".active"
    if active_file.exists() and active_file.read_text().strip() == profile_id:
        active_file.unlink()


def _downsample_24k_to_8k(audio_24k: np.ndarray) -> bytes:
    """Downsample 24kHz float audio to 8kHz 16-bit mono PCM with loudness normalization."""
    from pydub import AudioSegment

    if audio_24k.dtype == np.int16:
        audio_24k = audio_24k.astype(np.float32) / 32768.0

    buf = io.BytesIO()
    sf.write(buf, audio_24k, INDICF5_SAMPLE_RATE, format="WAV")
    buf.seek(0)
    seg = AudioSegment.from_file(buf, format="wav")

    target_dBFS = -20.0
    if seg.dBFS < -60:
        return b""
    seg = seg.apply_gain(target_dBFS - seg.dBFS)

    seg = seg.set_frame_rate(TELEPHONY_SAMPLE_RATE).set_channels(1).set_sample_width(2)
    return seg.raw_data


def synthesize_to_file(text: str, output_path: str, profile: dict | None = None) -> bool:
    """Synthesize text to a WAV file. Returns True on success."""
    model = _load_model()
    if model is None:
        return False

    if profile is None:
        profile = get_active_profile()
    if profile is None:
        print("[IndicF5] No active voice profile configured")
        return False

    try:
        audio = model(text, ref_audio_path=profile["audio_path"], ref_text=profile["transcript"])
        if isinstance(audio, np.ndarray):
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            sf.write(output_path, audio, INDICF5_SAMPLE_RATE)
        return True
    except Exception as e:
        print(f"[IndicF5] Synthesis error: {e}")
        import traceback
        traceback.print_exc()
        return False


class IndicF5TTS:
    """Async TTS adapter for the streaming pipeline. Outputs 8kHz 16-bit mono PCM."""

    def __init__(self):
        pass

    async def synthesize(self, text: str, language: str, voice_tier: str) -> bytes:
        """Generate speech and return 8kHz 16-bit mono PCM bytes."""
        import asyncio

        model = _load_model()
        if model is None:
            raise RuntimeError(f"IndicF5 model not available: {_model_error}")

        profile = get_active_profile()
        if profile is None:
            raise RuntimeError("No active voice profile. Configure one in Voice Tuning.")

        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            None,
            lambda: model(text, ref_audio_path=profile["audio_path"], ref_text=profile["transcript"]),
        )

        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)

        return _downsample_24k_to_8k(audio)
