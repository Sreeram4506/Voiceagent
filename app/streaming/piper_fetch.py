"""Downloads the Piper voice. The server runs this on start when the voice is missing (piper_tts.warmup);
running it in the build instead makes the voice part of the build:

Render build command:  pip install -r requirements.txt && python -m app.streaming.piper_fetch

  PIPER_MODEL_URL set → that .onnx (a custom-trained voice) and the .onnx.json beside it
                        (PIPER_MODEL_TOKEN: a Hugging Face read token, for a private repo)
  otherwise           → PIPER_VOICE from the public rhasspy/piper-voices collection
Files land in PIPER_VOICES_DIR (default models/piper_voices). Existing files are kept.
"""

import os
import sys
from pathlib import Path

import httpx

PUBLIC_VOICES = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
DOWNLOAD_TIMEOUT_SECONDS = 300


def voice_urls(voice: str, model_url: str = "") -> list[str]:
    """The .onnx and .onnx.json URLs for a custom model URL or a public voice name like te_IN-venkatesh-medium."""
    if model_url:
        return [model_url, f"{model_url}.json"]
    locale, name, quality = voice.split("-")
    base = f"{PUBLIC_VOICES}/{locale.split('_')[0]}/{locale}/{name}/{quality}/{voice}.onnx"
    return [base, f"{base}.json"]


def fetch(urls: list[str], target_dir: Path, token: str = "") -> list[Path]:
    """Download each URL into target_dir; token is a Hugging Face read token for a private voice repo."""
    target_dir.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    saved = []
    for url in urls:
        path = target_dir / url.rsplit("/", 1)[-1].split("?")[0]
        if not path.exists():
            print(f"[Piper] downloading {path.name}")
            with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                              timeout=DOWNLOAD_TIMEOUT_SECONDS) as r:
                r.raise_for_status()
                partial = path.with_suffix(path.suffix + ".part")
                with partial.open("wb") as out:
                    for chunk in r.iter_bytes():
                        out.write(chunk)
                partial.rename(path)
        saved.append(path)
    return saved


def main() -> int:
    from app.streaming.piper_tts import PIPER_VOICE, PIPER_VOICES_DIR

    try:
        paths = fetch(voice_urls(PIPER_VOICE, os.environ.get("PIPER_MODEL_URL", "")), Path(PIPER_VOICES_DIR),
                      token=os.environ.get("PIPER_MODEL_TOKEN", ""))
    except (httpx.HTTPError, ValueError) as e:
        print(f"[Piper] voice download failed: {e}")
        return 1
    print(f"[Piper] voice ready: {paths[0]} ({paths[0].stat().st_size // 1_000_000} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
