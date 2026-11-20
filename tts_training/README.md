# Our own Telugu voice (Piper, CPU on Render)

Train a natural Telugu voice once on a GPU, then run it on the Render server's CPU
in place of Microsoft Edge TTS.

**Where this voice is heard.** Calls on OpenAI Realtime or Gemini Live (the default)
make their own voice, so they never use a TTS. Piper speaks only on calls that run the
classic pipeline (`VA_VOICE_PIPELINE=classic`, or Realtime unavailable), the same
calls Edge serves today.

## What already works

| Piece | File |
|---|---|
| Piper voice in the call pipeline, loaded once, cached phrases, sentence-by-sentence | `app/streaming/piper_tts.py` |
| Spoken-form Telugu: ₹ amounts, dates, phone numbers, final ం → "m", English in Telugu letters | `app/streaming/telugu_text.py` |
| Switch: `VA_TTS_ENGINE=piper` (falls back to Edge when no voice file is found) | `app/streaming/factory.py` |
| Build-time voice download, so Render never fetches it on start | `app/streaming/piper_fetch.py` |

Two public Telugu Piper voices work today with no training: `te_IN-padmavathi-medium`
(female) and `te_IN-venkatesh-medium` (male), both CC BY 4.0. Measured on the 20 lines
in `test_sentences.txt` (Mac CPU, transcribed back with gpt-4o-transcribe):

| Voice | Words heard right | Time per sentence |
|---|---|---|
| Piper padmavathi | 0.96 | ~120 ms |
| Piper venkatesh | 0.95 | ~100 ms |
| Microsoft Edge | 0.93 | ~1,230 ms (network) |

"Heard right" checks clarity, not how human the voice sounds. Listen before deciding.

## The plan, step by step

### Day 1-2: speaker, consent, recording

1. **Consent first.** Get the speaker's written consent that their voice may be used
   to train a synthetic voice for commercial phone calls, for how long, and how they
   can withdraw. Keep it with the recordings. No consent, no training.
2. One native speaker, quiet room, USB or condenser mic, about 20 cm away, no music
   or fan. Record mono WAV at 44.1 or 48 kHz (the script resamples). 30-45 minute
   sessions, water nearby, same mic and room every day.
3. Half reading prepared lines (dialogues, everyday sentences, the kind of things the
   agent says on calls), half free talk (daily life, food, films) with natural
   Telugu-English mixing. Speak the way you'd talk to a customer, not a newsreader.
4. Don't use phone-call recordings (8 kHz, noisy) and don't mix in other speakers.
5. Put the recordings in `tts_training/data/raw/`.

Check espeak-ng reads Telugu: done. It phonemizes Telugu script well; the one flaw
(a word-final ం read as "n") is fixed by the normalizer.

### Day 3-4: clips and transcripts (on your laptop, in this repo)

```bash
python tts_training/scripts/prepare_clips.py tts_training/data/raw tts_training/data/wavs
OPENAI_API_KEY=... python tts_training/scripts/transcribe_clips.py tts_training/data/wavs tts_training/data/transcripts.tsv
```

Then **listen to every clip and correct `transcripts.tsv`**. The voice learns exactly
what the text says. The convention, the same one live calls use:

* everything in Telugu script; English words written the way they were said:
  ఓకే, లోన్, మీటింగ్, సార్
* numbers as they were said, in words or digits (digits go through the same normalizer)
* a clip with a cough, a laugh over words, or a mistake: put `#` in front of its line

```bash
python tts_training/scripts/build_metadata.py tts_training/data/wavs tts_training/data/transcripts.tsv tts_training/data/metadata.csv
```

It writes Piper's `metadata.csv` (`clip.wav|text`), lists every clip it left out and
why, and reports the hours. Aim for 2-3 h to start, 6-7 h for the most natural voice.

### Day 5-7: training (GPU)

Upload `tts_training/data/` (`metadata.csv` + `wavs/`) and `tts_training/train.sh`
to Colab, Kaggle or a rented GPU (8 GB VRAM or more), then:

```bash
export DATA=/content/data VOICE=te_IN-suma-medium
bash train.sh setup
bash train.sh smoke    # minutes; if this fails, nothing longer will work
bash train.sh fit      # hours; run again to resume after a disconnect
```

Every few hours: `bash train.sh export`, then download the `.onnx`, `.onnx.json`
and `_sample.wav`, and on your laptop:

```bash
python tts_training/scripts/compare_voices.py --piper path/to/te_IN-suma-medium.onnx --edge
```

Listen in `tts_training/output/`: casual lines, English words, amounts, dates. Robotic
tone or a wrong akshara usually means a wrong transcript or too little of that sound;
fix the text or record targeted lines, rebuild `metadata.csv`, and `fit` again. Stop
when it stops getting better.

**Starting checkpoint.** `train.sh` starts from Piper's base model (LibriTTS-R,
CC BY 4.0). `BASE=maya` starts from a Telugu Piper voice instead and learns Telugu
faster, but that voice was trained on IIT Madras IndicTTS data, and that licence may
not allow commercial use. Check with IIT Madras before shipping a voice built on it.

### Day 8-9: ship it on Render

1. Put `te_IN-suma-medium.onnx` and `.onnx.json` in a **private** Hugging Face model
   repo (the voice is the company's asset and the speaker's likeness).
2. Render → Environment:
   ```
   VA_TTS_ENGINE=piper
   PIPER_VOICE=te_IN-suma-medium
   PIPER_MODEL_URL=https://huggingface.co/<org>/<repo>/resolve/main/te_IN-suma-medium.onnx
   PIPER_MODEL_TOKEN=<a Hugging Face read token>
   ```
   To use a public voice instead, set only `VA_TTS_ENGINE=piper` and
   `PIPER_VOICE=te_IN-padmavathi-medium`.
3. Deploy. On start the server downloads the voice if it isn't on disk (about 15 s for a
   63 MB voice; Telugu calls use Edge until it's ready), then loads it. Optional: set the
   Build Command to `pip install -r requirements.txt && python -m app.streaming.piper_fetch`
   so the voice is in the build and the start skips the download.
4. The log shows `[Startup] Piper voice warmed up` and `[Factory] Piper TTS lang=te voice=...` on a
   classic-pipeline call. The voice and its runtime add about **340 MB of RAM**
   (measured), so use a Render instance with at least 1 GB.

### Day 10: polish

Compare against Edge with `compare_voices.py --edge`, then tune on Render (no
redeploy of code needed, just the environment):

| Setting | Default | Effect |
|---|---|---|
| `PIPER_LENGTH_SCALE` | 0.65 | pace; lower is faster (the agent's Speech Rate scales it) |
| `PIPER_NOISE_SCALE` | 0.667 | voice variation; higher sounds livelier, too high gets shaky |
| `PIPER_NOISE_W` | 0.8 | rhythm variation; higher sounds less metronomic |

## Retrain with new data

Add recordings to `data/raw/`, run the three Day 3-4 commands (new clips are
appended to `transcripts.tsv`), correct the new lines, then `bash train.sh fit`
on the GPU: it resumes from the newest checkpoint. Export, upload over the same
files in the Hugging Face repo, and press **Manual Deploy → Clear build cache &
deploy** on Render so the build downloads the new voice.

## Changes from the original plan

* Commands use the current Piper (`piper1-gpl`: `python -m piper.train fit`), not the
  archived `piper_train` module.
* The voice runs inside the call server rather than as a separate `/tts` service: no
  network hop between the call and its voice, one deploy, one bill.
* Fine-tuning starts from a licence-clean checkpoint by default (see above).
