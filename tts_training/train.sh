#!/usr/bin/env bash
# Day 5-8 on a GPU machine (Colab, Kaggle, RunPod, Vast.ai), with the current Piper (piper1-gpl).
#
#   export DATA=/content/data            # holds metadata.csv and wavs/ from build_metadata.py
#   export VOICE=te_IN-suma-medium       # <language>-<name>-medium, the name the server will load
#   bash train.sh setup                  # once per machine: Piper training tools + base checkpoint
#   bash train.sh smoke                  # minutes: proves data, checkpoint and export all work
#   bash train.sh fit                    # the real fine-tune; run it again to resume after a disconnect
#   bash train.sh export                 # newest checkpoint -> $VOICE.onnx + $VOICE.onnx.json
#
# BASE picks the starting checkpoint:
#   base (default) - Piper's LibriTTS-R base model, CC BY 4.0: safe for commercial use
#   maya           - a Telugu voice trained on IIT Madras IndicTTS: learns Telugu faster, but that
#                    dataset's licence may not allow commercial use; check before you ship it
set -euo pipefail

WORK=${WORK:-$PWD/piper_work}
BASE=${BASE:-base}
BATCH=${BATCH:-16}
EPOCHS=${EPOCHS:-2000}     # Piper's suggested starting point for under 5 h of audio
SMOKE_CLIPS=50
CKPTS=https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main
MAYA_EPOCH=3277            # the maya checkpoint's own epoch count; resuming must go past it

SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
mkdir -p "$WORK"
cd "$WORK"

base_ckpt() {
  if [ "$BASE" = maya ]; then echo "$WORK/maya.ckpt"; else echo "$WORK/base.ckpt"; fi
}

newest_ckpt() {
  ls -t "$WORK"/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1 || true
}

setup() {
  if command -v apt-get >/dev/null; then
    SUDO=$(sudo -n true 2>/dev/null && echo sudo || true)
    $SUDO apt-get update -q && $SUDO apt-get install -y -q build-essential cmake ninja-build
  fi
  [ -d piper1-gpl ] || git clone --depth 1 https://github.com/OHF-voice/piper1-gpl.git
  (cd piper1-gpl && python3 -m pip install -q -e '.[train]' && ./build_monotonic_align.sh && python3 setup.py build_ext --inplace)
  if [ "$BASE" = maya ]; then
    [ -f maya.ckpt ] || curl -L -o maya.ckpt "$CKPTS/te/te_IN/maya/medium/epoch%3D3277-step%3D839202.ckpt"
  else
    [ -f base.ckpt ] || curl -L -o base.ckpt "$CKPTS/_base_model/base_model.ckpt"
  fi
  python3 -c "import piper.train" && echo "setup done"
}

# train <csv> <cache dir> <epochs to add>
train() {
  local csv=$1 cache=$2 add=$3 max resume
  local start=()
  resume=$(newest_ckpt)
  if [ -n "$resume" ]; then
    start=(--ckpt_path "$resume")
    max=$(( $(python3 -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" "$resume") + add ))
  elif [ "$BASE" = maya ]; then
    start=(--ckpt_path "$(base_ckpt)")   # same language and one speaker: resume it as-is
    max=$(( MAYA_EPOCH + add ))
  else
    start=(--model.warmstart_ckpt "$(base_ckpt)")  # 904 English speakers -> copy weights, fresh schedule
    max=$add
  fi
  python3 -m piper.train fit \
    --data.voice_name "$VOICE" \
    --data.csv_path "$csv" \
    --data.audio_dir "$DATA/wavs" \
    --model.sample_rate 22050 \
    --data.espeak_voice te \
    --data.cache_dir "$cache" \
    --data.config_path "$WORK/$VOICE.onnx.json" \
    --data.batch_size "$BATCH" \
    --trainer.max_epochs "$max" \
    "${start[@]}"
}

export_voice() {
  local ckpt
  ckpt=${1:-$(newest_ckpt)}
  [ -n "$ckpt" ] || { echo "no checkpoint yet"; exit 1; }
  python3 -m piper.train.export_onnx --checkpoint "$ckpt" --output-file "$WORK/$VOICE.onnx"
  python3 - "$WORK/$VOICE.onnx" <<'EOF'
import sys, wave
from piper.voice import PiperVoice
voice = PiperVoice.load(sys.argv[1])
with wave.open(sys.argv[1].replace(".onnx", "_sample.wav"), "wb") as out:
    voice.synthesize_wav("నమస్కారమ్ అండి, నేను సుమ మాట్లాడుతున్నాను. మీకు ఇప్పుడు రెండు నిమిషాలు వీలవుతుందా?", out)
EOF
  echo "exported from $ckpt -> $WORK/$VOICE.onnx (+ .onnx.json); listen to ${VOICE}_sample.wav"
}

case ${1:-} in
  setup) setup ;;
  smoke)
    head -n "$SMOKE_CLIPS" "$DATA/metadata.csv" > "$WORK/smoke.csv"
    rm -rf "$WORK/lightning_logs"   # a smoke run never leaves a checkpoint for fit to resume
    train "$WORK/smoke.csv" "$WORK/smoke_cache" 1
    export_voice
    rm -rf "$WORK/lightning_logs"
    echo "smoke run worked: the sample will sound rough after one epoch, that is expected" ;;
  fit) train "$DATA/metadata.csv" "$WORK/cache" "$EPOCHS" ;;
  export) export_voice "${2:-}" ;;
  *) sed -n '2,14p' "$SCRIPT"; exit 1 ;;
esac
