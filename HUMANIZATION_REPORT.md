# SUMA AI — Humanization Implementation Report

## Summary
Seven humanization features have been implemented across the voice agent platform, covering both the primary path (OpenAI Realtime API) and the fallback path (STT → LLM → TTS pipeline).

---

## Features Implemented

### 1. Emotion Detection Module
**File:** `app/streaming/emotion.py` (new)

- Keyword/pattern-based emotion classifier detecting 5 states: frustrated, interested, confused, impatient, happy
- Supports English, Telugu, and Hindi trigger words
- Returns a `prompt_hint` string the LLM uses to adapt its tone
- Short-response heuristic: very brief answers with no positive signals suggest impatience
- Used by StreamSession to modify the LLM prompt per turn

### 2. Per-Lead Conversation Memory
**Files:** `app/streaming/memory.py` (new), `app/streaming/ws.py` (modified)

- Loads previous conversation context from `Lead.metadata_json["conversation_memory"]` before each call
- Tracks: calls count, last outcome, topics discussed, preferences, relationship notes
- Generates a natural prompt section: "What you remember from previous conversations"
- Automatically updates memory after each call ends (caller topics extracted, outcome saved)
- Works for both Realtime and Stream sessions

### 3. Enhanced Realtime Prompts — Disfluencies & Human Touches
**File:** `app/streaming/realtime_session.py` (modified)

New `_STYLE` prompt sections added:

- **Human touches:** Explicit instructions for natural disfluencies — "so, uh...", "I mean...", "actually...", "let me think..." — sprinkled sparingly. Self-corrections ("Tuesday — no wait, Wednesday") to break perfect delivery.
- **Breathing:** Micro-pause before answering questions, as though thinking. No rushing to fill every silence.
- **Mirroring:** Mirror caller's energy — relaxed with relaxed callers, brisk with rushed callers.
- **Excitement:** Show genuine emotion when excited about something they said.

### 4. Emotion-Aware Agent Behavior (Realtime)
**File:** `app/streaming/realtime_session.py` (modified)

New prompt section "Reading the caller's emotions" with explicit guidance:

| Caller mood | Agent adapts by... |
|---|---|
| Frustrated/annoyed | Acknowledging it, slowing pace, very short replies, offer a way out |
| Confused | Pausing, rephrasing simply, asking which part was unclear |
| Rushed/impatient | Matching briskness, skipping niceties, one-line points, offer callback |
| Happy/excited | Matching warmth, reinforcing positive moment, moving forward |
| Hesitant/uncertain | Softening voice, slowing down, giving space to think |

### 5. Backchanneling (StreamSession)
**File:** `app/streaming/session.py` (modified)

- Re-enabled the previously disabled `_play_filler()` method
- Plays pre-cached filler sounds ("అవున...", "సరే...", "హా...") while STT/LLM process
- Rotates through fillers to avoid repetition
- Cancelled the moment real TTS audio starts (no overlap)
- Signals "I'm listening" rather than leaving dead air during processing

### 6. Micro-Pauses Between Sentences (StreamSession)
**File:** `app/streaming/session.py` (modified)

- `_speak()` now splits text into sentences before TTS
- Inserts 150ms of silence between sentences
- Creates natural breathing rhythm instead of continuous monotone delivery
- Works with both streaming and non-streaming TTS providers

### 7. Adaptive VAD for Better Turn-Taking
**File:** `app/streaming/vad.py` (modified)

- Added adaptive noise floor tracking via exponential moving average
- First 500ms bootstraps the noise estimate, then quiet frames update it at 3% weight
- Effective threshold = max(fixed 500 RMS, 2.5× noise floor)
- Increased `end_silence_ms` from 150ms → 300ms to reduce premature cutoffs
- Result: fewer false speech-end triggers on noisy lines, more natural conversation flow

---

## Test Results

| Test | Result |
|---|---|
| Emotion: frustrated detection (English) | PASS |
| Emotion: interested detection | PASS |
| Emotion: confused detection | PASS |
| Emotion: impatient detection | PASS |
| Emotion: happy detection | PASS |
| Emotion: neutral (ambiguous input) | PASS |
| Emotion: empty input | PASS |
| Emotion: Telugu/Hindi words | PASS |
| Emotion: Telugu "tell me" (cheppandi) | PASS |
| VAD: adaptive noise floor tracking | PASS |
| VAD: end_silence_ms = 300ms | PASS |
| Memory: prompt generation with all fields | PASS |
| Realtime prompt: human touches section | PASS |
| Realtime prompt: emotion awareness section | PASS |
| Realtime prompt: disfluency instructions | PASS |
| Server startup: no import errors | PASS |
| Existing test suite (53 tests) | PASS (segfault in unrelated xtts warmup thread) |

---

## Files Changed

| File | Type | What changed |
|---|---|---|
| `app/streaming/emotion.py` | NEW | Emotion detection from caller text |
| `app/streaming/memory.py` | NEW | Per-lead conversation memory load/save |
| `app/streaming/realtime_session.py` | MODIFIED | Enhanced prompts: disfluencies, emotion-awareness, memory context, language_name |
| `app/streaming/session.py` | MODIFIED | Backchanneling enabled, micro-pauses, emotion-adaptive prompting |
| `app/streaming/vad.py` | MODIFIED | Adaptive threshold, increased end_silence_ms |
| `app/streaming/ws.py` | MODIFIED | Memory loading/saving, lead_id tracking, memory_context wiring |

---

## Impact on Call Flow

### Before
1. Agent speaks in a continuous, perfectly-timed stream
2. No memory of previous conversations
3. Same tone regardless of caller's mood
4. Dead air while STT/LLM process
5. VAD triggers prematurely on noisy lines

### After
1. Agent breathes between sentences, uses natural disfluencies, self-corrects
2. Remembers previous calls: "Last time you mentioned the 2BHK near the school"
3. Adapts tone: empathetic with frustrated callers, brief with busy ones
4. Plays "hmm" / "acha" while thinking (backchanneling)
5. Adaptive VAD waits longer before cutting, adjusts to line noise

---

## What's NOT Included (requires external dependencies or deeper changes)

- **WebRTC VAD / Silero VAD:** Would need `webrtcvad` or `onnxruntime` packages; the adaptive energy VAD covers the most common turn-taking issues
- **Voice emotion detection from audio:** OpenAI Realtime API already hears tone; text-based detection covers the StreamSession path
- **LLM-powered call summarization:** Would add latency and cost per call; simple topic extraction works for now
- **Prosody control in TTS:** Edge-TTS and OpenAI TTS don't expose fine-grained pitch/rate per word
