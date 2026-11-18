# How the SUMA agent sounds now: test call report

**Tested:** 18 Nov 2026, on the local server with your OpenAI key. No real phone was dialled.
**How:** `simulate_call.py` connects to the server over the same WebSocket Plivo uses and streams a Telugu caller (a male voice) in real time. The caller says five lines, waiting for the agent to finish each time, and both sides are recorded as they'd be heard on the phone (8 kHz).

The test agent was Suma from Green Valley Properties, selling DTCP plots at Shadnagar from 18 lakhs and trying to book a weekend site visit.

## Recordings

| File | What it is |
|---|---|
| `baseline.wav` | **Before**: premium tier (gpt-realtime-2.1), old prompt |
| `after_premium.wav` | After the fixes, premium tier (gpt-realtime-2.1) |
| `after_standard.wav` | **After the fixes, standard tier (gpt-realtime-1.5). Recommended** |

## Numbers

| | Before | After, premium | **After, standard** |
|---|---|---|---|
| First word after pickup | 0.76 s | ~0 s | **~0 s** |
| Average wait after caller stops | 1.48 s | 1.32 s | **1.30 s** |
| Words per reply (avg / longest) | 27 / 48 | 14 / 24 | **12 / 19** |
| Filler words ("uh", "hmm", "అంటే"…) | 0 | 0 | **0** |
| Site visit booked | yes | no | **yes** |
| Says goodbye before hanging up | no | yes | **yes** |

"Wait after caller stops" runs from the caller's last sound to the agent's first sound. About 0.6 s of it is the agent making sure the caller has really finished, and about 0.7 s is the model starting to speak.

## How it speaks now (after, standard tier)

The agent's lines are exactly as transcribed. The caller's lines are what the test caller said.

> **Agent:** హలో, ఇది రవి గారేనా? నేను సుమా మాట్లాడుతున్నా.
> **Caller:** హా చెప్పండి, ఎవరు మాట్లాడుతున్నారు?
> **Agent:** హా రవి గారు, నేను సుమా అండి, గ్రీన్ వ్యాల్లి ప్రాపర్టీస్ నుంచి. మీతో సైట్ విజిట్ గురించి మాట్లాడాలనుకున్నా. ఒక నిమిషం టైమ్ ఉందా?
> **Caller:** ఓకే, ఆ ప్లాట్ రేట్ ఎంత ఉంటుంది?
> **Agent:** రేట్ ఎయిటీన్ లాక్స్ ప్లాటింగ్ నుంచి స్టార్ట్ అవుతుంది రవి గారు. ఇదే రేట్ మీకు ఎలా అనిపిస్తుంది?
> **Caller:** కొంచెం ఎక్కువ అనిపిస్తుంది. లొకేషన్ సరిగ్గా ఎక్కడ?
> **Agent:** శాద్నగర్ దగ్గర లొకేషన్ ఉంది రవి గారు, ఎయిర్‌పోర్ట్ నుంచి 30 నిమిషుల్లో చేరొచ్చు. మీకు రావడానికి సౌలభ్యం ఉంటుందా?
> **Caller:** సరే, ఈ ఆదివారం ఉదయం వచ్చి చూస్తాను.
> **Agent:** చాలా బాగుంది రవి గారు, ఈ ఆదివారం ఉదయం సైట్ విజిట్ బుక్ చేస్తున్నాను. *(booking saved)*
> **Caller:** ఓకే థాంక్యూ, బై.
> **Agent:** సరే రవి గారు, థ్యాంక్స్, గుడ్ నైట్. *(hangs up)*

Before, a single reply sounded like this:

> అవునండి రవి గారు, నేను సుమా, Green Valley Properties నుంచి. Shadnagar లో DTCP-approved open plots ఉన్నాయి, airport నుంచి around 30 minutes, రేట్ ఎయిటీన్ లాక్స్ నుంచి స్టార్ట్ అవుతుంది. ఈ వీకెండ్ site visit కి రావడానికి ఇంట్రెస్ట్ ఉందా అండి? Free pickup Shamshabad నుంచి available.

## What was wrong, and what changed

1. **The cheap pipeline was default.** Calls went through separate speech-to-text, LLM and Edge voice steps, which is why they were slow and robotic. **OpenAI Realtime is the default again.** The cheap pipeline is now opt-in with `VA_VOICE_PIPELINE=classic`.
2. **The prompt asked for filler words** ("uh", "hmm", "so", "think out loud"). That's removed; the agent now sounds human through tone and pace instead.
3. **Replies were speeches** that packed price, location, loan and pickup into one turn. The agent now gives one fact per reply, about 20 words at most, and lets the caller ask for more.
4. **It pushed the site visit in every reply.** Now it asks at most every two or three replies.
5. **English connecting words** ("around", "but", "full exact map details"). Those words now stay in Telugu; only nouns like plot, rate and site visit stay English.
6. **It jumped in at commas**, the first time the caller paused. Turn detection is now `semantic_vad` (balanced), which waits until the caller has finished their thought.
7. **Talked-over replies still played.** When the caller kept talking after a pause, a reply OpenAI had already cancelled still played over them. That audio is now dropped.
8. **It hung up without a goodbye.** When the model hangs up silently, the agent now says a short goodbye first, then ends the call.
9. **It narrated what it was about to do** ("టైమ్ ఫిక్స్ చేద్దాం, ఒక్కసారి చెక్ చేస్తున్నా"). It's now told to just do it. The premium 2.1 model still does this sometimes before booking, because it's a reasoning model. The standard model doesn't.

## Recommendation

- **Switch your agent's voice tier from Premium to Standard** in the dashboard. In these tests the standard model was as fast, booked cleanly, and didn't do the "let me check…" narration. It should also cost less than premium.
- These changes only reach real calls after they're committed and pushed (Render redeploys in about 90 s).

## Known limits

- **It can skip a price objection.** When the caller says "కొంచెం ఎక్కువ" and asks something else in the same breath, it answers the question but not always the concern. The prompt now says to cover both; it doesn't always.
- **The test caller is synthetic.** Transcription misheard it a few times (e.g. "ప్లాంట్ వెయిట్" for "ప్లాట్ రేట్"); a real person's voice transcribes better.
- **Minor Telugu slips appear** (e.g. "నిమిషుల్లో"). That comes from the model.
- **Realtime costs more per minute than the cheap pipeline.** Standard tier is the lower-cost Realtime option.

## Rerun it yourself

```bash
VA_DATABASE_URL=sqlite:///local_test.db python call_tests/simulate_call.py mytest standard
```

The local server must be running. The script only writes to the local SQLite database and refuses to run against production.
