# humanize.md: how SUMA agents should sound on a call

**Read this before every "humanize the prompt" change.** The rules come from published research and vendor guides (sources at the end), plus what our own test calls showed (marked **[our tests]**). The prompt that applies them is `_STYLE` and the tables around it in `app/streaming/realtime_session.py`, shared by the OpenAI and Gemini engines.

**Target:** the agent should sound like the business owner talking on the phone: relaxed, direct, warm, short, everyday Hyderabad Telugu with the English words people really use. It should never sound formal, never like a call centre, never like a reading.

---

## 1. Speed: the main thing that makes a call feel robotic

- **What people expect:** on the phone, people reply about **200 ms** after the other person stops. A pause of 1 s already feels slow, and after about 2 s callers say "hello?".
- **Fixed pause, not "semantic" detection:** end of turn is a fixed silence of **500 ms** (`VA_END_OF_TURN_MS`). Callers pause 300–450 ms at commas, so anything shorter makes the agent answer half a sentence. **[our tests]**
- **Avoid semantic turn detection for Telugu:** it can wait seconds on real speech. **[our tests]**
- **Greeting:** prepare it while the phone rings, so the first word plays the instant the call connects. Both engines do this. On OpenAI, send the greeting request right behind `session.update` instead of waiting for `session.updated`; the wait cost ~0.7 s at pickup. **[our tests]**
- **Engine reply delay, measured from caller stopping to agent speaking:** Gemini `gemini-3.8-live` **0.9 s**, OpenAI `gpt-realtime-1.5` **1.1 s**, the three-step `gpt-4o-mini` pipeline **3.5–4 s** (too slow). **[our tests]**
- **Avoid reasoning models** (`gpt-realtime-2.1`, Premium) for calls. They add delay and say "let me check…" before acting. **[our tests]**
- **Speaking pace:** about 150–160 words a minute. Keep the agent's **Speech Rate at 1.05–1.15×**; 0.9× sounds sleepy. Gemini ignores this setting, so its prompt asks for a brisk pace instead.
- **Keep replies short:** short replies also *feel* faster, because the caller gets the turn back sooner.

## 2. Prompt structure (OpenAI Realtime guide, Google Live API best practices)

- Use labelled sections in this order: **Role & objective → Personality & tone → Language → Pacing & length → Variety → Sample phrases → Sound like a person → Reading the caller → Conversation flow → Unclear audio → Guardrails → Tools**.
- Write short bullets, never long paragraphs.
- Make no rule contradict another. Conflicting rules make the model behave unpredictably.
- Put only the few rules that keep getting broken in CAPS. If everything is in capitals, nothing stands out.
- **Sample phrases steer the model most strongly.** Give a few example lines in the agent's language, mark them "style only, vary them", and use no business facts in them (the model copies facts from samples).
- **Pin the language** in both the prompt and the engine settings (Gemini: `languageCode: te-IN`, and transcription `languageCodes: ["te-IN"]`).
- **Write rules in plain words, not code-like logic.**
- **Spell out each tool's trigger:** when to call `book_appointment`, and when to call `end_call`.

## 3. The shape of a reply

- **One or two short sentences, about 15 words.** Never a speech, never a list.
- **One fact per reply.** Answer what was asked and let them ask for more. Don't pack price, location, loan and pickup into one reply. **[our tests]**
- **Most replies end with a statement.** Ask only when you truly need an answer, and ask at most one question.
- **A simple question gets a simple answer:** "అవునండి, ఉంది." is a complete reply.
- **Offer specifics instead of open questions:** "ఆదివారం పొద్దున పదింటికి పెట్టుకుందామా?", not "ఏ టైమ్ కావాలి?". **[our tests]**

## 4. AI tells to remove (all seen in our test calls)

| Tell | Fix |
|---|---|
| Caller's name in every reply ("రవి గారు" 6 of 6) | Name in the greeting, and at most once more in the call |
| A checking question tacked on ("ఇంకేమైనా డౌట్స్?", "ఎలా అనిపిస్తుంది?") | End with a statement |
| Introducing itself twice; a sales pitch when asked "who's calling?" | Intro once; "who's calling" gets one sentence: name, company, why |
| Formal written Telugu (సౌలభ్యం, ప్రారంభం, అందుబాటులో, ధన్యవాదాలు, సదుపాయం, ప్రయాణం) | Spoken forms (వీలవుతుందా, మొదలవుతుంది, ఉంది, థాంక్యూ అండి, ఉంది, డ్రైవ్) |
| Narrating its actions ("టైమ్ ఫిక్స్ చేసి కన్ఫర్మ్ చేస్తాను") | Just do it |
| Every reply starting the same way ("సరే రవి గారు…") | Vary openers; no opener at all is fine |
| Filler words as padding (uh, hmm, so, అంటే, అవునా… every turn) | None as padding; at most one natural one in a whole call. The owner disliked "stopwords". |
| Call-centre lines ("I understand your concern", "thank you for your valuable time") | Never |
| A full English sentence in a Telugu call | English nouns only, unless the caller switches |
| Digits read out, or "ఎయిటీన్ లాక్స్" when the style is pure Telugu | Numbers the way people say them |
| A tool name spoken aloud ("…bye! (end_call)") | Never say tool names; the engine also forces a goodbye if the model hangs up silently |
| Over-thanking or over-apologising | One thank-you, at the end |

## 5. Human touches that work

- **Warmth through word choice and pace,** not through fillers or exclamations.
- **A small reaction now and then,** such as "హా అండి" or "అవునండి". Never twice in a row; often none.
- **Mirror the caller:** quick and brief if they're rushed, relaxed if they're chatty.
- **Emotion:** acknowledge it in a few words, then act. Frustrated: slow down, keep it very short, offer a way out. Unsure: give them space. Happy: match the warmth.
- **Honest unknowns:** "అది చూసి చెప్తానండి."
- **When interrupted,** stop immediately and answer what they said. Never say "as I was saying".
- **A concern always comes first,** even when the caller asks something else in the same breath ("ఎక్కువ అనిపిస్తుంది, లొకేషన్ ఎక్కడ?" means address the price, then the location). **[our tests: still the weakest spot]**

## 6. Telugu specifics

- **English nouns people really say:** టైమ్, రేట్, ప్లాట్, సైట్ విజిట్, బుక్, కన్ఫర్మ్, లొకేషన్, లోన్, ఓకే. Small connecting words (but, and, so, around, also) stay in Telugu.
- **Spoken, not written forms:** చెప్తాను (not చెబుతాను), వీలవుతుందా (not సౌలభ్యం ఉంటుందా), మొదలవుతుంది (not ప్రారంభమవుతుంది), ఉంది (not అందుబాటులో ఉంది), థాంక్యూ అండి (not ధన్యవాదాలు).
- **Respect:** "గారు" with the name, used rarely; "అండి" for softness.
- **Numbers:** in the mixed style, Tenglish ("ఎయిటీన్ లాక్స్", "సండే మార్నింగ్ టెన్ కి"); in the pure style, Telugu words ("పద్దెనిమిది లక్షలు", "నూట యాభై గజాలు").
- **Mini models** need a short brief (`_MINI_STYLE`, about half the length of `_STYLE`); they garble long rule lists. The brief still asks for natural code-mixing: forcing plain Telugu made mini sound like a newsreader ("పరిష్కారాలను అందిస్తున్నాం", "ధన్యవాదాలు"). **[our tests]**

## 7. Turn-taking, silence and endings

- **Talked-over replies:** drop any audio that was already on its way; never play over the caller. **[our tests]**
- **Silence:** check in after about 5 s of quiet with a fixed line ("హలో అండి, వినపడుతుందా?", then "హలో... ఉన్నారా అండి?"). Give up politely after two check-ins. Send these without the conversation in view, or mini carries on its pitch instead. **[our tests]**
- **Booking:** book any agreed visit before the goodbye, on exactly the day the caller said. A call back is not a booking.
- **Ending:** say the goodbye out loud, ending in "bye", then hang up. Never hang up mid-goodbye. Mini never calls `end_call`, so the code hangs up once a reply ending in bye has played. **[our tests]**
- **Busy, not interested, wrong number:** accept it the first time, one line, bye. Never pitch to a wrong number.

## 8. Voice and engine settings

| Setting | OpenAI | Gemini |
|---|---|---|
| Model | `gpt-realtime-mini` (all tiers, since 2026-11-21) | `gemini-3.8-live` |
| Voice | `marin` (female), `echo` (male, closest to the owner's reference voice; `VA_MALE_VOICE` overrides) | `Kore` (female), `Charon` (male) |
| Turn detection | `server_vad`, 500 ms | automatic activity detection, 500 ms |
| Speed | Speech Rate 1.05–1.15× | prompt asks for a brisk pace |
| Language pin | prompt | prompt + `te-IN` + Telugu transcription |
| Cost per minute (measured) | ~₹9–10 | ~₹3.5–4 |

## 9. How to check a change

1. Run the same test call: `VA_DATABASE_URL=sqlite:///local_test.db python call_tests/simulate_call.py <label> standard`.
   For the awkward moments (cut-ins, hang-ups, busy, wrong number, silence), run the scenarios with a caller that reacts like a person:
   `VA_DATABASE_URL=sqlite:///local_test.db python call_tests/scenario_call.py <label> [scenario ...]`.
2. Compare these targets with the last recording:

| Measure | Target |
|---|---|
| Caller's name said | ≤ 2 |
| Replies ending in a checking question | 0 |
| Words per reply | ≤ 12 on average |
| Formal words | 0 |
| Reply delay | ≤ 1.2 s |
| First word after pickup | ≤ 0.6 s |
| Booking and goodbye | both happen |

3. **Listen to the recording:** numbers can't judge the voice.
4. **The real test:** someone who doesn't know it's an AI takes a 3-minute call. Note when, or whether, they notice.

## 10. Log of findings

- **2026-11-18:**
  - Filler-word instructions made the agent sound worse; removed.
  - Semantic turn detection felt slow; went back to a fixed 500 ms pause.
  - The agent said the name in every reply, tacked on checking questions, and used written Telugu; fixed in the prompt.
  - The mini models are too weak for Telugu.
  - Gemini 3.8 Live: about 60% cheaper and 0.2 s faster, transcribes Telugu exactly, but still uses some formal words.
  - The owner says calls still sound robotic and slow. Next: full section rewrite, sample phrases, pace rule.
- **2026-11-18, prompt rebuilt from this file** (sections in the section-2 order, Telugu/Hindi/English sample phrases, brisk-pace rule, unclear-audio rule):
  - **Gemini:** replies 0.87 s; 10 words per reply (was 12.2); formal words down from 3 to 1; suggests a specific time when booking; ₹2.74 per test call.
  - **Gemini copies the sample phrases** ("ధర ఎక్కువే అనిపించొచ్చు అండి, కానీ…"), so **sample phrases are the strongest lever.**
  - **Gemini drifts into newsreader Telugu** (కేవలం, ఏర్పాట్లు, ధన్యవాదాలు) unless they are banned in CAPS. The CAPS list fixed most of it; "ధన్యవాదాలు" in the goodbye still slips through.
  - **OpenAI** kept its gains (name once, no checking questions, no formal words) with replies of about 1.1–1.4 s.
  - **Test artifact:** the simulated caller says "ఎక్కువ అనిపిస్తుంది." and then "లొకేషన్ ఎక్కడ?" after a full stop, so the agent's answer to the concern gets cut off. A real caller usually waits, so judge the concern rule on real calls.
  - **Still to fix:** the goodbye turn sometimes packs three sentences (confirmation, pickup info, thanks).
- **2026-11-21, live calls moved to `gpt-realtime-mini`** (about a fifth of the cost: ₹0.92 vs ₹4.43 for the same 21 s test call on `gpt-realtime-2.1`):
  - The old mini prompt was written for plots only and banned English, so mini spoke textbook Telugu on a services call.
  - New general `_MINI_STYLE` (no business named, product and service names kept in English, everyday English words allowed, newsreader words banned) plus the greeting sent with the session settings. Simulated call, before → after: first word after pickup 1.35 s → 0.0 s; average reply 1.43 s → 1.20 s; "అవును అండీ" opening 4 of 6 replies → no opener used more than 3 times; formal words 3 → 0; booking confirmed as done instead of asked.
  - Mini used "గారు" on its own ("సరే గారు"); the respect rule now says గారు only right after the name.
  - **Still to fix:** mini repeats "సరే అండి" as an opener and sometimes garbles the goodbye.

- **2026-11-22, scenario calls with a reacting caller** (an LLM persona that hears the agent and answers with a natural OpenAI voice; seven scenarios; reference: a real Hyderabad sales call by the owner, male, ~113 Hz, heavy Telugu-English mix):
  - Before: the agent never hung up after a goodbye (0 of 5 calls), booked a callback as an appointment, said a different day than the caller, kept pitching after "no" and to a wrong number, said "వెళ్ళిపోరా" and "Good night" at random, and the silence check-ins carried on the pitch.
  - After: the agent ends 7 of 7 calls itself after "... bye", stops within 0.4–0.5 s when cut off, ends busy and not-interested calls in one line, and the silence check-ins are exact. Price is right after a "NEVER MAKE UP A PRICE" rule (it once made up "ఆరు రెండు లక్షలు").
  - Opening line given as a fill-in shape ("హలో Ravi గారు, నేను <name>, <company> నుంచి. ఒక్క నిమిషం మాట్లాడొచ్చా అండి?") stopped garbled greetings.
  - English-heavy Telugu got its own sample phrases from the reference call (అన్నమాట, as well as, basically).
  - Male voice `cedar` → `echo`: ~120 Hz against the reference's 113 Hz (cedar ~153 Hz).
  - **Still to fix (mini's limits):** "ధన్యవాదాలు" and other banned words still slip in about once a call; odd phrases now and then ("ఫిర్యాదు చేస్తే").
- **2026-11-22, real calls to the owner (through a tunnel to a laptop):**
  - On an unclear first answer the agent re-introduced itself and gave a 15 s speech with a meeting push; it ignored "ఎందుకంటే?". Flow now: the greeting asks if they are free; if free, why you are calling in one sentence and ask if they want to hear more; if busy, ask when they will be free and say bye; ask for the next step only after they show interest.
  - The caller's words came out garbled and the agent sounded unclear, but the laptop's internet was dropping (ping 12–500 ms, OpenAI timeouts). Judge voice quality on production calls, not through a laptop.
  - Clarity on simulated calls (words garbled, judged ignoring script): mini 10–15%, mini at 0.9× speed 10–11%, `gpt-realtime-1.5` 7–13% at about five times the cost. Slower speech and the bigger model did not fix it.

## Sources

- [OpenAI: Realtime prompting guide](https://developers.openai.com/cookbook/examples/realtime_prompting_guide)
- [Google: Live API best practices](https://ai.google.dev/gemini-api/docs/live-api/best-practices)
- [Ciela: How to make Vapi sound more human](https://ciela.ai/blogs/how-to-make-vapi-sound-more-human)
- [Deepgram: Backchannels vs interruptions in voice agents](https://deepgram.com/blog/backchannels-vs-interruptions-voice-agents)
- [Max Planck Institute: Why responding quickly is important in conversation](https://www.mpi.nl/blog/why-responding-quickly-important-conversation)
- [Kendrick & Torreira 2015, timing in conversation](https://eprints.whiterose.ac.uk/116177/1/Kendrick_and_Torreira_2015_.pdf)
