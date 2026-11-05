# AI Voice Agent Platform: Architecture

Goal: a multi-tenant platform where businesses create AI "employees" that place and answer phone calls in Indian languages, log outcomes, and bill per connected minute. This is a clean-room design modelled on the public product shape of Outpero. It does not reuse any of its code.

## 1. Components

| Component | Responsibility |
|---|---|
| API (FastAPI) | Tenants, AI employees, leads, campaigns, lead intake, outcome webhooks, credit top-ups |
| Campaign scheduler | Picks due leads, enforces calling windows, concurrency, retry and callback timing |
| Dialer worker | Asks Exotel to place a call and hands the call to the call engine |
| Call engine | Runs one live call: audio in, STT, LLM turn, TTS, audio out, in-call actions, end-of-call |
| Telephony adapter | Exotel: outbound call API, inbound call webhook, audio streaming, call status callbacks |
| Speech adapters | STT, TTS, and LLM behind interfaces so providers can be swapped |
| Action executor | Calendar booking, WhatsApp send, CRM or custom API calls, human transfer |
| Billing ledger | Per-second usage events converted to credit debits; append-only |
| Outcome delivery | Signed webhooks, email, Google Sheets; retries with backoff |
| Compliance guard | DND/NDNC screening, DLT template checks, consent flags, do-not-call-twice, carrier-refusal auto-pause |

## 2. Data model

- **Tenant**: id, name, credit_balance (cached from ledger), api_key_hash, webhook_secret.
- **AIEmployee**: id, tenant_id, name, language, voice_tier (standard | premium | custom), script (system prompt), capture_fields (JSON schema), phone_number_id, calling_window (start, end, timezone), max_concurrency.
- **Lead**: id, tenant_id, employee_id, phone_e164, name, source, metadata (JSON), status (new | queued | calling | done | failed | paused | dnd), attempts, next_attempt_at, do_not_call.
- **Campaign**: id, tenant_id, employee_id, type (instant | bulk | inbound), status, retry_policy.
- **Call**: id, lead_id, employee_id, provider_call_id, direction, started_at, ended_at, duration_seconds, status, outcome, summary, captured_fields (JSON), recording_url.
- **CallTurn**: call_id, role (agent | caller), text, started_at, ended_at (transcript).
- **LedgerEntry**: id, tenant_id, kind (topup | usage | fee | refund), amount_paise (integer, never float), call_id nullable, created_at.

Money is stored as integer paise, never as floats. Credit balance is derived from the ledger.

## 3. Call lifecycle

1. Trigger arrives (webhook, CSV upload, inbound call, or scheduler picks a due lead).
2. Compliance guard checks: tenant has credits, lead is not DND, not called in the last window, inside calling hours, carrier not paused.
3. Dialer creates a Call row in `dialing`, asks Exotel to connect.
4. On answer, Exotel opens an audio stream to the call engine websocket.
5. Turn loop: caller audio to STT, partial transcript to LLM with the employee script and conversation state, LLM reply to TTS, audio back to the stream. Barge-in cancels in-flight TTS.
6. Tool calls from the LLM (book_appointment, send_whatsapp, lookup_record, transfer_to_human) run through the action executor and return results into the turn.
7. On hangup or completion, the engine writes the transcript, produces a summary and captured fields, classifies the outcome, and emits a ledger usage entry for connected seconds.
8. Outcome delivery pushes a signed webhook and any configured sinks.
9. Retry policy schedules the next attempt for unanswered or busy calls, capped at the configured number of attempts.

## 4. Decisions

- **Python + FastAPI** for the API and workers. Mature speech and AI client libraries.
- **PostgreSQL in production, SQLite in tests**, through SQLAlchemy 2.0. Single schema.
- **Provider interfaces** (`TelephonyProvider`, `STTProvider`, `TTSProvider`, `LLMProvider`) so Exotel, a speech vendor, and an LLM can be replaced without touching the call engine. A `Mock*` implementation of each makes the whole flow testable offline.
- **Queue**: start with a database-backed job table polled by a worker, which avoids extra infrastructure. Move to Redis or a managed queue when throughput needs it.
- **Languages**: all seven target languages (Hindi, Telugu, Tamil, Kannada, Malayalam, Marathi, Indian English). Language is a per-employee setting that selects the STT, TTS voice, and prompt locale. Quality per language needs native-speaker review before launch; this is the biggest non-engineering risk.
- **Telephony**: Exotel for DLT-registered Indian numbers, outbound and inbound, and call audio streaming.

## 5. Compliance (must be done before real calls)

- DLT registration of the sender and of each message template used in SMS or WhatsApp follow-ups.
- Do-Not-Disturb registry screening for every outbound number before dialing. Calls to numbers on the registry are blocked unless the business has documented consent.
- Calling-hour windows enforced in the recipient's timezone, with a default that avoids early morning and late evening.
- Recording disclosure: the agent states it is an AI and that the call may be recorded, as the employee script requires.
- KYC for any number the platform provisions, in the customer's own name.
- Auto-pause a tenant's campaign when the carrier refuses a high share of calls.

These are product requirements, not optional features. Legal review is needed before commercial launch.

## 6. Cost model (per connected minute, rough)

Speech-to-text, LLM tokens, text-to-speech, and telephony each have a per-minute cost. Pricing must cover the sum with margin. The reference pricing in the Outpero research (₹5 to ₹6.8 per minute, ₹3.5 at volume) is a benchmark; actual provider rates must be quoted before setting prices.

## 7. Out of scope for the first build

Multi-region deployment, a full web dashboard, self-serve payments, and Shopify, CRM, and WhatsApp connectors beyond a generic webhook. These come after the core loop works end to end.
