# Voice Agent Platform (core)

A multi-tenant backend for AI phone agents that place and receive calls, log outcomes, and bill per connected second. Design details are in `docs/ARCHITECTURE.md`.

## What works today

- Tenants with hashed API keys and signed outcome webhooks
- AI employees: language, voice tier, script, calling window and timezone, concurrency, retries
- Lead intake: single, bulk, and the dispatch endpoint
- Compliance gate: calling windows in the recipient's timezone, DND registry, do-not-call status, retry spacing, attempt cap, campaign pause
- Carrier refusal auto-pauses the campaign
- Conversation loop with a pluggable LLM, TTS, and STT, plus tool calls (`book_appointment`)
- Per-second billing in integer paise, with an append-only ledger
- Outcome webhooks signed with HMAC-SHA256, retried on failure, with an attempt log
- Exotel outbound dialing adapter (endpoint paths need checking against your Exotel account)

## What is not built yet

- Real audio streaming. The call loop takes caller utterances as input; live STT and TTS streaming over Exotel's media websocket still needs to be written.
- Real speech and LLM vendors. Only offline mocks ship, so the voice quality per language is untested.
- Inbound call webhook handling and the Exotel status callback route.
- Dashboard, payments, CRM and WhatsApp connectors, and CSV upload endpoint.
- Real DND registry client and DLT template enforcement. The in-memory registry is a placeholder.
- A background worker. Dispatch currently runs when `POST /v1/dispatch` is called.

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
pytest -q
```

Set `EXOTEL_ACCOUNT_SID`, `EXOTEL_API_KEY`, `EXOTEL_API_TOKEN`, `VA_CALLER_ID`, and `VA_CALLBACK_URL` to place real calls. Without them the mock telephony is used.

## Compliance note

Before placing real calls in India you need DLT registration, DND/NDNC screening against the official registry, recorded consent for each list, and legal review. The code enforces the checks it can; the rest is a process requirement.

## Setting up a phone number

1. Create an Exotel account, complete KYC in Settings, then KYC / Company Details (1 to 3 business days; you need the company CIN, GST number, PAN, address proof, and a signatory ID).
2. In the Exotel dashboard, buy an ExoPhone. Virtual mobile numbers support outbound caller ID. Monthly rental applies.
3. Create a flow in App Bazaar with a stream applet pointing to `wss://<your-host>/v1/stream/{call_id}`. Copy the flow URL into `VA_EXOTEL_FLOW_URL`.
4. Assign the ExoPhone to an agent: `PUT /v1/employees/{id}/number` with `{"phone_number": "+91..."}`.
5. Set `EXOTEL_ACCOUNT_SID`, `EXOTEL_API_KEY`, `EXOTEL_API_TOKEN`, and `VA_EXOTEL_FLOW_URL`, then dispatch a test lead to your own phone.

Voice calls don't need DLT registration. DLT applies to SMS and WhatsApp templates. Calls still must respect DND and calling hours, which the compliance layer enforces.
