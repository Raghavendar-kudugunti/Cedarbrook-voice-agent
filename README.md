# Cedarbrook Hospital · OPD voice agent (Team Beta)

A real-time, speech-to-speech appointment agent for a fictional 300-bed hospital in Pune. A caller talks to it in the
browser over WebRTC. It books, reschedules and cancels OPD appointments through tool calls to a scheduling service, and
hands emergencies, complaints, billing and anyone who asks for a person to the front desk. Every call leaves a call
record, a masked transcript, per-turn latency and (with keys) a Langfuse trace.

All doctors, patients and phone numbers are synthetic.

## How a call flows

```
Browser mic ──WebRTC──> LiveKit Cloud ──> voice worker (agent/worker.py)
                                            │  Silero VAD + LiveKit multilingual turn detector (end of turn, barge-in)
                                            │  Deepgram Nova-3 `multi`  (streaming STT, Hindi-English code switching)
                                            │  on_user_turn_completed: CODE guardrails before any LLM call
                                            │     emergency → fixed message + transfer_to_human("emergency"), no booking
                                            │     "talk to a human" / complaint / billing → fixed transfer message
                                            │     medicine / dose / "is it serious" → fixed no-advice message
                                            │     caller's yes/no → read-back confirmation gate (deterministic)
                                            │  Groq gpt-oss-120b, tool calling, preemptive generation
                                            │  Cartesia Sonic-3 (en / hi), streamed sentence by sentence
                                            └─ HTTP tools ──> FastAPI scheduling service (backend/) ──> SQLite
Browser <── agent audio + data channel (masked transcript, live latency, barge-ins, call record)
Staff view (same page) <── /api/dashboard: today's calls, outcomes, transfers waiting, latency p50/p95, harness results
```

This is the STT → LLM → TTS cascade from the LiveKit reference, not a single speech-to-speech model. Cascades are what
make the safety rules enforceable. Code can read the transcript and stop an emergency or an unconfirmed booking before
the model acts. You can also swap any stage and measure each stage's latency.

## Run it

Requires Docker, plus keys for LiveKit Cloud, Deepgram, Cartesia and Groq (all have free tiers).

```powershell
Copy-Item .env.example .env   # fill in keys; AGENT_INTERNAL_TOKEN = any long random string
docker compose up --build
```

Two pages, for two audiences:

* <http://localhost:8000> is the **caller page**: one call button, live captions, the emergency number, and a plain
  summary of what happened when the call ends. Press **Start call** and allow the microphone.
* <http://localhost:8000/staff> is the **front-desk console**: people waiting for a human (emergencies first, with a
  waiting timer), reply time for the latest turn split into stages, today's calls with a details drawer
  (call record, masked conversation, scheduling actions), and the latest test run.

`api` serves both pages and the API; `voice` is the LiveKit worker. Without voice keys the console and API still work.
Set `PORT=8020` before `docker compose up` if port 8000 is taken.

Without Docker (Python 3.11+):

```powershell
python -m venv .venv; .venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m agent.worker download-files
.venv\Scripts\python -m uvicorn backend.api:app --port 8000      # terminal 1
.venv\Scripts\python -m agent.worker start                      # terminal 2
```

## Tests

```powershell
.venv\Scripts\python -m pytest -q tests          # deterministic: scheduling, safety gates, guardrails (no network)
.venv\Scripts\python -m tests.harness            # text harness: every script, LLM-simulated caller (uses Groq)
.venv\Scripts\python -m tests.harness B02 B09    # some scripts
```

* `tests/test_scheduling.py` covers: hours, days, leave, Sunday, 25 Dec, same-day walk-in, the 14-day window, about 60%
  pre-booked with a fixed seed, "never invent a slot", the commit refused without a yes, the digit-correction read-back,
  idempotent book and double cancel, a 12-thread race for one slot (exactly one wins), a slot taken between offer and
  hold, reschedule freeing the old slot, one active appointment per patient per doctor, a family sharing a number,
  the paediatrics age rule (15 yes, 16 no), no booking after an emergency, and one booking per call (prompt injection).
* `tests/test_guardrails.py` covers emergency phrases in English, Hinglish, Marathi and Devanagari (and
  non-emergencies), medical advice, human requests, complaints/billing, language detection, and the exact fixed messages.
* `tests/harness.py` runs `tests/scripts.json` (B01–B10 plus edge cases E01–E05) through **the same `CedarbrookAgent`**
  (prompt, guardrail hook, tools, scheduling service) in LiveKit's text mode. A second LLM plays the caller. It checks
  the final database state, tool order (read-back before commit), the call record and the safety invariants (no double
  booking, every commit confirmed). Results go to `docs/harness-results.md|json` and the staff view. "Today" is pinned
  to Monday 16 Nov 2026 so B07 (leave 17–21 Nov) and B08 (Sunday) are meaningful.

## The rules I had to choose

| Rule | Value / behaviour | Where |
|---|---|---|
| End of turn | VAD (min silence 0.25 s) + multilingual turn detector, endpoint delay 0.3–2.0 s, preemptive LLM start | `agent/worker.py` |
| Barge-in | VAD mode, 0.2 s of caller speech stops agent audio; false-interruption resume after 1.5 s | `agent/worker.py` |
| Filler | spoken if `get_slots` / `find_appointments` takes > 700 ms | `agent/core.py` |
| Confirmation | `prepare_confirmation` stores a read-back. The next caller utterance is classified in code. Yes = starts with a clear yes word (yes, haan ji, theek hai, हाँ, बरोबर…) with no negation and no digits. Anything with a digit or a no is a correction. `book/reschedule/cancel` refuse without a confirmed read-back matching the same hold / appointment / phone | `backend/scheduling.py` |
| Double booking | `slot_locks.slot_id` PRIMARY KEY + unique index on active appointment slot; writes in `BEGIN IMMEDIATE` | `backend/scheduling.py` |
| Hold | 120 s; only for slots `get_slots` returned in this call | same |
| Idempotency | `call_id:book:slot`, `call_id:reschedule:appt:slot`, `call_id:cancel:appt` (+ hold-id aliases for retries) | same |
| Emergencies | keyword detector (EN / Hinglish / Marathi / Devanagari), tuned for recall: a false positive is a transfer, a miss is not acceptable. The LLM has `transfer_to_human` as a backstop. The call is then locked: no hold, read-back or booking | `agent/guardrails.py` |
| Transfers | queued in the staff view (no outbound phone call), fixed message, then the agent hangs up | `agent/core.py` |
| Booking limit | one booking per patient name per call, at most 3 family members | `backend/scheduling.py` |
| Silence | 6 s → "Are you still there?" once, 6 s more → goodbye and end | `agent/worker.py` |
| Max call | 5 min → offers a transfer | `agent/worker.py` |
| Privacy | no audio stored; transcripts, tool logs and traces keep only the last 4 phone digits and the first name. The scheduling DB keeps full numbers because it must find appointments by phone | `backend/scheduling.py` |
| Spoken formats | "Wednesday, 18 November", "quarter past nine in the morning", phone as two groups of five digits | `backend/directory.py` |
| Dates | today (IST) plus the 14 bookable dates are in the system prompt, so "parso" and "next Friday" resolve to real dates | `backend/directory.py` |

## Models, providers and data terms

| Stage | Provider / model | Why | Data terms (check before real use) |
|---|---|---|---|
| Transport | LiveKit Cloud (India South) | WebRTC, agent dispatch, closest region | Media relayed, not stored by this app |
| STT | Deepgram Nova-3, `language=multi` | streaming, Hindi-English code switching | Deepgram may use audio for training unless you opt out (`mip_opt_out`) |
| VAD / turns | Silero VAD + LiveKit multilingual turn detector | runs locally | Local |
| LLM | Groq `openai/gpt-oss-120b`, reasoning effort low | fast first token, reliable tool calls | Groq states API inputs are not used for training |
| TTS | Cartesia Sonic-3, `en` / `hi` by detected language | low first-audio latency, Hindi | Check the Cartesia terms |
| Tracing | Langfuse via OTLP | one trace per call; turn → stt / llm / tool.* / tts spans | Masked data only |

Known gaps: Nova-3 `multi` does not list Marathi, so Marathi speech is transcribed less reliably than Hindi or
English. Marathi replies use the Hindi TTS voice (`TTS_LANG_MR`).

## Cost (estimates at about ₹85/USD; override in `.env`)

| Item | List price used | INR per unit |
|---|---|---|
| Deepgram Nova-3 multilingual streaming | $0.0092 / min | ₹0.013 / audio second |
| Cartesia Sonic | about $0.038 / 1k characters | ₹0.0032 / character |
| Groq gpt-oss-120b | $0.15 / M in, $0.60 / M out | ₹0.0000128 / ₹0.000051 per token |
| LiveKit Cloud | free tier minutes | ₹0 at demo volume |

A 2-minute booking call is roughly 60 s of caller audio (₹0.8), about 900 spoken characters (₹2.9) and about 40k LLM
tokens (₹0.6), so **about ₹4–5 per call**. Each call record stores its own `cost_inr`.

**Groq tier matters.** The free tier allows `gpt-oss-120b` 8,000 tokens/minute and 200,000 tokens/day. A booking call
uses about 30–50k tokens because the prompt and tools are resent each LLM step. That is only 4–6 calls a day, and a
fast multi-step turn can hit the per-minute limit mid-call. For the demo and the hidden test, move Groq to the
pay-as-you-go Dev tier (₹ cost above, far under the ₹2,000 LLM cap) and set a spend limit.

## Layout

```
agent/      core.py (agent, prompt, tools) · guardrails.py · session_hooks.py (transcript, latency, record) · worker.py · backend_client.py
backend/    api.py (FastAPI) · scheduling.py (rules, DB, call records) · directory.py (doctors, calendar, spoken formats) · telemetry.py (Langfuse)
frontend/   index.html + caller.js/.css (caller page) · staff.html + staff.js/.css (front-desk console) · base.css (shared tokens)
tests/      test_scheduling.py · test_guardrails.py · harness.py · scripts.json
docs/       harness-results.md / .json (latest run)
```

## Demo script

1. English: "I'd like Dr. Anil Mehta, next available morning." Show the read-back on the caller page and the reply-time
   panel on the staff console.
2. Hinglish: "Dr. Mehta ke saath appointment chahiye." Answer "skin wali", interrupt the slot list with "No no, Friday
   instead", and book Dr. Rekha Mehta.
3. "Mere papa ko abhi chest pain ho raha hai." Show the emergency message and the red transfer in the staff view.
4. Open a call row to show its call record, masked transcript, tool calls and per-turn latency. Show the Langfuse trace
   and the harness results.

## Latest harness status (23 Nov 2026)

The production model (`gpt-oss-120b`) could not be run: the Groq free-tier daily token cap was used up. Run on
`gpt-oss-20b` (a weaker stand-in) with a Qwen simulated caller:

* Passed: B02 (Hinglish skin doctor Friday evening), B09 (chest pain: fixed message + emergency transfer, no booking),
  B10 (no medical advice). Safety invariants held in every script: 0 double bookings, every commit confirmed.
* Failed, then fixed in code: holds lapsing while the caller gave details (now re-taken if still free), the model
  doing date arithmetic (code now resolves "friday" / "parso" / "this sunday"), dead ends on Sunday or leave
  (`get_slots` now returns next available slots and same-department doctors), the model transferring calls it should
  handle (guarded), and the model inventing a phone number (refused unless the caller said it).
* The rerun of B01 and B03–B08 after those fixes was blocked by the 20b daily cap too, so the fixes are covered by
  unit tests only. Rerun `python -m tests.harness` once Groq quota is available (Dev tier recommended).

## Not done yet

* Stretch goal (Twilio number through LiveKit SIP) is not wired.
* Audio runs (recorded scripts by two team members) need real recordings.
* The staff view has no login. It is a local demo tool.
