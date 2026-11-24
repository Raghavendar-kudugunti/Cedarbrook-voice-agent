# Text harness results — 2026-11-23 17:07:57

Model: `openai/gpt-oss-20b`. Public scripts: **3/10**. All scripts: **3/10**.

| Script | Result | Outcome | Checks |
|---|---|---|---|
| B01 English, Dr. Anil Mehta, next available morning | FAIL | no_action | failed: outcome, appointment_exists, readback_before_commit |
| B02 Hinglish skin doctor Friday evening | PASS | booked | all ok |
| B03 Ambiguous Dr. Mehta | FAIL | transferred | failed: outcome, appointment_exists, asks_which_mehta |
| B04 Changes day mid-list (barge-in in text form) | FAIL | transferred | failed: outcome, appointment_exists |
| B05 Reschedule by phone number | FAIL | transferred | failed: outcome, old_slot_freed, readback_before_commit |
| B06 Cancel, says yes twice | FAIL | transferred | failed: outcome, cancelled_once, readback_before_commit |
| B07 Dr. Kulkarni on leave 19 November | FAIL | transferred | failed: mentions |
| B08 Asks for Sunday | FAIL | transferred | failed: outcome, appointment_exists, mentions |
| B09 Chest pain emergency | PASS | transferred | all ok |
| B10 Medicine question | PASS | no_action | all ok |
