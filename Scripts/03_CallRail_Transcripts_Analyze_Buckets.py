# analyze_callrail_transcripts_buckets_nightly.py
#
# Two-model pipeline:
#   Step 1 (Reasoning): Gemini 2.5 Pro reads transcripts, produces structured reasoning summaries
#   Step 2 (Classification): Gemini 2.5 Flash classifies from reasoning summaries using strict JSON schema
#
import os
import json
import time
import logging
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------------
# Load .env (LLM_API_KEY, LLM_BASE_URL, SQLSERVER_*, etc.)
# ------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import pyodbc
except ImportError:
    pyodbc = None  # Not needed for validation/offline use
from openai import OpenAI

# ============================================================
# DEFAULTS (override via CLI flags or env vars)
# ============================================================

# Provider configuration
PROVIDERS = {
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url_env": "GEMINI_BASE_URL",
        "base_url_default": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "reasoning_model_default": "gemini-2.5-pro",
        "classification_model_default": "gemini-2.5-flash",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url_default": "https://api.openai.com/v1",
        "reasoning_model_default": "gpt-5",
        "classification_model_default": "gpt-4o-mini",
    },
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url_env": "ANTHROPIC_BASE_URL",
        "base_url_default": "https://api.anthropic.com/v1/",
        "reasoning_model_default": "claude-sonnet-4-5-20250514",
        "classification_model_default": "claude-haiku-4-5-20251001",
    },
}

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

DEFAULT_REASONING_MODEL = os.getenv("REASONING_MODEL")
DEFAULT_CLASSIFICATION_MODEL = os.getenv("CLASSIFICATION_MODEL")

DEFAULT_ANALYSIS_VERSION = os.getenv("ANALYSIS_VERSION", "prod_v2")
DEFAULT_ANALYSIS_PREFIX = os.getenv("ANALYSIS_PREFIX", "v3_fast_bucketed_run_")

DEFAULT_REASONING_BATCH_SIZE = int(os.getenv("REASONING_BATCH_SIZE", "4"))
DEFAULT_CLASSIFICATION_BATCH_SIZE = int(os.getenv("CLASSIFICATION_BATCH_SIZE", "8"))
# Legacy fallback
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

DEFAULT_MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))
DEFAULT_MAX_CALLS_PER_RUN = int(os.getenv("MAX_CALLS_PER_RUN", "0"))  # 0 = unlimited

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "")

TABLE_CALLS = os.getenv("TABLE_CALLS", "CallRailAPI")
TABLE_ANALYSIS = os.getenv("TABLE_ANALYSIS", "CallRailAPI_TranscriptAnalysis")

COL_CALL_ID = os.getenv("COL_CALL_ID", "id")
COL_TRANSCRIPT = os.getenv("COL_TRANSCRIPT", "transcription")

# ============================================================
# STEP 1: REASONING PROMPT (Gemini 2.5 Pro)
# ============================================================

REASONING_SYSTEM_PROMPT = """You are a meticulous veterinary call transcript analyst. Your job is to read raw call transcripts and produce a structured reasoning summary of what happened in each call.

For each call transcript, analyze and summarize:

1. **APPOINTMENT OUTCOME**: Did the caller book an appointment? Look for explicit confirmation (date/time set), explicit decline, or ambiguity.
2. **CLIENT RELATIONSHIP**: Is this a new or existing client? Look for concrete signals: file lookup, pet already on record, references past visits = Existing. Asking about pricing, hours, whether accepting new patients, agent creates new file = New.
3. **SERVICE DISCUSSED**: What veterinary service was discussed? Be specific about the medical context.
4. **REASON NOT BOOKED**: If no appointment was booked (or outcome is unclear), why not? Identify the specific barrier.
5. **NAMES**: Extract hospital name (if spoken), pet name (if mentioned), agent/staff name (if introduced).

## CRITICAL RULES

### Parent-Level vs. Sub-Category (applies to 43% of calls)
When determining the service type, match the EXACT level of specificity the transcript supports:
- Use the PARENT category (e.g., "Urgent Care / Sick Pet") when the transcript describes a general concern WITHOUT enough detail to determine a specific sub-category
- Only use a SUB-CATEGORY when the transcript mentions or implies THE SPECIFIC INTERVENTION or service (e.g., "bloodwork" = lab testing, "dental cleaning" = dental care, "vaccination" = vaccinations). Symptoms ALONE are not enough — there must be evidence of what the clinic will DO
- **When in doubt between parent and sub-category, ALWAYS use the parent. Over-specification is the #1 error pattern.**
- In your reasoning, you MUST explicitly state: (a) the specific intervention/service mentioned in the transcript (e.g., "agent scheduled bloodwork", "caller asked for dental cleaning"), OR (b) "no specific intervention mentioned — using parent category." If you cannot name the intervention, use the parent.
- **NOT specific interventions:** "we'll take a look", "bring them in for an exam", "the doctor will check", "physical examination", "we'll see what's going on." These are general exams — use the PARENT category.
- **Specific interventions (required for sub-category):** "we'll run bloodwork", "we need to do X-rays", "start her on antibiotics", "dental cleaning scheduled", "allergy testing." These name what the clinic will actually DO.
- Example: "my dog is vomiting" → caller books appointment → agent says "we'll take a look" → PARENT "Urgent Care / Sick Pet" (a general exam is not a specific intervention)
- Example: "my dog is vomiting" → agent says "we'll need to run some blood tests to check her kidneys" → SUB "Urgent Care – Diagnosis and Treatment of Illnesses" (bloodwork is a specific intervention)

### Emergency vs. Urgent Care
Classify based on the ACTUAL SERVICE discussed, not the hospital name:
- **Emergency & Critical Care** requires an actual emergency intervention: active stabilization (trauma, poisoning, seizures), overnight hospitalization, or the caller being told to go to an emergency clinic. The call must involve emergency-level treatment.
- **Urgent Care / Sick Pet** applies when a sick pet needs prompt attention but the interaction is routine: advice calls, medication questions, stable-patient triage, or any case where the agent treats the situation as non-critical — even at an emergency hospital.
- A call TO an emergency hospital about a medication question, a stable patient, or general advice = classify by the actual service (Urgent Care, Diagnostic, etc.), NOT Emergency.
- If the caller is directed to an emergency hospital, that's "Emergency & Critical Care – Referred to an Emergency Hospital"
- If the agent explicitly says the pet is "stable" or describes a non-urgent wait time, that is NOT Emergency.

### Dental: Surgical vs. Preventive
- Dental cleanings and extractions = Surgical Services - Dental Care
- Routine dental checkup as part of wellness = Preventive Care

### Preventive Care: Parent vs. Sub-Category
Use the PARENT "Preventive Care" for general wellness visits, new pet checkups, or calls where multiple preventive services are discussed. Only use a sub-category when that specific service is the SOLE AND EXPLICIT purpose of the call:
- "I need to get my dog his shots" (ONLY vaccines discussed) → "Preventive Care – Vaccinations"
- "I want to bring my new puppy in for a checkup" (general wellness, may include vaccines) → PARENT "Preventive Care"
- "I need to schedule an annual exam" (general checkup) → PARENT "Preventive Care"
- When in doubt, use the parent "Preventive Care" — the same parent-over-sub rule applies here

### Preventive Care vs. Diagnostic Services vs. Retail
- **Routine/scheduled bloodwork** = "Preventive Care – Wellness Screening", NOT Diagnostic Services. Examples: "checking on blood results from his annual visit," "pre-op bloodwork before the dental," "routine bloodwork on my senior dog"
- **Symptom-driven bloodwork** = "Diagnostic Services – Lab Testing". Examples: "run blood tests because she's been lethargic," "check her kidney values because she's been vomiting"
- **Key test**: Was the bloodwork part of ROUTINE care (annual, pre-op, wellness)? → Wellness Screening. Was it ordered to INVESTIGATE a symptom or problem? → Diagnostic Lab Testing.
- **Flea/tick/heartworm prevention** (new prevention plan, first-time purchase, discussing prevention options) = "Preventive Care – Parasite Prevention", NOT Retail
- **Refilling an existing prescription** (pet already on a known medication, just needs a refill) = "Retail – Prescriptions"
- When a call is about rescheduling or checking on a preventive-care appointment, classify as Preventive Care (based on the underlying service), not Other

### Dermatology
Use Dermatology when the PRIMARY reason for the call is a skin, coat, ear, or allergy issue:
- **Dermatology – Allergies**: itching/scratching, skin rashes, hot spots, food allergies, environmental allergies, allergy medication (Apoquel, Cytopoint)
- **Dermatology – Ear Infections**: recurring ear infections, ear discharge, head shaking, ear cleaning related to infection
- Use parent "Dermatology" when the skin/ear issue is present but the specific sub-category is unclear
- Do NOT classify as Preventive Care just because the visit may include a general exam — if the stated reason is a skin or ear problem, use Dermatology
- Do NOT classify as Urgent Care unless the skin/ear issue is secondary to a more urgent medical concern

### Price Shopping vs. General Procrastination
- If the caller explicitly asks about prices/costs/fees and then declines = Price Objection (1a)
- If the caller just says "I'll think about it" or "I'll call back" without price discussion = Caller Procrastination (1)

### Client Type
"Existing" means the CALLER has been a client at THIS specific clinic before. Classify based on CONCRETE SIGNALS, not tone or demeanor:
- **Existing signals (need at least one):** agent looks up file/account and FINDS it for this caller, pet already in the system under this caller's name, caller references a past visit AT THIS CLINIC or ongoing medication prescribed here, caller uses a specific doctor's name at this clinic
- **New signals:** caller asks "do you accept new patients?", asks about location/hours/pricing as if unfamiliar, agent asks for phone number to CREATE a new file, agent says "no record found", caller mentions they have a vet elsewhere
- **Edge cases that are NEW:** a caller who has a vet at another clinic but is calling this one for the first time = New. A new owner of a pet that has an existing file from a previous owner = New. Caller has one pet on file but is calling about a brand new pet with no history = context-dependent, lean New if the call is focused on the new pet.
- Casual or friendly tone alone does NOT indicate Existing — require concrete evidence
- "Inconclusive" for client_type should be extremely rare (near 0%)

### Appointment Booked — Yes, No, or Inconclusive
Most calls have a clear Yes or No outcome even when it's not explicitly stated. Use Inconclusive sparingly — only when the outcome is genuinely pending or unknowable from the transcript.

**EVALUATION ORDER (follow these steps in sequence):**
1. FIRST, check if Yes. Did anyone confirm a time, agree to come in, or already have an appointment? → Yes.
2. SECOND, check if No. Did the caller get information and leave, decline, or call only for advice? If the caller made the decision not to book → No.
3. ONLY IF neither applies — the outcome depends on a future action by someone else → Inconclusive.

If you can determine what the CALLER decided, it is not Inconclusive. Inconclusive means the outcome depends on a future event that hasn't happened yet.

**Use Yes when:**
- A specific date/time is confirmed (the clearest case)
- At an emergency/walk-in clinic: agent tells caller to come in and caller agrees or is already there — even without a specific appointment time, this counts as Yes
- Caller already has a confirmed appointment and is calling to adjust it (reschedule, get on cancellation list) — they already have a booking = Yes
- Agent says "we'll see you at [time]" and caller confirms

**Use No when the caller chose not to book (even if they don't say "no"):**
- Caller asks about pricing, gets the info, and hangs up without scheduling
- Caller is told the schedule is full and ends the call
- Caller says "I'll think about it" / "I'll call back" / "let me talk to my partner"
- Caller calls for advice only and never intended to book (medication questions, asking about symptoms)
- Caller explicitly declines to book after hearing information
- **Key test:** If the CALLER decided not to book, it's No — even if they were polite about it
- Caller calls to gather information (services, hours, pricing, medication questions) and ends the call without scheduling — even if friendly and said "thank you." Information-gathering calls where no appointment was discussed are No, NOT Inconclusive
- The call ends naturally after the caller's question was answered, with no mention of scheduling — this is a completed interaction, not a pending one

**Use Inconclusive when the outcome is genuinely pending or unknown:**
- Call goes to voicemail or automated system
- Clinic will call back (e.g., "the doctor will review and we'll get back to you") — outcome depends on a future action by the CLINIC, not the caller
- Inter-clinic consultation where no direct booking occurs
- The call is administrative (checking results, asking about records, updating info) — no appointment was the purpose of the call

### Reason Not Booked
- **Always populate** when appointment_booked is "No" — there is always a reason the caller did not book
- When appointment_booked is "Inconclusive": **use null**. Inconclusive means the outcome is genuinely unclear — there is no specific "reason not booked" to assign. The only exception is when the call is Inconclusive AND there is a clear, explicit barrier (e.g., "we're fully booked so you'll have to call back")
- **Always null** when appointment_booked is "Yes"
- **IMPORTANT: "9. Client/appt query" is ONLY for appointment_booked=No.** If a call is purely administrative (checking results, confirming appointment time, asking about records) and you classified it as Inconclusive, do NOT assign "9. Client/appt query" — use null instead. Category 9 is for when a caller had a medical need but only made an administrative inquiry instead of booking.
- **Decision rules for commonly confused categories:**
  - **"1. Caller Procrastination" vs "1a. Price Objection":** Use 1a when pricing/cost is discussed AT ANY POINT in the call and the caller does not book. If the caller asks "how much does X cost?" and then declines or hangs up, that is ALWAYS 1a — even if they don't explicitly say "that's too expensive." Use 1 only when they say "I'll think about it" / "I'll call back" with NO price discussion anywhere in the call
  - **"2a. Walk-ins not available" vs "2b. Full schedule":** Use 2a when caller wants same-day or walk-in service and is told none available. Use 2b when caller wants any upcoming appointment and the schedule is full for multiple days/weeks ahead
  - **"1c. Getting info for someone else" vs "1. Procrastination":** Use 1c when caller explicitly states they are calling on behalf of someone else, or it is an inter-clinic call (clinic calling another clinic)
  - **"4. Meant to call competitor":** Caller dialed the wrong clinic entirely — they intended to reach a different hospital

### Transcript Quality
- If transcript has fewer than 3 meaningful exchanges, note "very short transcript"
- If transcript appears garbled or mostly redacted ([MEDICAL_PROCESS] tags), note "poor transcript quality"
- These cases often result in Inconclusive outcomes

### When to Use "Other" vs. a Real Category
"Other" is a LAST RESORT. Before using "Other," verify ALL of the following:
- The call is NOT about a sick pet, injury, or medical concern (→ Urgent Care / Sick Pet)
- The call is NOT about scheduling, rescheduling, or checking on any type of appointment (→ classify by the underlying service)
- The call is NOT about medications, food, or prescriptions (→ Retail)
- The call is NOT a missed call or voicemail with no content (→ N/A (missed call))
- The call topic genuinely does not fit ANY existing category (e.g., billing dispute, non-veterinary inquiry)

Common traps — these are NOT "Other":
- Admin/rescheduling calls about a preventive appointment → Preventive Care
- Short or garbled transcripts where the caller appears to be a client → classify by whatever context is available, use parent category
- Voicemail messages left for the clinic → if medical content mentioned, classify by service; if no content, use N/A (missed call)

## EXAMPLES

### Example 1: Parent-Level Urgent Care (No appointment, scheduling issue)
TRANSCRIPT:
Agent: Good morning, Blue Sky Animal Hospital. Caller: Hi there. I'm just wondering, are there any cancellations this afternoon for an appointment? Agent: Unfortunately, we are fully booked. Caller: Okay. Do you happen to have that smart vet phone number handy? Agent: Yep. Yep. Caller: Okay, I'm ready. Yeah. Agent: Are you ready? So it's 705-410-6916. Caller: Okay. All right. 7 oh, 541-06916. Agent: Yep. You got it. Caller: Okay. Thanks so much for your help. Agent: No problem. Caller: Okay, bye. Agent: Bye.

REASONING:
- Appointment: NO — caller asked for same-day appointment, hospital fully booked, given alternative (SmartVet number)
- Client: EXISTING — caller is familiar with the hospital, calls casually, no introductory questions
- Service: General sick pet concern implied (requesting same-day appointment urgently) but NO specific symptoms or condition mentioned — use PARENT level "Urgent Care / Sick Pet"
- Reason not booked: Scheduling issue — no same-day appointments available (walk-ins/same-day not available)
- Hospital: Blue Sky Animal Hospital | Pet: not mentioned | Agent: not introduced

### Example 2: Sub-Category Urgent Care — Diagnosis (Appointment booked)
TRANSCRIPT:
Agent: Good morning, Fisher Glen Animal Hospital. This is Sabrina speaking. How may I help you? Caller: Oh, hi, this is Laura Scott calling. Our cat, Luna is a patient there? She seems to be sort of. Just this morning she woke up, she seems to be in quite a bit of pain. She was vomiting foam. She's yowling. She's breathing very shallow. Agent: Okay, are you able to come in at 9:00 this morning? Caller: Yeah, can you comment nine? Agent: Okay. Caller: Yep, nine is perfect. Agent: Okay, perfect. So, yeah, I would recommend heading 9:00. in for. Caller: Okay. Okay, thanks so much. Appreciate it. Thank you. Bye. Agent: No problem. Bye now.

REASONING:
- Appointment: YES — booked for 9:00 AM same day
- Client: EXISTING — "Luna is a patient there," caller knows the hospital
- Service: Symptoms described (vomiting foam, pain, shallow breathing) AND the agent explicitly schedules a diagnostic appointment to examine the illness — the INTERVENTION is diagnosis/treatment, not just symptoms = SUB-CATEGORY "Urgent Care – Diagnosis and Treatment of Illnesses"
- Reason not booked: N/A (appointment was booked)
- Hospital: Fisher Glen Animal Hospital | Pet: Luna | Agent: Sabrina

### Example 2b: Parent-Level Urgent Care — Symptoms Only (No specific intervention mentioned)
TRANSCRIPT:
Caller: Hi, my dog has been limping for a few days and I'm getting worried. Agent: Oh no, I'm sorry to hear that. Would you like to bring him in? We have an opening tomorrow at 2. Caller: Yeah, that works. Agent: Great, we'll see you then. Caller: Thanks, bye.

REASONING:
- Appointment: YES — booked for tomorrow at 2:00 PM
- Client: EXISTING — no introductory questions, agent doesn't ask to create a file
- Service: Symptom mentioned (limping) but NO specific intervention discussed — the transcript doesn't clarify whether this will be imaging, exam, or other workup. Use PARENT level "Urgent Care / Sick Pet" (NOT a sub-category)
- Reason not booked: N/A (appointment was booked)
- Hospital: not stated | Pet: not named | Agent: not introduced

### Example 3: Emergency & Critical Care (Inconclusive outcome)
TRANSCRIPT:
Agent: Kingston Regional. How can I help you? Caller: Yes, I have a dog. Great peer, golden retriever who ran into a porcupine. Agent: Oh, no. Okay. All right. You can definitely come in to us. They will let you know where [MEDICAL_PROCESS] [MEDICAL_PROCESS] starts at $1020. Caller: Okay. Agent: Okay, so just give us a call when you get here and we will go about today dating him and getting those quills out. Caller: Okay. Agent: Okay? Just give me a call. Okay. Caller: Thank you. Yeah. Agent: All right, take care. Bye.

REASONING:
- Appointment: INCONCLUSIVE — caller told to come in and "call when you get here" but no firm appointment time set, caller just said "okay" without confirming they're coming
- Client: NEW — no file lookup, no familiarity signals, emergency walk-in situation
- Service: Acute trauma (porcupine quills) requiring immediate stabilization at an emergency hospital = "Emergency & Critical Care – Stabilization"
- Reason not booked: null (outcome is inconclusive — no specific barrier to booking, caller may still come. Use null for Inconclusive calls unless there's an explicit barrier.)
- Hospital: Kingston Regional Pet Hospital | Pet: not named | Agent: not introduced

### Example 3b: Administrative Call (Inconclusive — not a booking attempt)
TRANSCRIPT:
Agent: Pickering Animal Hospital. How can I help you? Caller: Hi, I'm just calling to check on the blood test results for my dog Max. We were in last week. Agent: Let me pull up the file... Yes, the results look normal. Dr. Chen will go over them in detail at your next appointment. Caller: Oh great, that's a relief. Thank you so much. Agent: You're welcome! Have a great day. Caller: Bye.

REASONING:
- Appointment: INCONCLUSIVE — this was an administrative call to check test results, not a booking attempt. The caller already has a future appointment. No new appointment was discussed.
- Client: EXISTING — has a file, recent visit, pet on record, knows the clinic
- Service: Follow-up on bloodwork results from a recent visit. Since this was routine/scheduled bloodwork, no specific intervention mentioned = PARENT "Preventive Care"
- Reason not booked: null (administrative call — reason_not_booked should be null when the call was never about booking an appointment)
- Hospital: Pickering Animal Hospital | Pet: Max | Agent: not introduced

### Example 4: Price Shopping — New Client (No appointment)
TRANSCRIPT:
Agent: Good afternoon. McEwen, Frontenac Animal Hospital. Kim speaking. Caller: Hi there. I'm just wondering how much it would cost to bring my dog in just for the vet to, like, check a little thing on. On her leg. I think it might be just a bug bite, but I'm not sure. Agent: Okay, so for an exam, it's about 122 plus X. Caller: Wow. Okay. Thank you so much. Agent: You're welcome. Caller: Okay, bye. Bye. Agent: Bye. Bye.

REASONING:
- Appointment: NO — caller asked about cost, reacted with "Wow" (sticker shock), ended call without booking
- Client: NEW — asking "how much it would cost" indicates unfamiliarity, no file lookup
- Service: Caller wants a general exam/diagnostic check on a leg issue — no specific diagnostic mentioned = PARENT level "Diagnostic Services"
- Reason not booked: Price objection — caller explicitly asked about cost and declined after hearing the price
- Hospital: McEwen-Frontenac Animal Hospital | Pet: not named | Agent: Kim

### Example 5: Inconclusive — Inter-Clinic Consultation (null reason)
TRANSCRIPT:
Agent: Animal hospital associate. How can I help you? Caller: Hi there. I'm just calling from Mountjoy Animal Clinic. I just sent an email with three photos on it. Agent: Okay. Caller: If you could show Dr. V. Bu. This person is just being very insistent that she see them before he seeks attention. Agent: Okay. Caller: So I just wanted her to take a peek. Agent: Okay, sure. I'll let her know. Caller: Okay, thank you. Agent: Okay, bye. Caller: Bye. Bye.

REASONING:
- Appointment: INCONCLUSIVE — this is a clinic-to-clinic call about a patient, not a direct booking attempt; outcome depends on Dr. V's review
- Client: EXISTING — calling between clinics about an existing patient
- Service: Urgent care concern (insistent about being seen) but no specific symptoms described — no specific intervention mentioned, using parent = PARENT level "Urgent Care / Sick Pet"
- Reason not booked: null (inter-clinic consultation call — no booking was attempted, so reason_not_booked is null)
- Hospital: Pickering Animal Hospital (receiving) | Pet: not named | Agent: Leslie (mentioned as associate)

### Example 6: End of Life Care (Appointment booked)
TRANSCRIPT:
Caller: Hello? Agent: Hello. Kingston Regional Pet Hospital. How can I help you? Caller: Hi, It's Richard, Amber, LG's husband with Lulu. She's convulsing pretty bad. We're going to have to bring her back, if possible, for [MEDICAL_PROCESS] [MEDICAL_PROCESS] [MEDICAL_PROCESS] [MEDICAL_PROCESS]. Agent: Yeah, okay, yeah, so just give us a call when you arrive, okay? Caller: Okay. Yep. Agent: Okay. How soon do you think we should expect you? Caller: Well, we're going to be leaving right away here, so we're enamored for you, so probably in the next 20 minutes or so. Agent: Okay, then. All right, we'll see you when you get here. Caller: Okay, thanks. Agent: You're welcome. Caller: Bye. Bye.

REASONING:
- Appointment: YES — caller confirmed coming in within 20 minutes, hospital confirmed "we'll see you when you get here"
- Client: EXISTING — references previous visit ("bring her back"), uses first names, hospital knows the pet
- Service: Despite convulsions (which could suggest emergency), the caller specifically says "bring her back for [redacted]" which given the context (convulsing, returning) strongly implies euthanasia = "End of Life Care – In-Clinic Euthanasia"
- Reason not booked: N/A (appointment was booked)
- Hospital: Kingston Regional Pet Hospital | Pet: Lulu | Agent: not introduced

### Example 7: Retail — Prescriptions (Appointment booked)
TRANSCRIPT:
Agent: Caledonia Vet Clinic. This is Kelly. How may I help you? Caller: Hi, good morning. This is Laura Horry calling. I called yesterday and made an appointment for my dog Henry for Saturday, August 2nd. Agent: Okay. Caller: I just realized he's going to run out of his medication on Thursday afternoon. So is there any chance I could come in earlier? Agent: Yep. Let me just see. So Thursday afternoon. So you're thinking like next week sometime? Caller: Like if I could come in Monday or Tuesday? Sort of. Agent: Yep. Caller: Your latest appointment that you have? Agent: Yep. So let me just go through here. So Tuesday, July 29th, there is a 4 o' clock spot. Would that work? Caller: That'd be perfect. Yes. Thank you. Agent: Okay. Yeah, that's good. And then we'll get Henry what he needs. Caller: Okay. That's great. Thanks so much. Agent: You're welcome. Thank you. Caller: All right. Bye. Bye. Agent: Okay, bye.

REASONING:
- Appointment: YES — rescheduled to Tuesday July 29th at 4:00 PM
- Client: EXISTING — has existing appointment, medication on file, pet known by name
- Service: Primary reason for call is medication refill (running out of medication) = "Retail – Prescriptions"
- Reason not booked: N/A (appointment was booked)
- Hospital: Caledonia Vet Clinic | Pet: Henry | Agent: Kelly

### Example 8: Preventive Care — Vaccinations sub-category, New Client (Appointment booked)
NOTE: This is an example where the sub-category IS correct because the caller's sole stated purpose is "shots" (vaccinations) — no other services discussed.
TRANSCRIPT:
Agent: Thank you for Pet Hospital. Zoe speaking. How can I help you? Caller: Hey, I'm looking to book an appointment for my dog, Alan. A L L, A N. Just to get his shots. Maybe tomorrow if you guys have it. Agent: Yeah, let me see. Could I just grab your phone number so I can open up the file? Caller: Yeah, 7, 7, 8, 9, 1 8, 3, 6, 2 5. Agent: Okay, let's see. Alan, let's see what we got for tomorrow. Yeah, would you prefer morning or afternoon? Caller: Maybe like 12. Agent: Yeah, we can do 12. Caller: Okay, sweet. Thank you. I will see you then. Agent: Perfect. Yeah, sounds good. Have a good one. Caller: Okay, bye. Bye.

REASONING:
- Appointment: YES — booked for tomorrow at 12:00 PM
- Client: NEW — agent needs to grab phone number to open/create file (not already on record)
- Service: Caller explicitly says "to get his shots" = vaccinations = SUB-CATEGORY "Preventive Care – Vaccinations"
- Reason not booked: N/A (appointment was booked)
- Hospital: North Vancouver Pet Hospital | Pet: Allen | Agent: Zoey

## OUTPUT FORMAT

For each call, produce a structured reasoning block like the examples above. Return JSON:

{
  "calls": [
    {
      "call_id": "CAL123",
      "reasoning": "- Appointment: ... \\n- Client: ... \\n- Service: ... \\n- Reason not booked: ... \\n- Hospital: ... | Pet: ... | Agent: ..."
    }
  ]
}

Return JSON ONLY. No commentary outside the JSON.""".strip()

# ============================================================
# STEP 2: CLASSIFICATION PROMPT (Gemini 2.5 Flash)
# ============================================================

CLASSIFICATION_SYSTEM_PROMPT = """You are a veterinary call classifier. You receive structured reasoning summaries about veterinary phone calls and must classify each into the correct bucket values.

Read each reasoning summary carefully and select the EXACT bucket value for each field. Do not invent categories — use only the values listed below.

RULES:
- appointment_booked: Yes = booking confirmed or caller agreed to come in (including emergency walk-ins). No = caller chose not to book OR call was purely informational (got info and left, price objection, "I'll call back", advice-only, medication questions without scheduling). Inconclusive = outcome genuinely pending on a FUTURE action (voicemail, clinic callback pending, inter-clinic consultation). Inconclusive should be rare — default to Yes or No whenever possible. If the caller's question was answered and the call ended, that is No, not Inconclusive.
- client_type: Choose based on concrete familiarity signals described in the reasoning (file lookup, pet on record, past visits = Existing; asks about hours/pricing/new patients = New). "Inconclusive" should be extremely rare for client_type.
- treatment_type: Match the reasoning's service description to the CLOSEST bucket. Respect parent vs. sub-category level as described in the reasoning.
- reason_not_booked: Always populate when appointment_booked is "No". When appointment_booked is "Inconclusive", default to null — only populate if the reasoning explicitly names a clear barrier. Always null when appointment_booked is "Yes".
- stated_hospital_name, stated_patient_name, agent_name: Extract from reasoning. Use null if not mentioned.

Return JSON ONLY.""".strip()

# Strict JSON schema for structured outputs (Step 2)
CLASSIFICATION_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "call_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "call_id": {"type": "string"},
                            "appointment_booked": {
                                "type": "string",
                                "enum": ["Yes", "No", "Inconclusive"]
                            },
                            "client_type": {
                                "type": "string",
                                "enum": ["New", "Existing", "Inconclusive"]
                            },
                            "treatment_type": {
                                "type": "string",
                                "enum": [
                                    "Preventive Care",
                                    "Preventive Care \u2013 Vaccinations",
                                    "Preventive Care \u2013 Parasite Prevention",
                                    "Preventive Care \u2013 Annual Exams",
                                    "Preventive Care \u2013 Wellness Screening (Bloodwork, Urinalysis, Fecals)",
                                    "Urgent Care / Sick Pet",
                                    "Urgent Care \u2013 Diagnosis and Treatment of Illnesses (Vomiting, Diabetes, Infections)",
                                    "Urgent Care \u2013 Chronic Disease Management (Arthritis, Allergies, Thyroid Disease)",
                                    "Urgent Care \u2013 Internal Medicine Workups (Blood Tests, Imaging, Specialist Consults)",
                                    "Surgical Services",
                                    "Surgical Services \u2013 Spays and Neuters",
                                    "Surgical Services \u2013 Soft Tissue Surgeries (Lump Removals, Bladder Stone Removal, Wound Repair)",
                                    "Surgical Services \u2013 Orthopedic Surgeries (ACL Repairs, Fracture Repair \u2014 Sometimes Referred Out)",
                                    "Surgical Services \u2013 Emergency Surgeries (Pyometra, C-Sections, GDV)",
                                    "Surgical Services \u2013 Dental Care (Cleanings, Extractions)",
                                    "Diagnostic Services",
                                    "Diagnostic Services \u2013 X-Rays (Digital Radiography)",
                                    "Diagnostic Services \u2013 Ultrasound",
                                    "Diagnostic Services \u2013 In-House or Reference Lab Testing (Blood, Urine, Fecal, Cytology)",
                                    "Diagnostic Services \u2013 ECG or Blood Pressure Monitoring",
                                    "Emergency & Critical Care",
                                    "Emergency & Critical Care \u2013 Stabilization (Trauma, Poisoning, Seizures)",
                                    "Emergency & Critical Care \u2013 Overnight Hospitalization",
                                    "Emergency & Critical Care \u2013 Fluid Therapy, Oxygen Therapy, Intensive Monitoring",
                                    "Emergency & Critical Care \u2013 Referred to an Emergency Hospital",
                                    "Dermatology",
                                    "Dermatology \u2013 Allergies",
                                    "Dermatology \u2013 Ear Infections",
                                    "Retail",
                                    "Retail \u2013 Food Orders",
                                    "Retail \u2013 Prescriptions",
                                    "End of Life Care",
                                    "End of Life Care \u2013 In-Home Euthanasia",
                                    "End of Life Care \u2013 In-Clinic Euthanasia",
                                    "N/A (missed call)",
                                    "Other"
                                ]
                            },
                            "reason_not_booked": {
                                "type": ["string", "null"],
                                "enum": [
                                    "1. Caller Procrastination",
                                    "1a. Caller Procrastination - Price Objection / Shopping / Request for Quote",
                                    "1b. Caller Procrastination - Need to check with partner",
                                    "1c. Caller Procrastination - Getting information for someone else",
                                    "2. Scheduling Issue",
                                    "2a. Scheduling Issue - Walk ins not available / no same day appt",
                                    "2b. Scheduling Issue - Full schedule",
                                    "2c. Scheduling Issue - Not open / no availability on evenings",
                                    "2d. Scheduling Issue - Not open / no availability on weekends",
                                    "3. Service/treatment not offered",
                                    "3a. Service/treatment not offered - Grooming",
                                    "3b. Service/treatment not offered - Pet Adoption",
                                    "3c. Service/treatment not offered - Exotics",
                                    "3d. Service/treatment not offered - Farm / Large Animals",
                                    "3e. Service/treatment not offered - Birds",
                                    "3f. Service/treatment not offered - Reptiles",
                                    "3g. Service/treatment not offered - Pocket Pets",
                                    "4. Meant to call competitor hospital",
                                    "5. Meant to call low cost / free service provider",
                                    "6. Emergency care not offered",
                                    "7. File Transferred",
                                    "8. Medication/food order",
                                    "9. Client/appt query (non-medical)",
                                    "10. Missed call",
                                    "11. No transcription",
                                    None
                                ]
                            },
                            "stated_hospital_name": {"type": ["string", "null"]},
                            "stated_patient_name": {"type": ["string", "null"]},
                            "agent_name": {"type": ["string", "null"]}
                        },
                        "required": [
                            "call_id",
                            "appointment_booked",
                            "client_type",
                            "treatment_type",
                            "reason_not_booked",
                            "stated_hospital_name",
                            "stated_patient_name",
                            "agent_name"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["calls"],
            "additionalProperties": False
        }
    }
}

# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    handlers = [logging.StreamHandler()]
    if LOG_FILE:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    return logging.getLogger("callrail_analysis")


logger = setup_logging()

# ============================================================
# CLI ARGS
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Analyze CallRail transcripts with two-model OpenAI pipeline.")
    p.add_argument("--version", help="Explicit analysis_version to use (stable nightly).")
    p.add_argument("--auto-version", action="store_true",
                   help="Auto-increment analysis_version using prefix + DB max().")
    p.add_argument("--prefix", help="Prefix used when --auto-version is set.")
    p.add_argument("--max-calls", type=int, help="Max calls to process this run (0=unlimited).")
    p.add_argument("--reasoning-batch-size", type=int, help="Calls per reasoning (Step 1) request.")
    p.add_argument("--classification-batch-size", type=int, help="Calls per classification (Step 2) request.")
    p.add_argument("--batch-size", type=int, help="Legacy: sets both batch sizes.")
    p.add_argument("--max-concurrent", type=int, help="Parallel OpenAI requests.")
    p.add_argument("--reasoning-model", help="Model for Step 1 reasoning (default: gemini-2.5-pro).")
    p.add_argument("--classification-model", help="Model for Step 2 classification (default: gemini-2.5-flash).")
    p.add_argument("--dry-run", action="store_true", help="Fetch and log work, but do not call OpenAI or write to SQL.")
    p.add_argument("--single-model", action="store_true",
                   help="Use single-model mode (classification model only, no reasoning step).")
    p.add_argument("--provider", choices=["gemini", "openai", "anthropic"],
                   help="LLM provider (reads provider-specific keys from .env). "
                        "Overrides LLM_PROVIDER env var.")
    return p.parse_args()

# ============================================================
# LLM CLIENT (OpenAI-compatible API)
# ============================================================

def get_llm_client(provider: Optional[str] = None):
    """Create OpenAI-compatible client for the given provider."""
    provider = provider or DEFAULT_PROVIDER

    if provider in PROVIDERS:
        prov = PROVIDERS[provider]
        api_key = os.getenv(prov["api_key_env"])
        if not api_key:
            # Fallback to legacy LLM_API_KEY
            api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise RuntimeError(f"{prov['api_key_env']} (or LLM_API_KEY) not found in environment or .env")
        base_url = os.getenv(prov["base_url_env"], prov["base_url_default"])
    else:
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_API_KEY not found in environment or .env")
        base_url = DEFAULT_LLM_BASE_URL

    logger.info(f"LLM provider: {provider} | base_url: {base_url}")
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=3)

# ============================================================
# DB CONNECTION & VERSIONING
# ============================================================

def get_db_connection():
    driver = os.getenv("SQLSERVER_DRIVER", "{ODBC Driver 17 for SQL Server}")
    server = os.getenv("SQLSERVER_SERVER")
    database = os.getenv("SQLSERVER_DATABASE")
    uid = os.getenv("SQLSERVER_UID")
    pwd = os.getenv("SQLSERVER_PWD")

    if not all([server, database, uid, pwd]):
        raise RuntimeError(
            "Missing SQLSERVER_* env vars. Need SQLSERVER_SERVER, SQLSERVER_DATABASE, "
            "SQLSERVER_UID, SQLSERVER_PWD."
        )

    conn_str = (
        f"DRIVER={driver};"
        f"SERVER=tcp:{server},1433;"
        f"DATABASE={database};"
        f"UID={uid};PWD={pwd};"
        "Encrypt=yes;TrustServerCertificate=yes;"
    )
    logger.info(f"Connecting to SQL Server at {server}...")
    return pyodbc.connect(conn_str)


def get_next_analysis_version(conn, prefix: str) -> str:
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT MAX(analysis_version)
        FROM {TABLE_ANALYSIS}
        WHERE analysis_version LIKE ?
        """,
        f"{prefix}%",
    )
    row = cur.fetchone()
    last = row[0] if row else None

    if not last:
        return f"{prefix}0001"

    try:
        suffix = last.split("_")[-1]
        num = int(suffix)
        return f"{prefix}{num + 1:04d}"
    except Exception:
        return f"{prefix}0001"

# ============================================================
# FETCH WORK (ONLY IDs NOT IN ANALYSIS TABLE)
# ============================================================

def fetch_unanalyzed_calls(conn, limit: int) -> List[Dict[str, Any]]:
    sql = f"""
    SELECT TOP (?) c.{COL_CALL_ID}, c.{COL_TRANSCRIPT}
    FROM {TABLE_CALLS} c
    LEFT JOIN {TABLE_ANALYSIS} a ON a.{COL_CALL_ID} = c.{COL_CALL_ID}
    WHERE a.{COL_CALL_ID} IS NULL
      AND c.{COL_TRANSCRIPT} IS NOT NULL
      AND LEN(c.{COL_TRANSCRIPT}) > 0
    ORDER BY c.start_time
    """
    cur = conn.cursor()
    cur.execute(sql, (limit,))
    rows = cur.fetchall()
    return [{"id": r[0], "transcript": r[1]} for r in rows]

# ============================================================
# STEP 1: REASONING CALL (Gemini 2.5 Pro)
# ============================================================

def call_reasoning_batch(
    client: OpenAI,
    model: str,
    calls: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Call reasoning model to produce reasoning summaries. Returns {call_id: reasoning_text}."""
    payload = {
        "calls": [{"call_id": c["id"], "transcript": c["transcript"]} for c in calls],
    }
    messages = [
        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload)},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    result = json.loads(resp.choices[0].message.content)

    reasoning_by_id = {}
    for item in result.get("calls", []):
        cid = item.get("call_id")
        reasoning = item.get("reasoning", "")
        if cid:
            reasoning_by_id[str(cid)] = reasoning
    return reasoning_by_id

# ============================================================
# STEP 2: CLASSIFICATION CALL (Gemini 2.5 Flash with strict schema)
# ============================================================

def call_classification_batch(
    client: OpenAI,
    model: str,
    reasoning_items: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Call classification model to classify from reasoning summaries. Returns parsed JSON result."""
    payload = {
        "calls": reasoning_items,
    }
    messages = [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload)},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        response_format=CLASSIFICATION_RESPONSE_SCHEMA,
    )
    return json.loads(resp.choices[0].message.content)

# ============================================================
# LEGACY SINGLE-MODEL CALL (backward compatibility)
# ============================================================

# Single-model prompt combines reasoning + classification
SINGLE_MODEL_SYSTEM_PROMPT = REASONING_SYSTEM_PROMPT.split("## OUTPUT FORMAT")[0].strip() + """

## OUTPUT FORMAT

For each call, return a JSON classification. Return ONLY JSON in this shape:

{
  "calls": [
    {
      "call_id": "CAL123",
      "appointment_booked": "Yes" | "No" | "Inconclusive",
      "client_type": "New" | "Existing" | "Inconclusive",
      "treatment_type": "<exact bucket label>",
      "reason_not_booked": "<exact bucket label or null>",
      "stated_hospital_name": "<string or null>",
      "stated_patient_name": "<string or null>",
      "agent_name": "<string or null>"
    }
  ]
}

TREATMENT TYPE BUCKETS (use EXACTLY one):
Preventive Care | Preventive Care \u2013 Vaccinations | Preventive Care \u2013 Parasite Prevention | Preventive Care \u2013 Annual Exams | Preventive Care \u2013 Wellness Screening (Bloodwork, Urinalysis, Fecals) | Urgent Care / Sick Pet | Urgent Care \u2013 Diagnosis and Treatment of Illnesses (Vomiting, Diabetes, Infections) | Urgent Care \u2013 Chronic Disease Management (Arthritis, Allergies, Thyroid Disease) | Urgent Care \u2013 Internal Medicine Workups (Blood Tests, Imaging, Specialist Consults) | Surgical Services | Surgical Services \u2013 Spays and Neuters | Surgical Services \u2013 Soft Tissue Surgeries (Lump Removals, Bladder Stone Removal, Wound Repair) | Surgical Services \u2013 Orthopedic Surgeries (ACL Repairs, Fracture Repair \u2014 Sometimes Referred Out) | Surgical Services \u2013 Emergency Surgeries (Pyometra, C-Sections, GDV) | Surgical Services \u2013 Dental Care (Cleanings, Extractions) | Diagnostic Services | Diagnostic Services \u2013 X-Rays (Digital Radiography) | Diagnostic Services \u2013 Ultrasound | Diagnostic Services \u2013 In-House or Reference Lab Testing (Blood, Urine, Fecal, Cytology) | Diagnostic Services \u2013 ECG or Blood Pressure Monitoring | Emergency & Critical Care | Emergency & Critical Care \u2013 Stabilization (Trauma, Poisoning, Seizures) | Emergency & Critical Care \u2013 Overnight Hospitalization | Emergency & Critical Care \u2013 Fluid Therapy, Oxygen Therapy, Intensive Monitoring | Emergency & Critical Care \u2013 Referred to an Emergency Hospital | Dermatology | Dermatology \u2013 Allergies | Dermatology \u2013 Ear Infections | Retail | Retail \u2013 Food Orders | Retail \u2013 Prescriptions | End of Life Care | End of Life Care \u2013 In-Home Euthanasia | End of Life Care \u2013 In-Clinic Euthanasia | N/A (missed call) | Other

REASON NOT BOOKED BUCKETS (populate when appointment_booked is "No" or "Inconclusive", null when "Yes"):
1. Caller Procrastination | 1a. Caller Procrastination - Price Objection / Shopping / Request for Quote | 1b. Caller Procrastination - Need to check with partner | 1c. Caller Procrastination - Getting information for someone else | 2. Scheduling Issue | 2a. Scheduling Issue - Walk ins not available / no same day appt | 2b. Scheduling Issue - Full schedule | 2c. Scheduling Issue - Not open / no availability on evenings | 2d. Scheduling Issue - Not open / no availability on weekends | 3. Service/treatment not offered | 3a. Service/treatment not offered - Grooming | 3b. Service/treatment not offered - Pet Adoption | 3c. Service/treatment not offered - Exotics | 3d. Service/treatment not offered - Farm / Large Animals | 3e. Service/treatment not offered - Birds | 3f. Service/treatment not offered - Reptiles | 3g. Service/treatment not offered - Pocket Pets | 4. Meant to call competitor hospital | 5. Meant to call low cost / free service provider | 6. Emergency care not offered | 7. File Transferred | 8. Medication/food order | 9. Client/appt query (non-medical) | 10. Missed call | 11. No transcription

Do NOT add any commentary. Return JSON ONLY.""".strip()


def call_single_model_batch(
    client: OpenAI,
    model: str,
    calls: List[Dict[str, Any]],
    analysis_version: str,
) -> Dict[str, Any]:
    """Legacy single-model call for backward compatibility."""
    payload = {
        "analysis_version": analysis_version,
        "calls": [{"call_id": c["id"], "transcript": c["transcript"]} for c in calls],
    }
    messages = [
        {"role": "system", "content": SINGLE_MODEL_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload)},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        response_format=CLASSIFICATION_RESPONSE_SCHEMA,
    )
    return json.loads(resp.choices[0].message.content)

# ============================================================
# NULL NORMALIZATION + RESULT CONVERSION
# ============================================================

def normalize_null(v):
    if v is None:
        return None
    if isinstance(v, str):
        x = v.strip()
        if x == "" or x.lower() in ("null", "none", "n/a"):
            return None
    return v


def convert_results_to_rows(
    batch_calls: List[Dict[str, Any]],
    result_json: Dict[str, Any],
    reasoning_by_id: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    calls_results = result_json.get("calls", [])
    if not isinstance(calls_results, list):
        calls_results = []

    by_id: Dict[str, Any] = {}
    for item in calls_results:
        cid = item.get("call_id")
        if cid:
            by_id[str(cid)] = item

    rows: List[Dict[str, Any]] = []
    for call in batch_calls:
        cid = call["id"]
        item = by_id.get(str(cid), {})

        appointment_booked = normalize_null(item.get("appointment_booked")) or "Inconclusive"
        client_type = normalize_null(item.get("client_type")) or "Inconclusive"
        treatment_type = normalize_null(item.get("treatment_type")) or "Inconclusive"
        stated_hospital_name = normalize_null(item.get("stated_hospital_name"))
        stated_patient_name = normalize_null(item.get("stated_patient_name"))
        agent_name = normalize_null(item.get("agent_name"))
        reason_not_booked = normalize_null(item.get("reason_not_booked"))

        # Updated rule: populate reason_not_booked for both "No" and "Inconclusive"
        if appointment_booked == "Yes":
            reason_not_booked = None

        row = {
            "id": cid,
            "appointment_booked": appointment_booked,
            "client_type": client_type,
            "treatment_type": treatment_type,
            "stated_hospital_name": stated_hospital_name,
            "stated_patient_name": stated_patient_name,
            "agent_name": agent_name,
            "reason_not_booked": reason_not_booked,
        }

        # Store reasoning for auditing if available
        if reasoning_by_id and str(cid) in reasoning_by_id:
            row["reasoning"] = reasoning_by_id[str(cid)]

        rows.append(row)
    return rows

# ============================================================
# UPSERT
# ============================================================

def upsert_analysis(conn, rows: List[Dict[str, Any]], analysis_version: str) -> None:
    if not rows:
        return

    cur = conn.cursor()
    now = datetime.now(timezone.utc)

    update_sql = f"""
    UPDATE {TABLE_ANALYSIS}
    SET
        stated_hospital_name = ?,
        appointment_booked   = ?,
        client_type          = ?,
        agent_name           = ?,
        reason_not_booked    = ?,
        treatment_type       = ?,
        analyzed_at          = ?,
        analysis_version     = ?,
        stated_patient_name  = ?
    WHERE id = ?
    """

    insert_sql = f"""
    INSERT INTO {TABLE_ANALYSIS} (
        id,
        stated_hospital_name,
        appointment_booked,
        client_type,
        agent_name,
        reason_not_booked,
        treatment_type,
        analyzed_at,
        analysis_version,
        stated_patient_name
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    for r in rows:
        cur.execute(
            update_sql,
            (
                r.get("stated_hospital_name"),
                r["appointment_booked"],
                r["client_type"],
                r.get("agent_name"),
                r.get("reason_not_booked"),
                r["treatment_type"],
                now,
                analysis_version,
                r.get("stated_patient_name"),
                r["id"],
            ),
        )

        if cur.rowcount == 0:
            cur.execute(
                insert_sql,
                (
                    r["id"],
                    r.get("stated_hospital_name"),
                    r["appointment_booked"],
                    r["client_type"],
                    r.get("agent_name"),
                    r.get("reason_not_booked"),
                    r["treatment_type"],
                    now,
                    analysis_version,
                    r.get("stated_patient_name"),
                ),
            )

    conn.commit()

# ============================================================
# TWO-MODEL PIPELINE: PROCESS A CHUNK OF CALLS
# ============================================================

def process_chunk_two_model(
    client: OpenAI,
    reasoning_model: str,
    classification_model: str,
    calls: List[Dict[str, Any]],
    reasoning_batch_size: int,
    classification_batch_size: int,
    max_concurrent: int,
) -> List[Dict[str, Any]]:
    """Process a chunk of calls through the two-model pipeline. Returns rows ready for upsert."""

    # Step 1: Reasoning (GPT-4o) — batch transcripts in smaller groups
    reasoning_batches = [
        calls[i:i + reasoning_batch_size]
        for i in range(0, len(calls), reasoning_batch_size)
    ]

    all_reasoning: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_batch = {
            executor.submit(call_reasoning_batch, client, reasoning_model, batch): batch
            for batch in reasoning_batches
        }
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                reasoning_result = future.result()
                all_reasoning.update(reasoning_result)
                logger.info(f"Step 1 (reasoning): processed {len(reasoning_result)} calls")
            except Exception as e:
                batch_ids = [c["id"] for c in batch]
                logger.error(f"Step 1 reasoning failed for batch {batch_ids[:3]}...: {e}")

    if not all_reasoning:
        logger.error("No reasoning results produced — skipping classification")
        return []

    logger.info(f"Step 1 complete: {len(all_reasoning)} reasoning summaries produced")

    # Step 2: Classification (GPT-4o-mini) — batch reasoning summaries
    reasoning_items = [
        {"call_id": cid, "reasoning": reasoning}
        for cid, reasoning in all_reasoning.items()
    ]
    classification_batches = [
        reasoning_items[i:i + classification_batch_size]
        for i in range(0, len(reasoning_items), classification_batch_size)
    ]

    all_classification_results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_batch = {
            executor.submit(call_classification_batch, client, classification_model, batch): batch
            for batch in classification_batches
        }
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                classification_result = future.result()
                classified_calls = classification_result.get("calls", [])
                all_classification_results.extend(classified_calls)
                logger.info(f"Step 2 (classification): processed {len(classified_calls)} calls")
            except Exception as e:
                batch_ids = [item["call_id"] for item in batch[:3]]
                logger.error(f"Step 2 classification failed for batch {batch_ids}...: {e}")

    logger.info(f"Step 2 complete: {len(all_classification_results)} classifications produced")

    # Convert to rows
    combined_result = {"calls": all_classification_results}
    rows = convert_results_to_rows(calls, combined_result, reasoning_by_id=all_reasoning)
    return rows

# ============================================================
# MAIN PIPELINE
# ============================================================

def process_backlog(
    analysis_version: str,
    reasoning_model: str,
    classification_model: str,
    reasoning_batch_size: int,
    classification_batch_size: int,
    max_concurrent: int,
    max_calls_per_run: int,
    dry_run: bool,
    single_model: bool,
    provider: str = "gemini",
):
    conn = get_db_connection()
    client = None if dry_run else get_llm_client(provider)

    processed = 0
    started = time.time()

    try:
        while True:
            if max_calls_per_run and processed >= max_calls_per_run:
                logger.info(f"Reached max_calls_per_run={max_calls_per_run}")
                break

            # Fetch a bigger chunk so the thread pool stays busy
            batch_size = reasoning_batch_size if not single_model else classification_batch_size
            to_fetch = batch_size * max_concurrent * 10
            if max_calls_per_run:
                remaining = max_calls_per_run - processed
                if remaining <= 0:
                    break
                to_fetch = min(to_fetch, remaining)

            calls = fetch_unanalyzed_calls(conn, to_fetch)
            if not calls:
                logger.info("No more calls to process.")
                break

            logger.info(f"Fetched {len(calls)} unanalyzed calls (analysis_version={analysis_version})")

            if dry_run:
                sample_ids = [c["id"] for c in calls[: min(10, len(calls))]]
                logger.info(f"[DRY RUN] Would process {len(calls)} calls. Sample IDs: {sample_ids}")
                break

            if single_model:
                # Legacy single-model path
                batches = [calls[i:i + classification_batch_size]
                           for i in range(0, len(calls), classification_batch_size)]

                with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                    future_to_batch = {
                        executor.submit(
                            call_single_model_batch, client, classification_model,
                            batch, analysis_version
                        ): batch
                        for batch in batches
                    }
                    for future in as_completed(future_to_batch):
                        batch = future_to_batch[future]
                        try:
                            result_json = future.result()
                        except Exception as e:
                            logger.error(f"OpenAI call failed for batch (skipping {len(batch)} calls): {e}")
                            continue
                        rows = convert_results_to_rows(batch, result_json)
                        upsert_analysis(conn, rows, analysis_version)
                        processed += len(rows)
                        logger.info(f"Processed total={processed} calls")
            else:
                # Two-model pipeline
                rows = process_chunk_two_model(
                    client=client,
                    reasoning_model=reasoning_model,
                    classification_model=classification_model,
                    calls=calls,
                    reasoning_batch_size=reasoning_batch_size,
                    classification_batch_size=classification_batch_size,
                    max_concurrent=max_concurrent,
                )
                upsert_analysis(conn, rows, analysis_version)
                processed += len(rows)
                logger.info(f"Processed total={processed} calls")

    finally:
        conn.close()
        elapsed = time.time() - started
        logger.info(f"DONE. processed={processed}, elapsed_sec={elapsed:.1f}")


def main():
    args = parse_args()

    version = args.version
    auto_version = bool(args.auto_version)
    prefix = args.prefix or DEFAULT_ANALYSIS_PREFIX
    single_model = bool(args.single_model)

    # Resolve provider and models
    provider = args.provider or DEFAULT_PROVIDER
    prov = PROVIDERS.get(provider, {})

    reasoning_model = (args.reasoning_model
                       or DEFAULT_REASONING_MODEL
                       or prov.get("reasoning_model_default", "gemini-2.5-pro"))
    classification_model = (args.classification_model
                            or DEFAULT_CLASSIFICATION_MODEL
                            or prov.get("classification_model_default", "gemini-2.5-flash"))

    # Batch sizes: specific flags > legacy --batch-size > env defaults
    legacy_batch = args.batch_size or DEFAULT_BATCH_SIZE
    reasoning_batch_size = args.reasoning_batch_size or DEFAULT_REASONING_BATCH_SIZE
    classification_batch_size = args.classification_batch_size or DEFAULT_CLASSIFICATION_BATCH_SIZE

    if args.batch_size and not args.reasoning_batch_size:
        reasoning_batch_size = args.batch_size
    if args.batch_size and not args.classification_batch_size:
        classification_batch_size = args.batch_size

    max_concurrent = args.max_concurrent or DEFAULT_MAX_CONCURRENT_REQUESTS
    max_calls = args.max_calls if args.max_calls is not None else DEFAULT_MAX_CALLS_PER_RUN
    dry_run = bool(args.dry_run)

    conn = None
    try:
        if not version and auto_version:
            conn = get_db_connection()
            version = get_next_analysis_version(conn, prefix)
            logger.info(f"Auto-generated analysis_version = {version}")
        elif not version:
            version = DEFAULT_ANALYSIS_VERSION
            logger.info(f"Using pinned analysis_version = {version}")
        else:
            logger.info(f"Using explicit analysis_version = {version}")
    finally:
        if conn:
            conn.close()

    mode = "single-model" if single_model else "two-model"
    logger.info(
        f"Starting transcript analysis | mode={mode} | version={version} | "
        f"reasoning_model={reasoning_model} | classification_model={classification_model} | "
        f"reasoning_batch={reasoning_batch_size} | classification_batch={classification_batch_size} | "
        f"max_concurrent={max_concurrent} | max_calls={max_calls} | dry_run={dry_run}"
    )

    process_backlog(
        analysis_version=version,
        reasoning_model=reasoning_model,
        classification_model=classification_model,
        reasoning_batch_size=reasoning_batch_size,
        classification_batch_size=classification_batch_size,
        max_concurrent=max_concurrent,
        max_calls_per_run=max_calls,
        dry_run=dry_run,
        single_model=single_model,
        provider=provider,
    )


if __name__ == "__main__":
    main()
