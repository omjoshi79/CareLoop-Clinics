# Part 4 — GenAI: turn call-centre notes into structured reasons, then recommend

# Task 4.1 — One validated call (CRAFT prompt + JSON schema)

!pip install -q google-genai pydantic

import json
import time
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

# Enter API key at runtime (NEVER hardcode it; clear cell output before saving)
API_KEY = input("Paste your Gemini API key: ")
client = genai.Client(api_key=API_KEY)

# Use your Session 22.1 model name, or current Flash model in AI Studio
MODEL_NAME = "gemini-3.5-flash-lite"

ALLOWED_ROOT_CAUSES = [
    "forgot_or_no_reminder",
    "distance_or_transport",
    "cost_or_fee",
    "illness_or_emergency",
    "work_or_schedule",
    "other",
]

SYSTEM_TEXT = (
    "You are a healthcare operations analyst at CareLoop Clinics in India. "
    "You read short call-centre follow-up notes (sometimes in English or Hinglish) "
    "about missed outpatient appointments and classify the primary root cause accurately. "
    "Never invent details that are not in the note."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "root_cause": {"type": "string"},
        "confidence": {"type": "number"},
        "summary": {"type": "string"},
        "recommended_action": {"type": "string"},
    },
    "required": ["root_cause", "confidence", "summary", "recommended_action"],
}

api_call_count = 0

def build_craft_prompt(note_text: str) -> str:
    return f"""
Context: CareLoop Clinics' call-centre team phones patients who missed an outpatient appointment and logs a short note (which may be in English or Hinglish).
Role: Act as a clinic operations analyst extracting structured no-show reasons.
Action: Read the note below and classify the primary root_cause into EXACTLY ONE of these allowed values:
- forgot_or_no_reminder
- distance_or_transport
- cost_or_fee
- illness_or_emergency
- work_or_schedule
- other

If the note is ambiguous, unclear, or does not clearly fit the first five categories, answer "other" rather than guessing.
Tone: Objective, concise, and operational.
Format: Return valid JSON with keys root_cause, confidence (0.0 to 1.0), summary (at most 25 words), and recommended_action (concrete next step for the clinic).

Call-centre note: "{note_text}"
""".strip()

def analyse_note(note_text: str, custom_prompt: str | None = None, return_raw: bool = False):
    global api_call_count
    prompt = custom_prompt if custom_prompt is not None else build_craft_prompt(note_text)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_TEXT,
            temperature=0,
            max_output_tokens=1000,
            response_mime_type="application/json",
            response_schema=SCHEMA,
            # If your model rejects thinking_config, delete the next line
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        ),
    )
    api_call_count += 1
    raw_text = response.text
    parsed = json.loads(raw_text)
    return (parsed, raw_text) if return_raw else parsed

# Load notes and test on N001
notes_df = pd.read_csv("/followup_notes.csv")
n001_text = notes_df.loc[notes_df["note_id"] == "N001", "note_text"].iloc[0]

n001_result = analyse_note(n001_text)
print("Note N001 text:", n001_text)
print("Parsed dictionary:", json.dumps(n001_result, indent=2))

# Task 4.2 — Validate and process the first 10 notes

class FollowupInsight(BaseModel):
    root_cause: str = Field(min_length=3)
    confidence: float = Field(ge=0, le=1)
    summary: str = Field(min_length=5)
    recommended_action: str = Field(min_length=5)

first_10 = notes_df.head(10).copy()
results = []

for idx, row in first_10.iterrows():
    note_id = row["note_id"]
    appt_id = row["appointment_id"]
    note_text = row["note_text"]

    # Input validation: check NaN before .strip()
    if pd.isna(note_text) or str(note_text).strip() == "":
        results.append({
            "note_id": note_id,
            "appointment_id": appt_id,
            "root_cause": "other",
            "confidence": 0.0,
            "summary": "No note recorded",
            "recommended_action": "Re-call patient",
            "validation_status": "skipped_blank",
            "raw_response": "",
        })
        print(f"{note_id}: skipped_blank (no API call)")
        continue

    raw_text = ""
    data = {}
    status = "invalid"

    try:
        data, raw_text = analyse_note(str(note_text), return_raw=True)
        # Validate schema with Pydantic + allowed root_cause list
        insight = FollowupInsight(**data)
        if insight.root_cause in ALLOWED_ROOT_CAUSES:
            status = "valid"
        else:
            status = "invalid"
    except Exception as e:
        print(f"{note_id}: validation/API error -> {e}")
        status = "invalid"

    # Keep every reply as-is; never repair missing fields
    results.append({
        "note_id": note_id,
        "appointment_id": appt_id,
        "root_cause": data.get("root_cause", ""),
        "confidence": data.get("confidence", ""),
        "summary": data.get("summary", ""),
        "recommended_action": data.get("recommended_action", ""),
        "validation_status": status,
        "raw_response": raw_text,
    })
    print(f"{note_id}: {data.get('root_cause', '')} ({status})")

    # Pause to respect free-tier rate limit
    time.sleep(13)

analysed_df = pd.DataFrame(results, columns=[
    "note_id", "appointment_id", "root_cause", "confidence",
    "summary", "recommended_action", "validation_status", "raw_response"
])
analysed_df.to_csv("notes_analysed.csv", index=False)
print("\nSaved notes_analysed.csv with shape:", analysed_df.shape)
print(analysed_df[["note_id", "appointment_id", "root_cause", "confidence", "validation_status"]])

# Task 4.3 — Break it on purpose & prompt comparison (V1 vs V2)

# (a) Deliberately bad replies to test Pydantic ValidationError
bad_confidence = {
    "root_cause": "forgot_or_no_reminder",
    "confidence": 1.7,
    "summary": "Patient forgot the appointment.",
    "recommended_action": "Send SMS reminder.",
}

missing_action = {
    "root_cause": "cost_or_fee",
    "confidence": 0.9,
    "summary": "Patient said consultation fee was too high.",
}

print("--- (a) Pydantic ValidationError Tests ---")
for label, payload in [("confidence = 1.7", bad_confidence), ("missing recommended_action", missing_action)]:
    try:
        FollowupInsight(**payload)
    except ValidationError as e:
        print(f"\nCaught expected error for [{label}]:\n{e}")

# (b) Compare V1 (minimal prompt) vs V2 (CRAFT prompt) on N007 and N009
# Reuses Task 4.2 V2 results to stay well within the free-tier budget (total ~12 calls)
compare_rows = []
for nid in ["N007", "N009"]:
    text = notes_df.loc[notes_df["note_id"] == nid, "note_text"].iloc[0]

    # V1 call
    v1_prompt = f'Classify this note: "{text}"'
    v1_data = analyse_note(text, custom_prompt=v1_prompt)
    v1_rc = v1_data.get("root_cause", "")
    compare_rows.append({
        "note_id": nid,
        "prompt": "V1 (minimal)",
        "root_cause": v1_rc,
        "in_allowed_list": v1_rc in ALLOWED_ROOT_CAUSES,
    })
    time.sleep(13)

    # V2 result (already computed with CRAFT prompt at temperature=0 in Task 4.2)
    v2_rc = analysed_df.loc[analysed_df["note_id"] == nid, "root_cause"].iloc[0]
    compare_rows.append({
        "note_id": nid,
        "prompt": "V2 (CRAFT)",
        "root_cause": v2_rc,
        "in_allowed_list": v2_rc in ALLOWED_ROOT_CAUSES,
    })

compare_df = pd.DataFrame(compare_rows)
print("\n--- (b) Prompt Comparison: V1 vs V2 ---")
print(compare_df.to_string(index=False))
print(
    "\nDeployment choice: Deploy V2 (CRAFT) because it reliably maps Hinglish (N007) and "
    "ambiguous notes (N009) into the allowed taxonomy instead of inventing ad-hoc labels."
)

# (c) Total API calls used across Tasks 4.1–4.3
print(f"\n(c) Total Gemini API calls used across Tasks 4.1–4.3: {api_call_count} (limit < 25)")

# Task 4.4 — Merge & cross-tabulate (root_cause vs reminder_sent)

appts_clean = pd.read_csv("appointments_clean.csv")
notes_analysed = pd.read_csv("notes_analysed.csv")

merged = notes_analysed.merge(appts_clean, on="appointment_id", how="inner")
print("Merged shape:", merged.shape)

ct = pd.crosstab(merged["root_cause"], merged["reminder_sent"], margins=True)
print("\n--- Cross-Tab: root_cause vs reminder_sent ---")
print(ct)
