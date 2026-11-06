import os
import re
import json
import random
import time
import pandas as pd
import mock_llm

# Path configurations aligned with your capstone structure
RAW_DIR = os.path.join("data", "raw")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Processing parameters
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05

print("--- PART C: LLM COPILOT INITIALIZED ---")
print(f"Reading raw data from: {RAW_DIR}")
print(f"Outputs will be saved to: {OUTPUT_DIR}")
# =====================================================================
# TASK 1: Input Guardrails
# =====================================================================
def check_input_guardrails(text):
    """
    Checks ticket content text.
    Returns status: 'ok', 'skipped_empty', or 'flagged_long'
    """
    # Guardrail 1: Check for empty text
    if not text or str(text).strip() == "":
        return "skipped_empty"
    
    # Guardrail 2: Check for excessively long input tokens
    tokens = mock_llm.est_tokens(str(text))
    if tokens > MAX_INPUT_TOKENS:
        return "flagged_long"
        
    return "ok"

print("Task 1 Input Guardrails successfully implemented.")
# =====================================================================
# TASK 2: Redact Personal Information (PII)
# =====================================================================
def redact_pii(text):
    """
    Applies email, phone, and order ID redactions in exact specified order.
    """
    if not text:
        return "", 0, 0, 0, 0
        
    text_str = str(text)
    
    # 1. Email Redaction
    text_str, email_count = re.subn(r'[\w.+-]+@[\w-]+\.[\w.]+', '[EMAIL]', text_str)
    # 2. Phone Redaction
    text_str, phone_count = re.subn(r'\b[6-9]\d{9}\b', '[PHONE]', text_str)
    # 3. Order ID Redaction
    text_str, order_count = re.subn(r'ORD-\d{6}', '[ORDER_ID]', text_str)
    
    total_subs = email_count + phone_count + order_count
    return text_str, email_count, phone_count, order_count, total_subs

# Assertion check to prove nothing is left behind
def assert_no_pii_remaining(text):
    text_str = str(text)
    assert not re.search(r'[\w.+-]+@[\w-]+\.[\w.]+', text_str), "Email pattern leak detected!"
    assert not re.search(r'\b[6-9]\d{9}\b', text_str), "Phone pattern leak detected!"
    assert not re.search(r'ORD-\d{6}', text_str), "Order ID pattern leak detected!"

print("Task 2 PII Redaction and assertions successfully implemented.")
# =====================================================================
# TASK 3: Zero-Shot and Few-Shot Prompts
# =====================================================================
ZERO_SHOT_SYSTEM_PROMPT = """CLASSIFY
You are a support ticket classifier for Instamart.
Allowed categories: billing, delivery, technical, fraud.
You must reply with a JSON-only object of the form:
{"category": "...", "reason": "..."}
"""

FEW_SHOT_SYSTEM_PROMPT = """CLASSIFY
You are a support ticket classifier for Instamart.
Allowed categories: billing, delivery, technical, fraud.
You must reply with a JSON-only object of the form:
{"category": "...", "reason": "..."}

EXAMPLES:
- Text: "Someone hacked my account and used my wallet balance. This is fraud." -> {"category": "fraud", "reason": "unauthorized account use"}
- Text: "My order ORD-123456 has not arrived and the rider is not picking up." -> {"category": "delivery", "reason": "late or missing parcel"}
"""

print("Task 3 Prompts successfully declared.")
# =====================================================================
# TASK 4: Resilient Model Calls with Retries
# =====================================================================
def call_llm_with_retry(messages, system_prompt):
    """
    Retries up to 3 times on TransientLLMError with exponential backoff & jitter.
    """
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    
    for attempt in range(4): # 0, 1, 2, 3 (max 3 retries)
        try:
            response = mock_llm.chat(full_messages)
            return response
        except mock_llm.TransientLLMError as e:
            if attempt == 3:
                raise e # Exhausted retries
            
            # Wait calculation: BASE_DELAY * 2**n * random.uniform(0.9, 1.15)
            wait = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)
            print(f"[RETRY LOG] Attempt {attempt} failed ({e}). Waiting {wait:.4f}s...")
            time.sleep(wait)

print("Task 4 Resilient LLM retry logic successfully implemented.")
# =====================================================================
# TASK 5: Output Validation
# =====================================================================
ALLOWED_CATS = {"billing", "delivery", "technical", "fraud"}

def validate_and_parse_response(response_content):
    """
    Validates model output into classified_json, classified_keyword, or flagged_invalid_output.
    """
    content_str = response_content or ""
    
    # 1. Try parsing JSON
    try:
        data = json.loads(content_str)
        cat = data.get("category")
        if cat in ALLOWED_CATS:
            return "classified_json", cat
    except json.JSONDecodeError:
        pass
        
    # 2. Fallback: extract lowercase words via regex
    words = re.findall(r"[a-z]+", content_str.lower())
    found_cats = [w for w in words if w in ALLOWED_CATS]
    
    if len(found_cats) == 1:
        return "classified_keyword", found_cats[0]
        
    return "flagged_invalid_output", None

print("Task 5 Output Validation successfully implemented.")
# =====================================================================
# TASK 6 & 7: Run Full Pipeline, Budget Demo, and Final Routing
# =====================================================================
def run_triage_pipeline(few_shot=False, token_budget=100000):
    mock_llm.reset()
    tickets_df = pd.read_csv(os.path.join(RAW_DIR, "tickets.csv"), keep_default_na=False)
    
    # Points to data/clean/ where Part B successfully wrote the segments
    SEGMENTS_DIR = "data/clean"
    segments_df = pd.read_csv(os.path.join(SEGMENTS_DIR, "customer_segments.csv"))
    premium_customers = set(segments_df[segments_df["is_premium"] == 1]["customer_id"])
    
    system_prompt = FEW_SHOT_SYSTEM_PROMPT if few_shot else ZERO_SHOT_SYSTEM_PROMPT
    
    results = []
    running_tokens = 0
    total_redactions = {"email": 0, "phone": 0, "order": 0, "total": 0}
    
    for idx, row in tickets_df.iterrows():
        ticket_id = row["ticket_id"]
        customer_id = row["customer_id"]
        raw_text = row["text"]
        
        # 1. Input Guardrails
        guard_status = check_input_guardrails(raw_text)
        if guard_status != "ok":
            results.append({
                "ticket_id": ticket_id, "customer_id": customer_id, "status": guard_status,
                "category": "", "segment_name": "", "priority": "NORMAL", "route": "human_review", "tokens_used": 0
            })
            continue
            
        # 2. PII Redaction
        redacted_text, e_c, p_c, o_c, tot_c = redact_pii(raw_text)
        total_redactions["email"] += e_c
        total_redactions["phone"] += p_c
        total_redactions["order"] += o_c
        total_redactions["total"] += tot_c
        assert_no_pii_remaining(redacted_text)
        
        # 3. Token Budget Check
        if running_tokens >= token_budget:
            results.append({
                "ticket_id": ticket_id, "customer_id": customer_id, "status": "budget_exhausted",
                "category": "", "segment_name": "", "priority": "NORMAL", "route": "human_review", "tokens_used": 0
            })
            continue
            
        # 4. LLM Call with Retry
        messages = [{"role": "user", "content": redacted_text}]
        try:
            response = call_llm_with_retry(messages, system_prompt)
            tokens_used = response.usage.total_tokens
            running_tokens += tokens_used
            
            # 5. Output Validation
            val_status, category = validate_and_parse_response(response.choices[0].message.content)
            
            # Ensure keyword extract list collapses back into string single item
            if isinstance(category, list):
                category = category[0]
            
        except Exception:
            tokens_used = 0
            val_status = "flagged_invalid_output"
            category = None
            
        # 6. Routing Rules
        segment_name = "Premium VIPs" if customer_id in premium_customers else "Standard"
        is_prem = customer_id in premium_customers
        
        if val_status in ["flagged_invalid_output"]:
            route, priority = "human_review", "NORMAL"
            category = ""
            segment_name = ""
        elif category == "fraud":
            route, priority = "security_team", "HIGH"
        elif is_prem:
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"
            
        results.append({
            "ticket_id": ticket_id, "customer_id": customer_id, "status": val_status,
            "category": category if val_status != "flagged_invalid_output" else "",
            "segment_name": segment_name if val_status != "flagged_invalid_output" else "",
            "priority": priority, "route": route, "tokens_used": tokens_used
        })
        
    return pd.DataFrame(results), total_redactions

# Execute Budget Demo
budget_df, _ = run_triage_pipeline(few_shot=False, token_budget=6000)
budget_df.to_csv(os.path.join(OUTPUT_DIR, "triage_budget_demo.csv"), index=False)
exhausted_count = (budget_df["status"] == "budget_exhausted").sum()
print(f"Budget Demo Completed. Exhausted tickets count: {exhausted_count}")

# Execute Final Few-Shot Triage Run
final_df, redaction_totals = run_triage_pipeline(few_shot=True, token_budget=100000)
final_df.to_csv(os.path.join(OUTPUT_DIR, "triage_results.csv"), index=False)
print("\n--- FINAL TRIAGE COMPLETE ---")
print(f"Triage Results Saved. Total PII Redacted: {redaction_totals['total']} items.")
print(f"Emails: {redaction_totals['email']} | Phones: {redaction_totals['phone']} | Orders: {redaction_totals['order']}")
