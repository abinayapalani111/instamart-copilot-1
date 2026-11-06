import os
import glob
import re
import json
import time
import sqlite3
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

# Paths & Setup
POLICIES_DIR = os.path.join("data", "policies")
RAW_DIR = os.path.join("data", "raw")
OUTPUT_DIR = "outputs"
STATE_DIR = "state"
DB_PATH = os.path.join("data", "instamart.db")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)
# =====================================================================
# TASKS 1 to 3: Document Chunking, Retrieval & RAG Assistant
# =====================================================================
def chunk_text(text, filename, chunk_size=40, overlap=10):
    words = text.split()
    chunks = []
    start = 0
    step = chunk_size - overlap # 30 words per step
    
    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        chunk_str = " ".join(chunk_words)
        chunks.append({"source": filename, "text": chunk_str})
        
        if start + chunk_size >= len(words):
            break
        start += step
        
    return chunks

def build_vector_index(overlap=10):
    all_chunks = []
    policy_files = sorted(glob.glob(os.path.join(POLICIES_DIR, "*.txt")))
    
    for filepath in policy_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = chunk_text(content, filename, chunk_size=40, overlap=overlap)
        all_chunks.extend(chunks)
        
    texts = [c["text"] for c in all_chunks]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    return all_chunks, vectorizer, tfidf_matrix

def retrieve(query, all_chunks, vectorizer, tfidf_matrix, k=3):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:k]
    
    results = []
    for idx in top_indices:
        results.append({
            "source": all_chunks[idx]["source"],
            "score": float(scores[idx]),
            "text": all_chunks[idx]["text"]
        })
    return results

def answer_policy_question(question, all_chunks, vectorizer, tfidf_matrix):
    top_chunks = retrieve(question, all_chunks, vectorizer, tfidf_matrix, k=3)
    
    context_str = "\n".join([f"[{c['source']}] {c['text']}" for c in top_chunks])
    
    system_msg = {
        "role": "system",
        "content": "POLICY ASSISTANT: You answer user questions strictly based on the provided context. If the information is not found in the context, reply that you could not find the information in the policy documents."
    }
    
    user_content = f"CONTEXT:\n{context_str}\nQUESTION:\n{question}"
    user_msg = {"role": "user", "content": user_content}
    
    response = mock_llm.chat([system_msg, user_msg])
    ans_text = response.choices[0].message.content
    sources_used = [c["source"] for c in top_chunks]
    
    return ans_text, sources_used

print("Tasks 1, 2, and 3: Chunking engines, retrieval vectors, and RAG frameworks successfully implemented.")
# =====================================================================
# TASK 4: Policy Retrieval & Grounding Evaluation
# =====================================================================
def evaluate_rag():
    eval_df = pd.read_csv(os.path.join(RAW_DIR, "policy_eval.csv"))
    
    # Evaluate Overlap 10
    chunks_10, vec_10, mat_10 = build_vector_index(overlap=10)
    
    doc_matches, fact_matches, fallback_matches = 0, 0, 0
    
    for _, row in eval_df.iterrows():
        q, exp_doc, exp_fact = row["question"], row["expected_doc"], row["expected_fact"]
        ans, sources = answer_policy_question(q, chunks_10, vec_10, mat_10)
        
        if exp_doc == "NONE":
            if ans.startswith("I could not find") or "could not find" in ans.lower():
                fallback_matches += 1
        else:
            if exp_doc in sources:
                doc_matches += 1
            if str(exp_fact).lower() in ans.lower():
                fact_matches += 1

    print("\n--- TASK 4 RAG EVALUATION (OVERLAP = 10) ---")
    print(f"Expected Doc Retrieved: {doc_matches}/12")
    print(f"Expected Fact Matched: {fact_matches}/12")
    print(f"Fallback Correctly Triggered: {fallback_matches}/2")
    
    # Evaluate Overlap 0
    chunks_0, vec_0, mat_0 = build_vector_index(overlap=0)
    failed_questions = []
    
    for _, row in eval_df.iterrows():
        if row["expected_doc"] == "NONE":
            continue
        q, exp_doc, exp_fact = row["question"], row["expected_doc"], row["expected_fact"]
        ans, sources = answer_policy_question(q, chunks_0, vec_0, mat_0)
        
        if str(exp_fact).lower() not in ans.lower():
            failed_questions.append(q)
            
    print("\n--- TASK 4 RAG EVALUATION (OVERLAP = 0) ---")
    print(f"Failed Questions with 0 Overlap: {len(failed_questions)}")
    if failed_questions:
        print(f"Example Failed Question: '{failed_questions[0]}'")

# Add execution trigger for evaluation block
if __name__ == "__main__":
    evaluate_rag()
# =====================================================================
# TASK 5: Define Refund Tools & Tool Map
# =====================================================================
def tool_get_order(order_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return {"error": "order not found"}
        
    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]) if row[2] is not None else 0.0,
        "delivery_minutes": int(row[3]) if row[3] is not None and row[3] != "" else None
    }

def tool_search_policy(query):
    all_chunks, vectorizer, tfidf_matrix = build_vector_index(overlap=10)
    results = retrieve(query, all_chunks, vectorizer, tfidf_matrix, k=3)
    return {"results": results}

def tool_calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None or delivery_minutes <= 45:
        return {"refund_inr": 0.0}
    return {"refund_inr": float(order_amount)}

def tool_issue_refund(order_id, refund_inr):
    ledger_entry = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "tool": "issue_refund",
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }
    with open(os.path.join(OUTPUT_DIR, "refund_ledger.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    return {"refund_inr": float(refund_inr)}

def tool_escalate_to_human(order_id, refund_inr):
    ledger_entry = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "tool": "escalate_to_human",
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }
    with open(os.path.join(OUTPUT_DIR, "refund_ledger.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    return {"refund_inr": float(refund_inr)}

TOOL_MAP = {
    "get_order": tool_get_order,
    "search_policy": tool_search_policy,
    "calculate_refund": tool_calculate_refund,
    "issue_refund": tool_issue_refund,
    "escalate_to_human": tool_escalate_to_human
}

print("Task 5 Refund execution tools and mapping schema successfully initialized.")
# =====================================================================
# TASK 6 to 8: ReAct Loop, Safety Limits, and Resumption
# =====================================================================
SYSTEM_AGENT_PROMPT = "AGENT: You are an automated Instamart customer support agent handling refund requests using tools."

def save_checkpoint(run_id, scenario_id, messages, tokens_used, step, tools_executed):
    state_file = os.path.join(STATE_DIR, f"{run_id}.json")
    state_data = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step": step,
        "tools_executed": tools_executed
    }
    with open(state_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(state_data, indent=2))

def run_agent(user_message, scenario_id, run_id=None, max_iterations=6, token_limit=6000, time_limit_seconds=10, simulate_crash_at_step=None):
    if not run_id:
        run_id = f"run_{scenario_id}_{int(time.time()*1000)}"
        
    start_time = time.monotonic()
    tokens_used = 0
    llm_calls = 0
    steps = []
    tools_executed = []
    
    messages = [
        {"role": "system", "content": SYSTEM_AGENT_PROMPT},
        {"role": "user", "content": user_message}
    ]
    
    for iteration in range(1, max_iterations + 1):
        # 1. Timeout Check
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            return {
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK: Execution timed out. Handing over to human support.",
                "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used
            }
            
        # 2. Call LLM
        response = mock_llm.chat(messages)
        llm_calls += 1
        call_tokens = response.usage.total_tokens
        
        # 3. Token Limit Check
        if (tokens_used + call_tokens) > token_limit:
            return {
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK: Token budget limit reached. Handing over to human support.",
                "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used
            }
            
        tokens_used += call_tokens
        choice = response.choices[0].message
        
        # 4. Final Answer Check (No tool call)
        if not choice.tool_calls:
            final_content = choice.content or ""
            status = final_content.split(":")[0].strip() if ":" in final_content else "DENIED"
            return {
                "status": status,
                "final_answer": final_content,
                "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used
            }
            
        # 5. Execute Tool
        tool_call = choice.tool_calls[0]
        tool_name = tool_call.name
        tool_args = tool_call.arguments
        
        tool_func = TOOL_MAP.get(tool_name)
        tool_result = tool_func(**tool_args)
        
        steps.append({
            "tool": tool_name,
            "args": tool_args,
            "observation": tool_result
        })
        tools_executed.append(tool_name)
        
        # Append assistant & tool messages to conversation history
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tool_call.id, "name": tool_name, "arguments": tool_args}]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_name,
            "content": json.dumps(tool_result)
        })
        
        # Save Checkpoint
        save_checkpoint(run_id, scenario_id, messages, tokens_used, iteration, tools_executed)
        
        # Optional Simulated Crash
        if simulate_crash_at_step and iteration == simulate_crash_at_step:
            raise RuntimeError(f"Simulated Crash triggered at step {iteration}")
            
    # Exceeded max_iterations
    return {
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "FALLBACK: Maximum tool steps reached without final resolution. Handing over to human support.",
        "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used
    }

def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    state_file = os.path.join(STATE_DIR, f"{run_id}.json")
    with open(state_file, "r", encoding="utf-8") as f:
        state_data = json.load(f)
        
    scenario_id = state_data["scenario_id"]
    messages = state_data["messages"]
    tokens_used = state_data["tokens_used"]
    start_step = state_data["step"]
    tools_executed = state_data["tools_executed"]
    
    start_time = time.monotonic()
    llm_calls = 0
    steps = []
    
    for iteration in range(start_step + 1, max_iterations + 1):
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            return {
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK: Execution timed out.",
                "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used
            }
            
        response = mock_llm.chat(messages)
        llm_calls += 1
        call_tokens = response.usage.total_tokens
        
        if (tokens_used + call_tokens) > token_limit:
            return {
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK: Token limit reached.",
                "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used
            }
            
        tokens_used += call_tokens
        choice = response.choices[0].message
        
        if not choice.tool_calls:
            final_content = choice.content or ""
            status = final_content.split(":")[0].strip() if ":" in final_content else "DENIED"
            return {
                "status": status,
                "final_answer": final_content,
                "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
                "tools_executed_total": tools_executed
            }
            
        tool_call = choice.tool_calls[0]
        tool_name = tool_call.name
        tool_args = tool_call.arguments
        
        tool_func = TOOL_MAP.get(tool_name)
        tool_result = tool_func(**tool_args)
        
        steps.append({"tool": tool_name, "args": tool_args, "observation": tool_result})
        tools_executed.append(tool_name)
        
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tool_call.id, "name": tool_name, "arguments": tool_args}]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_name,
            "content": json.dumps(tool_result)
        })
        
        save_checkpoint(run_id, scenario_id, messages, tokens_used, iteration, tools_executed)

    return {
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "FALLBACK: Max iterations reached.",
        "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
        "tools_executed_total": tools_executed
    }

print("Tasks 6, 7, and 8: ReAct runtime loops, state persistence hooks, and recovery engines successfully bound.")
# =====================================================================
# TASK 6 to 8 Execution & Scenarios Driver
# =====================================================================
def run_all_agent_scenarios():
    mock_llm.reset()
    scenarios_df = pd.read_csv(os.path.join(RAW_DIR, "agent_scenarios.csv"))
    
    agent_runs = []
    
    print("\n--- TASK 6 AGENT SCENARIOS RUN ---")
    for _, row in scenarios_df.iterrows():
        sid = row["scenario_id"]
        msg = row["message"]
        
        res = run_agent(msg, sid)
        
        # Save output line
        run_record = {
            "scenario_id": sid,
            "status": res["status"],
            "final_answer": res["final_answer"],
            "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"],
            "steps": res["steps"]
        }
        agent_runs.append(run_record)
        
        print(f"Scenario {sid} -> Status: {res['status']} | LLM Calls: {res['llm_calls']} | Tokens: {res['tokens_used']}")

    # Write outputs/agent_runs.jsonl
    with open(os.path.join(OUTPUT_DIR, "agent_runs.jsonl"), "w", encoding="utf-8") as f:
        for record in agent_runs:
            f.write(json.dumps(record) + "\n")
            
    # Task 7 Demos
    print("\n--- TASK 7 STOP SAFELY DEMONSTRATIONS ---")
    s1_msg = scenarios_df.iloc[0]["message"]
    demo_token = run_agent(s1_msg, "S1_token_demo", token_limit=200)
    print(f"Token Limit Demo Status: {demo_token['status']}")
    
    demo_timeout = run_agent(s1_msg, "S1_timeout_demo", time_limit_seconds=0)
    print(f"Timeout Demo Status: {demo_timeout['status']}")
    
    # Task 8 Resumption Demo
    print("\n--- TASK 8 CRASH AND RESUME DEMO ---")
    crash_run_id = "run_S1_crash_test"
    try:
        run_agent(s1_msg, "S1", run_id=crash_run_id, simulate_crash_at_step=2)
    except RuntimeError as e:
        print(f"Caught expected crash: {e}")
        
    resumed_res = resume_agent(crash_run_id)
    print(f"Resumed Run Status: {resumed_res['status']}")
    print(f"Executed Tools across both halves: {resumed_res.get('tools_executed_total')}")

if __name__ == "__main__":
    print("\nEvaluating RAG Systems...")
    evaluate_rag()
    print("\nRunning Autonomous Agent Scenarios Loop...")
    run_all_agent_scenarios()
