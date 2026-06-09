#!/usr/bin/env python3
"""HealthBench eval with a faithful API judge (OpenAI simple-evals methodology).

The local model under test answers the open-ended conversations; an API grader scores
each rubric criterion as criteria_met (true/false) using OpenAI's exact GRADER_TEMPLATE.
Per-example score = sum(points of met criteria) / sum(positive points), clamped to [0,1].
Mean over examples = HealthBench score. (This is one of the bundled rubric-graded
benchmarks; MCQ sets go through eval_mcq.py instead.)

Usage: healthbench.py <N> [tag] [think|nothink] [hard|consensus]
Env: BENCHY_SERVER (default http://127.0.0.1:8000), BENCHY_MODEL (default: auto-detected),
     GRADER_MODEL (override the per-provider grader; default gpt-4.1 / claude-sonnet-4-6 / gemini-2.5-pro).
Reads the grader API key from a .apikey file next to this script (provider auto-detected; never logged).
"""
import json, os, re, sys, time, random, datetime, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); RES = os.path.join(HERE, "results"); DET = os.path.join(RES, "details")
SERVER_BASE = os.environ.get("BENCHY_SERVER", "http://127.0.0.1:8000").rstrip("/")
SERVER = SERVER_BASE + "/v1/chat/completions"
SEED = 1234

def resolve_model():
    """Model id for chat payloads: BENCHY_MODEL override > server's /v1/models > 'default'."""
    if os.environ.get("BENCHY_MODEL"): return os.environ["BENCHY_MODEL"]
    try:
        with urllib.request.urlopen(SERVER_BASE + "/v1/models", timeout=5) as r:
            ids = [m.get("id") for m in (json.load(r).get("data") or []) if m.get("id")]
        if ids: return ids[0]
    except Exception:
        pass
    return "default"

MODEL = resolve_model()
URLS = {"hard": "https://huggingface.co/datasets/openai/healthbench/resolve/main/hard_2025-05-08-21-00-10.jsonl",
        "consensus": "https://huggingface.co/datasets/openai/healthbench/resolve/main/consensus_2025-05-09-20-00-46.jsonl"}
# default grader per provider; OpenAI gpt-4.1 = the official HealthBench judge
GRADER = {"openai": "gpt-4.1", "anthropic": "claude-sonnet-4-6", "google": "gemini-2.5-pro"}

GRADER_TEMPLATE = """Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
{conversation}

# Rubric item
{rubric_item}

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true if all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that the patient should drink fluids for hydration, such as water, juice, or sports drinks", and the response just says "the patient should drink water", the criteria should be met.

# Example response
{{"explanation": "The response correctly identifies ...", "criteria_met": true}}

Return just the json object."""

def load_key():
    p = os.path.join(HERE, ".apikey")
    if not os.path.exists(p):
        sys.exit("No .apikey found next to healthbench.py — create it:  echo 'sk-...' > .apikey  (then chmod 600 .apikey)")
    k = open(p).read().strip()
    prov = "anthropic" if k.startswith("sk-ant-") else "google" if k.startswith("AIza") else "openai"
    model = os.environ.get("GRADER_MODEL", GRADER[prov])
    return k, prov, model

def grader_call(key, prov, model, prompt):
    if prov == "anthropic":
        req = urllib.request.Request("https://api.anthropic.com/v1/messages",
            data=json.dumps({"model": model, "max_tokens": 700,
                "messages": [{"role": "user", "content": prompt}]}).encode(),
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())["content"][0]["text"]
    if prov == "google":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        req = urllib.request.Request(url, data=json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode(),
            headers={"content-type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())["candidates"][0]["content"]["parts"][0]["text"]
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
        data=json.dumps({"model": model, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]

def parse_met(text):
    m = re.search(r'"criteria_met"\s*:\s*(true|false)', text, re.I)
    if m: return m.group(1).lower() == "true"
    try:
        return bool(json.loads(re.search(r"\{.*\}", text, re.S).group(0)).get("criteria_met"))
    except Exception:
        return False

def ask_model(messages, think):
    body = {"model": MODEL, "messages": messages, "temperature": 0.0,
            "max_tokens": 4000 if think else 1200, "think": bool(think)}
    req = urllib.request.Request(SERVER, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    txt = json.loads(urllib.request.urlopen(req, timeout=1200).read())["choices"][0]["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.S | re.I).strip()

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    tag = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    think = (sys.argv[3] if len(sys.argv) > 3 else "think") != "nothink"
    subset = sys.argv[4] if len(sys.argv) > 4 else "hard"
    key, prov, model = load_key()
    bench = "healthbench_" + subset
    print(f"grader={model} ({prov}) · {bench} · N={n} · {'thinking' if think else 'nothink'}")
    path = os.path.join(DATA, f"healthbench_{subset}.jsonl")
    if not os.path.exists(path):
        print("downloading", subset, "..."); urllib.request.urlretrieve(URLS[subset], path)
    rows = [json.loads(l) for l in open(path) if l.strip()]
    random.seed(SEED); random.shuffle(rows); rows = rows[:n]
    stream = os.path.join(RES, "stream.jsonl"); open(stream, "w").close()
    os.makedirs(DET, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    detfile = os.path.join(DET, f"{bench}__{'thinking' if think else 'nothink'}__{run_id}.jsonl")
    open(detfile, "w").close()
    scores, t0 = [], time.time()
    for i, ex in enumerate(rows, 1):
        q0 = time.time()
        msgs, rubrics = ex["prompt"], ex.get("rubrics", [])
        try: resp = ask_model(msgs, think)
        except Exception as e: print("  gen error", e); resp = ""
        conv = "\n\n".join(f"{m['role']}: {m['content']}" for m in msgs) + f"\n\nassistant: {resp}"
        total = sum(r["points"] for r in rubrics if r["points"] > 0)
        achieved = 0; crit = []
        for r in rubrics:
            pr = GRADER_TEMPLATE.format(conversation=conv, rubric_item=f"[{r['points']}] {r['criterion']}")
            expl = ""
            try:
                gtext = grader_call(key, prov, model, pr); met = parse_met(gtext)
                me = re.search(r'"explanation"\s*:\s*"(.*?)"', gtext, re.S)
                expl = (me.group(1) if me else gtext)[:240]
            except Exception:
                met = False
            if met: achieved += r["points"]
            crit.append({"criterion": r["criterion"], "points": r["points"], "met": bool(met), "explanation": expl})
        score = max(0.0, min(1.0, achieved / total)) if total else 0.0
        scores.append(score); acc = round(sum(scores) / len(scores) * 100, 1); dt = round(time.time() - q0, 1)
        ev = {"i": i, "n": n, "ok": score >= 0.5, "pred": f"{score*100:.0f}%", "gold": "rubric", "acc": acc,
              "t": dt, "q": (msgs[-1]["content"] if msgs else "")[:260],
              "pred_txt": f"score {score*100:.0f}% ({achieved}/{total}pts)", "gold_txt": f"{len(rubrics)} criteria", "ans": resp[:180]}
        open(stream, "a").write(json.dumps(ev) + "\n")
        open(detfile, "a").write(json.dumps({"i": i, "conversation": msgs, "response": resp,
            "score": round(score * 100, 1), "achieved": achieved, "total": total, "criteria": crit}) + "\n")
        json.dump({"running": True, "benchmark": bench, "mode": "thinking" if think else "nothink",
                   "tag": tag, "i": i, "n": n, "accuracy": acc,
                   "ts": datetime.datetime.now().isoformat(timespec="seconds")},
                  open(os.path.join(RES, "live.json"), "w"))
        print(f"  [{i}/{n}] score {score*100:.0f}%  running {acc}%  ({dt}s)")
    final = round(sum(scores) / len(scores) * 100, 1) if scores else 0.0
    run = {"ts": time.strftime("%Y-%m-%d"), "tag": tag, "benchmark": bench,
           "mode": "thinking" if think else "nothink", "n": len(scores), "accuracy": final, "seed": SEED,
           "duration_s": round(time.time() - t0, 1), "sec_per_q": round((time.time() - t0) / max(1, len(scores)), 1),
           "grader": model, "details": os.path.basename(detfile),
           "notes": f"HealthBench {subset}, faithful API judge ({model})"}
    open(os.path.join(RES, "runs.jsonl"), "a").write(json.dumps(run) + "\n")
    json.dump({"running": False}, open(os.path.join(RES, "live.json"), "w"))
    print(f"DONE — HealthBench {subset}: {final}%  (N={len(scores)}, grader={model})")

if __name__ == "__main__":
    main()
