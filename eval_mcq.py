#!/usr/bin/env python3
"""Multiple-choice benchmark scorer against any OpenAI-compatible chat API, with
structured + live per-question streaming for the dashboard.

Works on any dataset in the unified MCQ format — {question, options{A..}, answer_idx} —
so you can drop in your own JSONL (knowledge, reasoning, coding-MCQ, …), not just the
bundled sets.

Usage: eval_mcq.py <jsonl> <N> [think|nothink] [tag] [notes...]

Env: BENCHY_SERVER (default http://127.0.0.1:8000), BENCHY_MODEL (default: auto-detected
from the server's /v1/models).

Writes (under results/):
  runs.jsonl   one summary record per completed run
  live.json    aggregate live progress
  stream.jsonl one line per question (cleared at run start) — the live Q/A feed
Stdlib only. Greedy, deterministic.
"""
import json, sys, re, random, urllib.request, time, datetime, os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
RUNS = os.path.join(RESULTS, "runs.jsonl")
LIVE = os.path.join(RESULTS, "live.json")
STREAM = os.path.join(RESULTS, "stream.jsonl")
DETAILS = os.path.join(RESULTS, "details")
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

def load(path, n):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    random.Random(SEED).shuffle(rows)
    return rows if n == 0 else rows[:n]

def build_prompt(r):
    opts = r["options"]
    keys = [k for k in "ABCDEFGHIJ" if k in opts]
    lines = [r["question"].strip(), ""]
    for k in keys:
        lines.append(f"{k}. {opts[k]}")
    lines += ["", "Answer with only the single letter of the correct option. No explanation."]
    return "\n".join(lines), keys

def ask(prompt, think):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 3072 if think else 8, "temperature": 0.0, "think": bool(think)}
    req = urllib.request.Request(SERVER, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]

def extract(text, keys):
    t = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    cands = re.findall(r"\b([%s])\b" % "".join(keys), t.upper())
    if cands:
        return cands[-1]
    m = re.search(r"[%s]" % "".join(keys), t.upper())
    return m.group(0) if m else "?"

def server_timing():
    """Best-effort per-question timing from server.log: prefill seconds, total seconds, gen tokens, avg t/s.
    The eval is sequential (one request at a time), so the most recent finished block is this question's."""
    log = os.path.join(RESULTS, "server.log")
    try:
        with open(log, "rb") as f:
            f.seek(0, 2); sz = f.tell(); f.seek(max(0, sz - 9000)); tail = f.read().decode("utf-8", "ignore")
        prefill = total = gen = avg = None
        for ln in tail.splitlines():
            m = re.search(r"prompt done ([\d.]+)s", ln)
            if m: prefill = round(float(m.group(1)), 1)
            m = re.search(r"gen=(\d+) finish=\S+ ([\d.]+)s", ln)
            if m: gen = int(m.group(1)); total = float(m.group(2))
            m = re.search(r"avg=([\d.]+) t/s", ln)
            if m: avg = float(m.group(1))
        gen_s = round(total - prefill, 1) if (total is not None and prefill is not None) else (round(total, 1) if total is not None else None)
        return {"prefill_s": prefill, "gen_tokens": gen, "gen_tps": avg, "gen_s": gen_s}
    except Exception:
        return {}

def write_live(d):
    os.makedirs(RESULTS, exist_ok=True)
    d["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
    tmp = LIVE + ".tmp"; open(tmp, "w").write(json.dumps(d)); os.replace(tmp, LIVE)

def main():
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    think = len(sys.argv) > 3 and sys.argv[3] == "think"
    tag = sys.argv[4] if len(sys.argv) > 4 else "baseline"
    notes = " ".join(sys.argv[5:]) if len(sys.argv) > 5 else ""
    mode = "thinking" if think else "nothink"
    bench = os.path.basename(path).replace(".jsonl", "")
    rows = load(path, n)
    n_options = max((len(build_prompt(r)[1]) for r in rows), default=0)  # #answer options (for chance line + bias χ²)
    os.makedirs(RESULTS, exist_ok=True)
    open(STREAM, "w").close()  # clear the live feed
    os.makedirs(DETAILS, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    detfile = os.path.join(DETAILS, f"{bench}__{mode}__{run_id}.jsonl")
    open(detfile, "w").close()
    correct = 0; dist = {}; t0 = time.time()
    for i, r in enumerate(rows):
        prompt, keys = build_prompt(r)
        q0 = time.time()
        try:
            out = ask(prompt, think)
        except Exception as e:
            out = f"ERR:{e}"
        pred = extract(out, keys); gold = r["answer_idx"]; ok = pred == gold; correct += ok
        dist[pred] = dist.get(pred, 0) + 1
        acc = 100 * correct / (i + 1)
        opts = r.get("options", {})
        ans = re.sub(r"<think>.*?</think>", "", out, flags=re.S | re.I).strip()
        think_txt = " ".join(re.findall(r"<think>(.*?)</think>", out, flags=re.S | re.I))
        think_tokens = len(think_txt.split()) if think_txt else 0
        tm = server_timing()
        ev = {"i": i + 1, "n": len(rows), "ok": bool(ok), "pred": pred, "gold": gold,
              "acc": round(acc, 1), "t": round(time.time() - q0, 1),
              "q": r["question"].strip()[:260],
              "pred_txt": opts.get(pred, "")[:90], "gold_txt": opts.get(gold, "")[:90],
              "ans": ans[:180], "think_tokens": think_tokens,
              "prefill_s": tm.get("prefill_s"), "gen_s": tm.get("gen_s"),
              "gen_tokens": tm.get("gen_tokens"), "gen_tps": tm.get("gen_tps")}
        with open(STREAM, "a") as f: f.write(json.dumps(ev) + "\n")
        with open(detfile, "a") as f:
            f.write(json.dumps({"i": i + 1, "question": r["question"], "options": opts,
                                "pred": pred, "gold": gold, "ok": bool(ok), "answer": ans,
                                "think_tokens": think_tokens, "prefill_s": tm.get("prefill_s"),
                                "gen_s": tm.get("gen_s"), "gen_tokens": tm.get("gen_tokens"),
                                "gen_tps": tm.get("gen_tps")}) + "\n")
        write_live({"running": True, "tag": tag, "benchmark": bench, "mode": mode,
                    "i": i + 1, "n": len(rows), "correct": correct,
                    "accuracy": round(acc, 1), "elapsed_s": round(time.time() - t0)})
        print(f"{i+1:>4}/{len(rows)} pred={pred} gold={gold} {'OK ' if ok else ' x '} acc={acc:5.1f}%", flush=True)
    dt = time.time() - t0; acc = 100 * correct / len(rows)
    rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "tag": tag,
           "benchmark": bench, "mode": mode, "n": len(rows), "correct": correct,
           "accuracy": round(acc, 1), "seed": SEED, "duration_s": round(dt),
           "sec_per_q": round(dt / len(rows), 1), "letter_dist": dist, "n_options": n_options,
           "notes": notes, "details": os.path.basename(detfile)}
    open(RUNS, "a").write(json.dumps(rec) + "\n")
    write_live({"running": False, "tag": tag, "benchmark": bench, "mode": mode,
                "i": len(rows), "n": len(rows), "correct": correct,
                "accuracy": round(acc, 1), "elapsed_s": round(dt)})
    print(f"\n=== {bench} [{mode}] tag={tag} N={len(rows)} "
          f"accuracy = {correct}/{len(rows)} = {acc:.1f}%  ({dt:.0f}s, {dt/len(rows):.1f}s/q) ===")

if __name__ == "__main__":
    main()
