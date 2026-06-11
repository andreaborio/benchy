#!/usr/bin/env python3
"""Shared helpers for the benchy runners (eval_mcq / eval_code / healthbench):
model resolution, per-run provenance metadata, the shared chat client, the unified
runner CLI and the RunWriter that owns a run's artifacts. Stdlib only, no third-party deps."""
import os, sys, json, time, socket, hashlib, argparse, datetime, subprocess, platform
import urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))

__version__ = "0.2.0"

# Canonical values of the "mode" field in runs.jsonl records (the CLI tokens are
# think|nothink — parse_run_args maps think -> MODE_THINKING).
MODE_THINKING = "thinking"
MODE_NOTHINK = "nothink"
# Canonical values of the "kind" field in runs.jsonl records.
KIND_MCQ = "mcq"
KIND_CODE = "code"
KIND_RUBRIC = "rubric"

# The ONLY place the default server URL is defined — everything else goes through settings().
DEFAULT_SERVER = "http://127.0.0.1:8000"


def settings():
    """Resolved runtime settings shared by runners and dashboard. server_base precedence:
    env BENCHY_SERVER > optional "server_base" string key in config.json (next to this
    file) > DEFAULT_SERVER. Reads no network; safe at import time."""
    base = os.environ.get("BENCHY_SERVER")
    if not base:
        try:
            v = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8")).get("server_base")
            base = v if isinstance(v, str) and v.strip() else None
        except Exception:
            base = None
    return {"server_base": (base or DEFAULT_SERVER).rstrip("/")}


class ChatError(Exception):
    """A chat-completions request that still failed after all retries."""


def chat(messages, think=False, max_tokens=256, model=None, get_model=None,
         seed=None, server_base=None, timeout=600):
    """The one OpenAI-chat-completions client all runners share: POST /v1/chat/completions
    and return the assistant text. Deterministic by construction (temperature pinned to 0,
    optional seed forwarded), keeps the ds4 "think" request field, 2 retries with
    exponential backoff (2s, 8s) on URLError/HTTPError/timeout, raises ChatError after the
    final failure. `messages` is a plain prompt string (wrapped as one user message) or a
    full messages list; the model comes from `model` or the runner's lazy `get_model`."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    base = (server_base or settings()["server_base"]).rstrip("/")
    body = {"model": model if model is not None else (get_model() if get_model else "default"),
            "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "think": bool(think)}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    err = None
    for delay in (2, 8, None):  # 2 retries with exponential backoff, then give up
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"]
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            err = e
            if delay is not None:
                time.sleep(delay)
    raise ChatError("chat request failed after retries: %s" % err)


def _split_think(feed):
    """Build an incremental <think>…</think> splitter. Returns a function feed(chunk) that
    routes content to on-the-fly ('reasoning', text) / ('answer', text) callbacks via `feed`,
    holding back a partial trailing tag so a tag split across two stream chunks is never
    mislabelled. Call the returned splitter with chunks, then close() to flush the tail."""
    state = {"in": False, "buf": ""}
    OPEN, CLOSE = "<think>", "</think>"
    def push(chunk):
        state["buf"] += chunk
        while state["buf"]:
            if not state["in"]:
                i = state["buf"].find(OPEN)
                if i == -1:
                    keep = len(OPEN) - 1                      # could be a partial "<think" tail
                    if len(state["buf"]) > keep:
                        feed("answer", state["buf"][:-keep] if keep else state["buf"])
                        state["buf"] = state["buf"][-keep:] if keep else ""
                    break
                if i: feed("answer", state["buf"][:i])
                state["buf"] = state["buf"][i + len(OPEN):]; state["in"] = True
            else:
                i = state["buf"].find(CLOSE)
                if i == -1:
                    keep = len(CLOSE) - 1
                    if len(state["buf"]) > keep:
                        feed("reasoning", state["buf"][:-keep] if keep else state["buf"])
                        state["buf"] = state["buf"][-keep:] if keep else ""
                    break
                if i: feed("reasoning", state["buf"][:i])
                state["buf"] = state["buf"][i + len(CLOSE):]; state["in"] = False
    def close():
        if state["buf"]:
            feed("reasoning" if state["in"] else "answer", state["buf"]); state["buf"] = ""
    push.close = close
    return push


def chat_stream(messages, think=False, max_tokens=256, model=None, get_model=None,
                seed=None, server_base=None, timeout=600, on_delta=None):
    """Streaming sibling of chat() for the live 'generation' box. POSTs with stream:true,
    parses the SSE token deltas, and calls on_delta(kind, text) as content arrives — kind is
    'reasoning' (a provider reasoning/reasoning_content delta, or text inside an inline
    <think>…</think> span) or 'answer' (everything else). It accumulates and returns the SAME
    full assistant text chat() would return (message.content; provider reasoning_content is a
    separate field, surfaced to on_delta but NOT folded into the returned text), so the scorer
    downstream is unchanged. A mid-stream transport failure restarts the stream from scratch
    within the same 2s/8s retry budget (on_delta('reset','') clears the box); a server that
    doesn't actually stream (non event-stream response) falls back to one blocking read, routed
    through the same splitter so the box still shows reasoning/answer (all at once). on_delta
    is display-only and best-effort — exceptions from it never abort the run."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    base = (server_base or settings()["server_base"]).rstrip("/")
    body = {"model": model if model is not None else (get_model() if get_model else "default"),
            "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "think": bool(think), "stream": True}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})

    def emit(kind, text):
        if text and on_delta:
            try: on_delta(kind, text)
            except Exception: pass

    err = None
    for delay in (2, 8, None):
        full = []                                            # message.content pieces (the return value)
        split = _split_think(emit)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if "text/event-stream" not in (resp.headers.get("Content-Type") or ""):
                    # server ignored stream:true — one blocking read, routed through the splitter
                    content = json.loads(resp.read())["choices"][0]["message"]["content"] or ""
                    split(content); split.close()
                    return content
                done = False
                for raw in resp:                             # SSE: one JSON object per "data:" line
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True; break
                    try:
                        choice = json.loads(data)["choices"][0]
                    except Exception:
                        continue
                    if choice.get("finish_reason"):
                        done = True                          # last content chunk — stream is complete
                    delta = choice.get("delta", {})
                    rc = delta.get("reasoning_content") or delta.get("reasoning")
                    if rc:
                        emit("reasoning", rc)                # separate field — not part of content
                    piece = delta.get("content")
                    if piece:
                        full.append(piece); split(piece)     # inline <think> handled by the splitter
                if not done:
                    # the stream ended without [DONE] or a finish_reason — treat it as truncated
                    # and retry, since this text feeds scoring (BENCHY_LIVE_STREAM expects a
                    # spec-compliant streaming server: ds4 / vLLM / llama.cpp all send one)
                    raise ConnectionError("stream ended without a terminal marker")
                split.close()
                return "".join(full)
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            err = e
            if on_delta:                                     # tell the box to drop the partial buffer
                try: on_delta("reset", "")                   # (empty text — bypass emit's text guard)
                except Exception: pass
            if delay is not None:
                time.sleep(delay)
    raise ChatError("chat stream failed after retries: %s" % err)


def stream_into(writer, qnum, n, question, **chat_kw):
    """Run chat_stream(**chat_kw) and funnel its reasoning/answer deltas into writer.gen()
    (throttled to ~10 Hz) so the dashboard's live generation box fills token-by-token, then a
    final flush with the complete answer. Returns the SAME assembled text chat() would (so the
    scorer is unchanged). qnum is the 1-based question-in-flight number; shared by the runners
    so the streaming glue lives in one place."""
    buf = {"reasoning": "", "answer": ""}
    st = {"last": 0.0}
    def flush(force=False):
        now = time.time()
        if not force and now - st["last"] < 0.1:   # throttle: at most ~10 writes/sec
            return
        st["last"] = now
        writer.gen({"running": True, "i": qnum, "n": n,
                    "phase": "answering" if buf["answer"] else "reasoning",
                    "q": (question or "")[:200],
                    "reasoning": buf["reasoning"][-8000:], "answer": buf["answer"][-8000:],
                    "gen_tokens": len((buf["reasoning"] + " " + buf["answer"]).split())})
    def on_delta(kind, text):
        if kind == "reset":                         # a retry restarted the stream — drop partials
            buf["reasoning"] = ""; buf["answer"] = ""; flush(force=True); return
        if kind in buf:
            buf[kind] += text
        flush()
    out = chat_stream(on_delta=on_delta, **chat_kw)
    flush(force=True)                               # final state: the complete answer
    return out


def parse_run_args(argv=None, prog=None, bench_help="benchmark JSONL path"):
    """The ONE runner CLI, shared by eval_mcq / eval_code / healthbench:
        positional BENCH N MODE TAG, optional --seed INT (default 1234).
    MODE is think|nothink on the command line; the returned namespace carries the
    canonical mode (.mode = MODE_THINKING|MODE_NOTHINK) plus a .think bool."""
    p = argparse.ArgumentParser(prog=prog)
    p.add_argument("bench", help=bench_help)
    p.add_argument("n", type=int, help="number of questions to run (0 = all)")
    p.add_argument("mode", choices=("think", "nothink"), help="request extended thinking or not")
    p.add_argument("tag", help="free-form run label stored in runs.jsonl")
    p.add_argument("--seed", type=int, default=1234, help="dataset shuffle seed (default 1234)")
    args = p.parse_args(argv)
    args.think = args.mode == "think"
    args.mode = MODE_THINKING if args.think else MODE_NOTHINK
    return args


class RunWriter:
    """Owns one run's artifacts under results/: clears then appends the per-question
    stream.jsonl live feed and the per-run details file, atomically updates live.json
    (via write_live) and appends the final summary record to runs.jsonl. Every record it
    writes carries ts (isoformat, seconds), kind, an errors count (0 default) and the
    run_meta() provenance — existing field names are never changed, only added to."""

    def __init__(self, bench, mode, tag, kind, results_dir=None):
        self.bench, self.mode, self.tag, self.kind = bench, mode, tag, kind
        self.results = results_dir or os.path.join(HERE, "results")
        self.details_dir = os.path.join(self.results, "details")
        self.runs_path = os.path.join(self.results, "runs.jsonl")
        self.live_path = os.path.join(self.results, "live.json")
        self.stream_path = os.path.join(self.results, "stream.jsonl")
        os.makedirs(self.details_dir, exist_ok=True)
        self.run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.details_path = os.path.join(self.details_dir,
                                         "%s__%s__%s.jsonl" % (bench, mode, self.run_id))
        self.gen_path = os.path.join(self.results, "gen.json")  # live token-stream buffer (display only)
        self._last_gen = None
        open(self.stream_path, "w", encoding="utf-8").close()   # clear the live feed
        open(self.details_path, "w", encoding="utf-8").close()
        write_live(self.gen_path, {"running": False})           # reset the generation box for this run

    def stream(self, ev):
        """Append one per-question event to the live stream.jsonl feed."""
        with open(self.stream_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(ev) + "\n")

    def gen(self, obj):
        """Atomic gen.json update: the live token-streaming buffer for the question currently
        being generated (the dashboard's reasoning+answer box). Display only — gen.json never
        feeds scoring, so a torn/partial buffer can only affect what the box shows, never a
        number. tag/benchmark/mode/ts are filled in; the last write is remembered for gen_finish."""
        out = {"tag": self.tag, "benchmark": self.bench, "mode": self.mode}
        out.update(obj)
        out["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
        self._last_gen = out
        write_live(self.gen_path, out)

    def gen_finish(self):
        """At run end, freeze the box on the last question's content with running:false (so the
        final answer stays visible) — or a bare idle state if nothing was ever streamed."""
        out = dict(self._last_gen or {"running": False})
        out["running"] = False
        write_live(self.gen_path, out)

    def detail(self, row):
        """Append one per-question row to this run's details file."""
        with open(self.details_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row) + "\n")

    def live(self, d):
        """Atomic live.json update; tag/benchmark/mode/ts are filled in automatically."""
        out = {"tag": self.tag, "benchmark": self.bench, "mode": self.mode}
        out.update(d)
        out["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
        write_live(self.live_path, out)

    def finish(self, fields, model, server_base, dataset_path=None):
        """Append the final summary record to runs.jsonl and return it. `fields` are the
        runner-specific numbers (n, accuracy, errors, ...); ts/tag/benchmark/mode/kind,
        errors=0 and the details filename are defaulted, run_meta() is merged last."""
        rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
               "tag": self.tag, "benchmark": self.bench, "mode": self.mode,
               "kind": self.kind, "errors": 0}
        rec.update(fields)
        rec.setdefault("details", os.path.basename(self.details_path))
        rec.update(run_meta(model, server_base, dataset_path))
        with open(self.runs_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec) + "\n")
        return rec


def write_live(path, obj):
    """Atomic JSON write (tmp + os.replace): a concurrent reader (the dashboard) sees either
    the old file or the new one, never a torn half-write."""
    tmp = path + ".tmp"
    open(tmp, "w", encoding="utf-8", newline="\n").write(json.dumps(obj))
    os.replace(tmp, path)


def resolve_model(server_base):
    """Model id for chat payloads: BENCHY_MODEL override > server's /v1/models > 'default'."""
    if os.environ.get("BENCHY_MODEL"):
        return os.environ["BENCHY_MODEL"]
    try:
        with urllib.request.urlopen(server_base.rstrip("/") + "/v1/models", timeout=5) as r:
            ids = [m.get("id") for m in (json.load(r).get("data") or []) if m.get("id")]
        if ids:
            return ids[0]
    except Exception:
        pass
    return "default"


def git_sha():
    """Short git revision of the benchy checkout, so a number is tied to the code that made it."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                             capture_output=True, encoding="utf-8", timeout=3)
        return out.stdout.strip() or None
    except Exception:
        return None


def sha256_file(path):
    """Full SHA-256 hex digest of a file's bytes — the ONE content-hash primitive behind both
    the lockfile pin (api.content_sha, full digest) and run provenance (file_sha, a 12-hex
    prefix of the same digest), so the two are always prefix-comparable. None if missing."""
    if not path:
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def file_sha(path):
    """Short content hash of a dataset file, so a published number is tied to the exact rows.
    The first 12 hex chars of sha256_file — i.e. of the lockfile's content_sha when locked."""
    d = sha256_file(path)
    return d[:12] if d else None


def check_dataset_lock(path):
    """Runtime dataset-integrity gate shared by the runners: when `path` IS a registered
    benchmark's data file and its benchmarks.lock.json entry pins a content_sha, recompute
    the file's hash and compare. Returns True (verified against the lock) or False (nothing
    to check: BYO file, no lock entry, no pinned hash yet). A mismatch ABORTS the run;
    BENCHY_SKIP_LOCK_CHECK=1 downgrades the abort to a stderr warning (the run is then
    recorded with locked: false)."""
    try:
        import api  # local module; lazy so importing benchy_common stays registry-free
        key = os.path.basename(path or "")
        key = key[:-6] if key.endswith(".jsonl") else key
        want = (api._load_lock()["benchmarks"].get(key) or {}).get("content_sha")
        if not want or os.path.realpath(path) != os.path.realpath(api.data_path(key)):
            return False   # unlocked / not hashed yet, or a BYO file that shares the name
    except Exception:
        return False       # no registry/lockfile available — BYO setups keep working
    got = sha256_file(path)
    if got == want:
        return True
    msg = ("dataset drifted from lock: data/%s.jsonl hashes %s but benchmarks.lock.json pins "
           "%s (run 'python3 api.py verify %s' to restore the pinned snapshot, or 'relock' to "
           "accept the new content)" % (key, (got or "?")[:12], want[:12], key))
    if os.environ.get("BENCHY_SKIP_LOCK_CHECK", "").lower() in ("1", "true", "yes", "on"):
        print("⚠ BENCHY_SKIP_LOCK_CHECK — " + msg, file=sys.stderr)
        return False
    sys.exit("⛔ " + msg)


def run_meta(model, server_base, dataset_path=None):
    """Provenance stamped into every runs.jsonl record so results are reproducible/comparable:
    which model/quant, which server, which benchy version + git revision (sha is None outside
    a git checkout — the version still covers provenance), which dataset snapshot, which host.
    'model' is the single most important field for quantized-model benchmarking — capture it."""
    return {"model": model, "server": server_base, "benchy_version": __version__,
            "benchy_sha": git_sha(), "data_sha": file_sha(dataset_path),
            "host": platform.platform(), "py": platform.python_version()}
