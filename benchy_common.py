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
        open(self.stream_path, "w", encoding="utf-8").close()   # clear the live feed
        open(self.details_path, "w", encoding="utf-8").close()

    def stream(self, ev):
        """Append one per-question event to the live stream.jsonl feed."""
        with open(self.stream_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(ev) + "\n")

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
