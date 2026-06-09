#!/usr/bin/env python3
"""Shared helpers for the benchy runners (eval_mcq / eval_code / healthbench):
model resolution and per-run provenance metadata. Stdlib only, no third-party deps."""
import os, json, hashlib, subprocess, platform, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


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
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or None
    except Exception:
        return None


def file_sha(path):
    """Short content hash of a dataset file, so a published number is tied to the exact rows."""
    if not path:
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    except Exception:
        return None


def run_meta(model, server_base, dataset_path=None):
    """Provenance stamped into every runs.jsonl record so results are reproducible/comparable:
    which model/quant, which server, which code revision, which dataset snapshot, which host.
    'model' is the single most important field for quantized-model benchmarking — capture it."""
    return {"model": model, "server": server_base, "benchy_sha": git_sha(),
            "data_sha": file_sha(dataset_path), "host": platform.platform(),
            "py": platform.python_version()}
