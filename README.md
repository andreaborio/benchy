# benchy

**A local LLM benchmark suite and live dashboard.**

`benchy` runs a panel of question-answering benchmarks against any OpenAI-compatible
inference server, scores them deterministically, and shows the results — accuracy,
confidence intervals, per-question drill-down, and live system metrics — in a
single-command, zero-dependency web dashboard. The model id is **auto-detected** from the server and the scoring core runs
anywhere Python does; nothing about your model is hardcoded, so you clone it and run. (The
*live host-metrics* panels are macOS-oriented — see [Methodology](#methodology--caveats-read-before-quoting-numbers).)

It talks plain OpenAI Chat Completions, so it works against llama.cpp / vLLM / Ollama /
LM Studio / [ds4](#works-with-any-openai-compatible-server) — anything that serves
`/v1/chat/completions` and `/v1/models`.

## Highlights

- **Fetch well-known benchmarks** from the UI or CLI — MMLU-Pro, SuperGPQA, HumanEval/MBPP
  (executed), MedXpertQA, MedMCQA, MedQA, plus a legacy panel — each tagged **current**
  (still discriminates mid-2026 models) or **legacy** (saturated). `fetch_benchmarks.py current`.
- **Bring your own benchmark** — any `{question, options{A..}, answer_idx}` JSONL drops
  straight in (`answer_idx` may be the option **letter** `"B"` or a 0-based **integer** `1`).
  Add coding, domain, or private sets without touching the code.
- **Deterministic, bias-aware scoring** — greedy decoding, fixed seed, **anchored
  case-sensitive answer extraction** (prose like “…a common cause” is never mistaken for
  option *A*), **per-question option-order randomization** to average out letter-position
  bias (`BENCHY_SHUFFLE_OPTIONS=0` to disable), letter-bias (χ²) checks, Wilson 95% CIs, and
  an **unparseable-rate** recorded per run.
- **Paired A/B significance (McNemar)** — compare two run tags on the *same* questions with
  an exact McNemar test. This is the correct way to ask “does quant A differ from quant B”
  and is far more powerful than eyeballing two overlapping Wilson bars. (In the dashboard:
  the **Paired A/B significance** card.)
- **Live dashboard** — per-question feed with drill-down, running accuracy, accuracy vs.
  your own reference baselines, and live host/server metrics (model RSS, decode t/s, system
  memory & swap). ⚠ **The live system-metrics panels are macOS-only and decode t/s is parsed
  from a ds4-style server log** — on other OSes/servers those panels stay empty; the
  benchmarks themselves still run everywhere.
- **Local-only & guarded** — the dashboard binds to `127.0.0.1` and its process-control
  endpoints require a per-launch CSRF token plus same-origin/Host checks, so a random web
  page you visit can’t drive your local server. Code-generation benchmarks that **execute
  model-written code** are **off by default** (`BENCHY_ALLOW_CODE_EXEC=1` to enable).
- **Reproducible run records** — every `runs.jsonl` row stamps the **model id**, server,
  benchy git SHA, dataset content hash, and host, so a published number is traceable.
- **Auto-detect + guided setup** — the model id is read live from `/v1/models`; an in-UI
  **Setup** panel lets you add optional comparison baselines and a display title (saved to
  a git-ignored `config.json`). No static values baked into the repo.
- **`think` / `no-think` modes** — toggle server-side reasoning to measure the
  test-time-reasoning delta (sent as a `think` field; servers that don't use it ignore it).
- **Zero pip dependencies** for the core — Python standard library only. (HealthBench
  grading needs one API key; the dashboard's third-party assets — Chart.js, marked,
  DOMPurify, highlight.js — are vendored under `static/vendor/` (pinned versions, licenses
  in `NOTICE`), so the dashboard is fully offline: no CDN, no Google Fonts, no external
  requests at all.)

## Why benchy

A **local, single-command, zero-dependency companion** to your inference server, built for the tight
**quantize → measure → compare** loop. What it's good at:

- **Zero-friction & self-contained** — one command, `python3 dashboard.py` (the web UI ships
  alongside in `dashboard.html`), Python standard library only, no `pip install`, no config
  to write. Clone and run next to whatever server you already have.
- **Live, not batch** — a real-time per-question feed, running accuracy, and live host/server
  metrics (decode t/s, model RSS, memory) *while the eval runs*: you watch the model work,
  not just read a final number.
- **Statistically honest by default** — greedy/seeded determinism, Wilson 95% CIs,
  option-order randomization, letter-bias χ², an unparseable-rate, and a **paired McNemar A/B
  test** that tells you whether two builds (e.g. two quantizations) *really* differ on the
  same questions — not just whether two error bars happen to overlap.
- **Any OpenAI-compatible server** — llama.cpp, vLLM, Ollama, LM Studio, ds4; nothing is
  patched or engine-specific for scoring.
- **Broad + a ready medical panel** — general reasoning & code (MMLU-Pro, SuperGPQA,
  HumanEval/MBPP) plus a medical set ready out of the box (MedQA, MedMCQA, MedXpertQA,
  HealthBench) — and **bring-your-own** JSONL drops straight in.
- **Reproducible & safe** — every run stamps the model/quant, git SHA, dataset hash and host;
  the control plane is localhost-only with a CSRF guard, and code-execution benchmarks are
  off by default.

## How it works

```
              ┌────────────────┐   OpenAI /v1/chat/completions   ┌──────────────────┐
   benchy ───▶│  eval_mcq.py   │────────────────────────────────▶│  your model      │
   runners    │  healthbench.py│◀────────────────────────────────│  server (:8000)  │
              └───────┬────────┘   completions                    └────────┬─────────┘
                      │ writes                                              │ server.log
                      ▼                                                     │ (live timing)
              results/{runs,details,stream,metrics}.jsonl ◀────────────────┘
                      │
                      ▼
              dashboard.py (:8050) ── live web UI
```

- The **runners** send each question to the server and score the reply. MCQ runners
  (`eval_mcq.py`) parse a single answer letter; `healthbench.py` sends the conversation and
  grades the answer against its rubric with an external judge model.
- The **dashboard** reads only files under `results/` plus the server's `server.log` for
  live decode-speed/timing. It auto-detects the model id (`/v1/models`) and host specs and
  does not modify the model or the server.

## Quickstart

**Requirements:** Python 3.10+ (standard library only). A running OpenAI-compatible server
on `:8000`. For HealthBench grading, an OpenAI/Anthropic/Google API key.

```sh
# 1. Fetch some benchmarks into data/ (see DATA.md for sources & licenses)
python3 fetch_benchmarks.py list                 # see what's available (tiered)
python3 fetch_benchmarks.py current              # fetch the recommended current set
#   …or pick specific ones: python3 fetch_benchmarks.py mmlu_pro humaneval medqa_test
#   …or use the ⬇ Benchmarks button in the dashboard.
#   Fetching pins each set's upstream revision + content hash into benchmarks.lock.json
#   (the lock contract — explicit form: python3 api.py lock <keys>). Runners verify the
#   lock at runtime and abort on drift (BENCHY_SKIP_LOCK_CHECK=1 to bypass).

# 2. Start any OpenAI-compatible server on :8000 (whatever you use)
#    e.g. llama-server -m model.gguf --port 8000   (or vLLM / Ollama / LM Studio / ds4)

# 3. Run a multiple-choice benchmark
#    eval_mcq.py <data.jsonl> <N> <think|nothink> <tag> [--seed INT]
python3 eval_mcq.py data/mmlu_pro.jsonl 60 think run1

# 4. (optional) Run HealthBench — rubric-graded, needs an API key for the grader
#    healthbench.py <hard|consensus> <N> <think|nothink> <tag> [--seed INT]
echo "sk-..." > .apikey            # grader key (OpenAI/Anthropic/Google auto-detected); chmod 600
python3 healthbench.py hard 20 think run1

# 5. Open the live dashboard
python3 dashboard.py 8050          # → http://localhost:8050
```

The dashboard can also fetch benchmarks, start/stop the server, and launch runs from the
browser. **Config:** set `BENCHY_SERVER` (default `http://127.0.0.1:8000`) and
`BENCHY_MODEL` (default: auto-detected from `/v1/models`) if your setup differs.
`BENCHY_RESULTS=<dir>` points the dashboard at another checkout's `results/` — useful to
watch a run launched from a different benchy copy without restarting it.

**Live generation box** — run an eval with `BENCHY_LIVE_STREAM=1` and the dashboard's *Live
run* section shows a **reasoning + answer box** that fills token-by-token as the model decodes
the current question (alongside the decode-t/s chart). It works against any server that
supports streaming (`stream:true`) — unlike the t/s chart, which needs a ds4-style log — so it
also fills the live view on vLLM / llama.cpp. The runner streams the OpenAI SSE deltas, splits
chain-of-thought (a `reasoning_content` field or an inline `<think>…</think>` span) from the
answer, and assembles the *same* text the blocking call would, so **scoring is unchanged** — the
box is display-only and the default scoring path stays the plain blocking request (the flag is
opt-in because a few servers differ subtly between streamed and non-streamed output). It is
honored only at `BENCHY_CONCURRENCY=1` (the default); with `N>1` there is no single "current"
generation to show, so the box reports that the live stream is shown at concurrency 1 while the
t/s chart and per-question feed keep working.

## Reproducible snapshots — `api.py` + the lockfile

Other tools build on benchy as their benchmark source (e.g.
[forgequant](https://github.com/andreaborio/forgequant) calibrates quantization
imatrices on these sets). For that, benchmark data must be **pinned and verifiable**, and
benchy must expose a **stable contract** that won't break when internals change.

`api.py` is that contract. Consumers import only `api`, never `fetch_benchmarks` internals;
`api.API_VERSION` bumps on a breaking change.

```sh
python3 api.py status                 # lock state + upstream-drift for every set
python3 api.py prelock                 # pin upstream revisions (Hub refs only — no download)
python3 api.py lock <key|all>          # fetch + lock (revision + content SHA-256 + row count)
python3 api.py relock <key>            # accept upstream changes: re-pin + re-hash
python3 api.py verify <key|all>        # re-fetch the pinned revision, check the content hash
```

`benchmarks.lock.json` (tracked) records, per benchmark: the upstream HF dataset commit,
the content SHA-256 of the normalized rows, and the row count. `api.fetch(key)` fetches the
**pinned** revision and verifies the hash — so a result (or a calibration corpus) is always
tied to an exact snapshot, and any upstream drift is **detected, not silently absorbed**.
Benchmark rows themselves are never committed (`data/` is gitignored — see licenses in
`DATA.md`); only the lock travels with the repo, so everyone fetches the same thing.

```python
import api
path = api.fetch("humaneval")          # pinned + verified; fetches on demand
api.registry()                          # [{key,name,domain,tier,fit,license,present,locked}]
api.lock_status()                       # per-set drift vs upstream
```

## Benchmarks

Fetchable via `fetch_benchmarks.py` / the dashboard (MCQ sets normalised to
`{question, options{A..}, answer_idx}`; code sets executed for pass@1). Each is tagged
**current** (still discriminates strong mid-2026 models) or **legacy** (saturated — top
models near ceiling, useful only as a small/quantized-model regression check). Fetch the
recommended set with `python3 fetch_benchmarks.py current` or the dashboard's **⬇ Benchmarks**
(both go through the lock contract: each fetch pins the upstream revision + content hash into
`benchmarks.lock.json`, which the runners verify at run time — see
[Reproducible snapshots](#reproducible-snapshots--apipy--the-lockfile)).

**Current (recommended):**

| Benchmark        | Domain        | Fit       | Source (HF)                       |
|------------------|---------------|-----------|-----------------------------------|
| MMLU-Pro         | reasoning     | mcq (≤10) | `TIGER-Lab/MMLU-Pro`             |
| SuperGPQA        | reasoning     | mcq       | `m-a-p/SuperGPQA`                |
| MMLU — logic     | reasoning     | mcq (4)   | `cais/mmlu` (formal_logic)       |
| TruthfulQA (MC1) | truthfulness  | mcq       | `truthfulqa/truthful_qa`         |
| HumanEval        | code          | exec pass@1 | `openai/openai_humaneval`      |
| MBPP             | code          | exec pass@1 | `google-research-datasets/mbpp`|
| MedXpertQA (Text)| medical       | mcq (≤10) | `TsinghuaC3I/MedXpertQA`         |
| MedMCQA          | medical       | mcq (4)   | `openlifescienceai/medmcqa`      |
| MedQA (USMLE)    | medical       | mcq (4)   | `GBaker/MedQA-USMLE-4-options`   |

**Legacy / saturated** (fetchable, off by default): ARC-Challenge, HellaSwag, CommonsenseQA,
OpenBookQA, WinoGrande, MMLU-CS, PubMedQA, MMLU-medical.

### Adding a benchmark — a data PR, no code

The registry is **declarative**: every benchmark is a JSON file in `benchmarks/`, loaded
by `registry.py`. To add one, drop a file — no Python — describing where the rows come
from and how to map one row to `{question, options{A..}, answer_idx}`:

```jsonc
// benchmarks/my_bench.json
{
  "name": "My Benchmark", "domain": "reasoning", "tier": "current",
  "license": "… (verify)", "desc": "one line shown in the UI", "cap": 800,
  "source": [{"dataset": "org/dataset", "config": "default", "split": "test"}],
  "map": {
    "question": "question",                       // or {key, context, template}
    "options":  {"from": "list", "key": "choices"},
    "answer":   {"from": "index", "key": "answer"}
  }
}
```

`options.from`: `list` · `dict` · `labeled` · `keys` · `pair` · `fixed`.
`answer.from`: `index` · `letter` · `answerKey` · `map` · `match`. Dotted keys
(`mc1_targets.choices`) index nested fields. Only odd shapes (code execution, context
joins) need a Python `hook` instead of a `map`. The push CI validates new entries against
the live source. See the existing files in `benchmarks/` as templates.

**Health check.** `python3 healthcheck.py` probes one row of every benchmark and runs its
normalizer, so a renamed/changed upstream dataset is caught before it breaks a run; it also
flags lock drift. It runs in CI weekly and on every registry edit
(`.github/workflows/benchmarks-healthcheck.yml`). `python3 healthcheck.py --local` audits
the already-fetched files against `benchmarks.lock.json` offline (no network at all).

**Manual / gated:** **GPQA** Diamond (the frontier science discriminator — gated, needs a HF
token), **HLE** (gated, mostly free-form/multimodal), and **HealthBench** (rubric-graded, run
by `healthbench.py`). Relevant but **out of scope** for this harness (would need new runners):
SWE-bench & LiveCodeBench (agentic / stdin-stdout code), AIME/MATH/FrontierMath (numeric/proof
grading), SimpleQA & IFEval (judge / programmatic verifiers), ARC-AGI (grid program synthesis).

**Code benchmarks** (HumanEval, MBPP) are run by `eval_code.py`: the model writes a function
and `benchy` **executes it against the task's tests** to score pass@1.
> ⚠ This runs model-generated **and benchmark-supplied** code on your machine — each candidate
> in a separate process with a timeout (`BENCHY_CODE_TIMEOUT`, default 12s) but **no sandbox**
> (no filesystem/network isolation). It is therefore **off by default**: enable it deliberately
> with `BENCHY_ALLOW_CODE_EXEC=1` (set it in the dashboard's environment to allow code runs from
> the UI), and only for models/benchmarks you trust. (Agentic / repo-level sets like SWE-bench
> are out of scope.)

**HealthBench** is fetched and run separately by `healthbench.py` (rubric-graded by an
external judge — see Quickstart step 4). Rubric scores are a 0–100 rubric mean, **not**
percent correct, and are excluded from the MCQ macro-average. Gated sets (GPQA, HLE) need a
manual download (a HF token) — see **[DATA.md](DATA.md)**.

Sources, citations, and **per-dataset licenses** are in **[DATA.md](DATA.md)**. The datasets
are **not** redistributed here; `fetch_benchmarks.py` downloads them on demand.

## Methodology & caveats (read before quoting numbers)

`benchy` is built to be honest about its own limits:

- **Comparing two builds? Use the paired test.** Two independent runs’ Wilson CIs overlapping
  does **not** mean the builds are equal — on the same questions a few-point quant delta is
  often significant. Use the **Paired A/B significance (McNemar)** card, which pairs the runs
  question-by-question. For that to be valid, run both tags on the **same benchmark file** so
  they share questions (and keep `BENCHY_SHUFFLE_OPTIONS` at its default so both see the same
  per-question option order).
- **Answer extraction is heuristic.** The MCQ scorer reads the chosen letter from free text
  with an anchored, case-sensitive parser and records an **unparseable rate** per run — a
  rising unparseable rate is itself a quant-quality signal. It is robust but not perfect; for
  the cleanest measurement prefer a server with constrained/logprob decoding.
- **Position bias is mitigated, not eliminated.** Option order is randomized per question
  (seeded by the question text, so every model/quant sees the same order). Letter-position
  bias is also reported (χ²); residual bias on tiny N can still move a point or two.
- **Small-N noise.** Quick runs use small N; accuracies carry **Wilson 95% CIs**. Treat two
  results whose CIs overlap as indistinguishable.
- **Reference baselines.** `benchy` ships published **frontier-model scores** for several
  benchmarks (MMLU-Pro, GPQA, HumanEval, MBPP) in `references.json` — each with its source +
  date. They are **eval-setup-dependent** (CoT / shots / harness differ from `benchy`'s) and
  shown for **context, not size-matched head-to-head comparisons**. Edit or add your own per
  benchmark in **Setup** (your `config.json` overrides the shipped numbers by label).
- **Rubric scores ≠ accuracy.** HealthBench-style scores are the fraction of weighted rubric
  points met (0–100), graded by an external judge — not directly comparable to MCQ accuracy
  and excluded from the macro-average.
- **Live system metrics are macOS-only and host-level where labelled** (system memory/swap
  are whole-machine via `vm_stat`/`sysctl`; "model RSS" is the server process via `ps`). On
  Linux/Windows these panels are empty — the benchmarks and scoring are unaffected.
- **Decode t/s is parsed from a ds4-style server log** and is engine-specific (best-effort).
  For llama.cpp / vLLM / Ollama / LM Studio the format differs, so the decode-throughput card
  and per-question prefill/decode timing may show nothing. The accuracy results do not depend
  on it.
- **Determinism** assumes a greedy, temperature-0 server; `temperature=0` is **not** a
  guarantee — batched/quantized inference can still vary run-to-run, and results drift with
  thermal throttling, page-cache state, and server build. Prefer larger N (and the paired
  test) when a delta matters.
- **Code-execution benchmarks are off by default.** HumanEval/MBPP execute model-written
  *and* benchmark-supplied Python on your host with only a subprocess + timeout (no sandbox).
  Enable deliberately with `BENCHY_ALLOW_CODE_EXEC=1` and only for files you fetched yourself.

## Works with any OpenAI-compatible server

`benchy` only consumes the standard OpenAI API (`/v1/chat/completions`, `/v1/models`) for
benchmarking — it neither patches nor requires any specific engine, so **accuracy/scoring work
with any OpenAI-compatible server**. The *live decode-t/s* panel is the one exception: it
scrapes the server's stdout log in a **ds4-style format**, so for llama.cpp / vLLM / Ollama /
LM Studio that panel may stay empty (the benchmarks still run). `benchy`
was developed against **ds4** (a.k.a. *DwarfStar*), the DeepSeek-V4-Flash inference engine
by **Salvatore Sanfilippo ([antirez](https://github.com/antirez))**; the in-UI "Start server"
button falls back to a ds4 checkout (`DS4_DIR`) but you can point it at any command via
`config.json` (`"server": {"cmd": [...]}`). ds4 is a separate project under its own license —
please refer to and cite it directly.

## Acknowledgments

- **ds4 / DwarfStar** — Salvatore Sanfilippo (antirez) — the inference engine `benchy` was
  first built to exercise.
- The authors of each benchmark dataset (see **[DATA.md](DATA.md)** for papers and licenses)
  and of **HealthBench** (OpenAI).

## A note on the bundled medical datasets

Some bundled benchmarks (MedMCQA, PubMedQA, MMLU-medical, HealthBench) are medical. They are
**research/evaluation material only** — benchmark scores measure performance on static
question sets and must **not** be used for diagnosis, treatment, or any clinical decision.
Any model you evaluate remains the responsibility of its operator.

## License

Source code: **MIT** — see [LICENSE](LICENSE). The benchmark datasets and any inference engine
are **not** covered by this license and retain their own terms ([DATA.md](DATA.md)).
