# benchy

**A multi-benchmark test suite and live dashboard for local medical LLMs.**

`benchy` runs a panel of medical question-answering benchmarks against an
OpenAI-compatible inference server, scores them deterministically, and shows the
results — accuracy, confidence intervals, per-question drill-down, and live system
metrics — in a single-file web dashboard. It was built for and is tested with
[**ds4 / DwarfStar**](#built-for-ds4) serving DeepSeek-V4-Flash on Apple Silicon,
but the benchmark runners talk plain OpenAI Chat Completions and work against any
compatible endpoint.

> ⚕️ **Research / evaluation tool only — not a medical device.** See the
> [disclaimer](#medical-disclaimer).

---

## Highlights

- **6 medical benchmarks** out of the box — 5 multiple-choice (MedQA-USMLE, MedMCQA,
  MMLU-medical, PubMedQA, MedXpertQA) + **HealthBench Hard** (OpenAI rubric-graded).
- **Deterministic scoring** — greedy decoding, fixed seed, letter-bias (χ²) checks,
  Wilson 95% confidence intervals.
- **Live dashboard** — per-question feed with drill-down, running accuracy, accuracy
  vs. published reference baselines, and live host/server metrics (model RSS, decode
  t/s, system memory & swap).
- **`think` / `no-think` modes** — toggles ds4's reasoning so you can measure the
  test-time-reasoning delta.
- **Data-harvesting tools** — mine the questions the model got wrong (for fine-tuning
  sets) and assemble an importance-matrix calibration corpus from real eval traffic.
- **Zero pip dependencies** for the core — Python standard library only. (HealthBench
  grading needs one API key; see below.)

## How it works

```
              ┌────────────────┐   OpenAI /v1/chat/completions   ┌──────────────────┐
   benchy ───▶│  eval_mcq.py   │────────────────────────────────▶│   ds4-server     │
   runners    │  healthbench.py│◀────────────────────────────────│  (:8000)         │
              └───────┬────────┘   completions                    │  DeepSeek-V4-... │
                      │ writes                                     └────────┬─────────┘
                      ▼                                                     │ server.log
              results/{runs,details,stream,metrics}.jsonl ◀────────────────┘ (live timing)
                      │
                      ▼
              dashboard.py (:8050) ── live web UI
```

- The **runners** send each question to the server and score the reply. MCQ runners
  parse a single answer letter; HealthBench sends the conversation, takes the model's
  answer, and grades it against its rubric with an external judge model.
- The **dashboard** reads only files under `results/` plus the server's `server.log`
  for live decode-speed/timing. It does not modify the model or the server.

`benchy` does **not** patch or require any change to ds4 — it uses the server's
standard API. (The reasoning toggle is sent as a `think` field, which ds4 understands.)

## Quickstart

**Requirements:** Python 3.10+ (standard library only). A running OpenAI-compatible
server on `:8000` — e.g. `ds4-server`. For HealthBench grading, an OpenAI API key.

```sh
# 1. Fetch the benchmark datasets (downloads to data/, see DATA.md for sources/licenses)
python3 fetch_benchmarks.py

# 2. Start your model server on :8000 (example: ds4)
ds4-server -m model.gguf --ssd-streaming --ssd-streaming-cache-experts 40GB --ctx 16384 --port 8000

# 3. Run a multiple-choice benchmark
#    eval_mcq.py <data.jsonl> <N> [think|nothink] [tag] [notes...]
python3 eval_mcq.py data/medqa_test.jsonl 60 think iq2-baseline

# 4. (optional) Run HealthBench Hard — rubric-graded, needs an API key
echo "sk-..." > .apikey            # OpenAI key, used only for the grader; chmod 600
python3 healthbench.py 20 iq2-baseline think hard

# 5. Open the live dashboard
python3 dashboard.py 8050          # → http://localhost:8050
```

The dashboard can also start/stop the server and launch runs from the browser.

## Benchmarks

| Benchmark      | Type            | Options | Metric                    |
|----------------|-----------------|:-------:|---------------------------|
| MedQA (USMLE)  | clinical MCQ    | 4       | accuracy                  |
| MedMCQA        | medical MCQ     | 4       | accuracy                  |
| MMLU-medical   | medical MCQ     | 4       | accuracy                  |
| PubMedQA       | abstract QA     | 3       | accuracy                  |
| MedXpertQA     | expert MCQ      | 10      | accuracy                  |
| HealthBench Hard | open-ended    | —       | rubric score (0–100), not % correct |

Sources, citations, and **per-dataset licenses** are in **[DATA.md](DATA.md)**. The
datasets are **not** redistributed here. `fetch_benchmarks.py` auto-downloads
**MedMCQA, PubMedQA, and MMLU-medical** from the Hugging Face datasets server;
**MedQA, MedXpertQA, and HealthBench** are obtained from their own upstreams (see DATA.md).

## Methodology & caveats (read before quoting numbers)

`benchy` is built to be honest about its own limits:

- **Small-N noise.** Quick runs use small N; accuracies carry **Wilson 95% CIs**. Treat
  two results whose CIs overlap as indistinguishable.
- **HealthBench is a rubric score, not percent-correct.** It is the fraction of weighted
  rubric points met (0–100), graded by an external judge model — so it is not directly
  comparable to MCQ accuracy and is **excluded from the MCQ macro-average**. Grading
  costs API calls and the harness does not restrict prompt language.
- **Reference baselines are external published numbers**, cited per benchmark; some
  (e.g. on saturated benchmarks) are explicitly marked approximate. They are shown for
  **context, not as size-matched head-to-head comparisons** — a small local model and a
  large frontier model are different weight classes.
- **Live system metrics are host-level where labelled** (system memory/swap are
  whole-machine; "model RSS" is the server process). Decode t/s is parsed from
  `ds4-server`'s log format and is engine-specific.
- **Determinism** assumes a greedy, temperature-0 server; results can still drift with
  thermal throttling, page-cache state, and server build.

## Built for ds4

`benchy` was developed alongside **ds4** (a.k.a. *DwarfStar*), the narrow
DeepSeek-V4-Flash inference engine by **Salvatore Sanfilippo
([antirez](https://github.com/antirez))**. ds4 is a separate project under its own
license; `benchy` only consumes its OpenAI-compatible API and log output. All credit for
the inference engine belongs to its authors — please refer to and cite the ds4 project
directly. `benchy` neither vendors nor modifies it.

## Acknowledgments

- **ds4 / DwarfStar** — Salvatore Sanfilippo (antirez) — the inference engine this suite
  was built to exercise.
- The authors of each benchmark dataset (MedQA, MedMCQA, MMLU, PubMedQA, MedXpertQA) and
  of **HealthBench** (OpenAI) — see [DATA.md](DATA.md) for papers and licenses.
- Published reference scores credit their respective reports (e.g. the MedGemma technical
  report, arXiv:2507.05201, and model system cards), cited in-app and in `DATA.md`.

## Medical disclaimer

`benchy` is a **research and model-evaluation tool**. It is **not** a medical device and
its outputs must **not** be used for diagnosis, treatment, or any clinical
decision-making. Benchmark scores measure performance on static question sets and do not
establish clinical safety or efficacy. Any model evaluated here remains the
responsibility of its operator.

## License

Source code: **MIT** — see [LICENSE](LICENSE). The benchmark datasets and the ds4 engine
are **not** covered by this license and retain their own terms ([DATA.md](DATA.md)).
