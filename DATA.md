# Benchmark data — sources, citations, licenses

`benchy` does **not** redistribute any benchmark data. `fetch_benchmarks.py` downloads each
set from the [Hugging Face datasets server](https://huggingface.co/docs/datasets-server) and
normalises it to the unified MCQ format `{question, options{A..}, answer_idx}` under `data/`
(which is git-ignored). HealthBench, MedQA and MedXpertQA come from their own upstreams.

**Each dataset is governed by its own license and terms — review them upstream before using
or redistributing.** The notes below are a starting point, not legal advice; verify the
current license on each source. Some sets derive from exams or clinical material and may
carry additional restrictions.

## Fetchable via `fetch_benchmarks.py`

Tier is **current** (discriminates mid-2026 models) or **legacy** (saturated — regression only).

| Benchmark | Tier | Domain | Source (HF) | Paper | License (verify upstream) |
|---|---|---|---|---|---|
| MMLU-Pro | current | reasoning | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | Wang et al. 2024, arXiv:2406.01574 | MIT (verify) |
| SuperGPQA | current | reasoning | [m-a-p/SuperGPQA](https://huggingface.co/datasets/m-a-p/SuperGPQA) | M-A-P et al. 2025, arXiv:2502.14739 | ODC-BY (verify) |
| MMLU — formal logic | current | reasoning | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | Hendrycks et al., ICLR 2021, arXiv:2009.03300 | MIT |
| TruthfulQA (MC1) | current | truthfulness | [truthfulqa/truthful_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa) | Lin et al., ACL 2022, arXiv:2109.07958 | Apache-2.0 (verify) |
| HumanEval | current | code (exec) | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) | Chen et al. 2021, arXiv:2107.03374 | MIT (verify) |
| MBPP (sanitized) | current | code (exec) | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) | Austin et al. 2021, arXiv:2108.07732 | CC-BY-4.0 (verify) |
| MedXpertQA (Text) | current | medical | [TsinghuaC3I/MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) | Zuo et al., ICML 2025, arXiv:2501.18362 | verify; may be non-commercial |
| MedMCQA | current | medical | [openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa) | Pal et al., CHIL 2022 | MIT (verify) |
| MedQA (USMLE) | current | medical | [GBaker/MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) | Jin et al. 2020, arXiv:2009.13081 | research use (verify) |
| ARC-Challenge | legacy | reasoning | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | Clark et al. 2018, arXiv:1803.05457 | CC-BY-SA-4.0 (verify) |
| HellaSwag | legacy | commonsense | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) | Zellers et al., ACL 2019, arXiv:1905.07830 | MIT (verify) |
| CommonsenseQA | legacy | commonsense | [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa) | Talmor et al., NAACL 2019, arXiv:1811.00937 | MIT (verify) |
| WinoGrande | legacy | commonsense | [allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande) | Sakaguchi et al. 2019, arXiv:1907.10641 | CC-BY (verify) |
| OpenBookQA | legacy | knowledge | [allenai/openbookqa](https://huggingface.co/datasets/allenai/openbookqa) | Mihaylov et al., EMNLP 2018, arXiv:1809.02789 | Apache-2.0 (verify) |
| MMLU — CS cluster | legacy | knowledge | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | Hendrycks et al., ICLR 2021, arXiv:2009.03300 | MIT |
| PubMedQA | legacy | medical | [qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) | Jin et al., EMNLP 2019, arXiv:1909.06146 | MIT (verify) |
| MMLU — medical cluster | legacy | medical | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | Hendrycks et al., ICLR 2021, arXiv:2009.03300 | MIT |

`MMLU — CS cluster` = college_computer_science, high_school_computer_science, machine_learning.
`MMLU — medical cluster` = anatomy, clinical_knowledge, college_biology, college_medicine,
medical_genetics, professional_medicine. (`fetch_benchmarks.py current` grabs the current tier.)

## Manual / gated

These are not pulled by `fetch_benchmarks.py` (gated, or a different runner):

| Benchmark | Source | Paper | Notes |
|---|---|---|---|
| GPQA (Diamond) | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) | Rein et al. 2023, arXiv:2311.12022 | gated — accept terms / use a HF token. The frontier science MCQ. |
| Humanity's Last Exam | [cais/hle](https://huggingface.co/datasets/cais/hle) | Phan et al. 2025, arXiv:2501.14249 | gated; mostly free-form/multimodal — reference only |
| HealthBench (Hard / Consensus) | [openai/healthbench](https://huggingface.co/datasets/openai/healthbench) (data) · [openai/simple-evals](https://github.com/openai/simple-evals) (method) | OpenAI, 2025 | rubric-graded; fetched & run by `healthbench.py` |

## Code execution & out-of-scope benchmarks

HumanEval and MBPP are **executed** (`eval_code.py`: generate → run unit tests → pass@1). Other
2026-relevant benchmarks need runners this suite intentionally does not have: **SWE-bench**
(agentic/repo-level + Docker), **LiveCodeBench / BigCodeBench** (stdin-stdout or extra-deps
execution), **AIME / MATH / FrontierMath** (numeric/symbolic grading), **SimpleQA / IFEval**
(LLM-judge / programmatic verifiers), **ARC-AGI** (grid program synthesis). These are tracked as
external references, not run here.

## Citations

```bibtex
@inproceedings{hendrycks2021mmlu,
  title={Measuring Massive Multitask Language Understanding},
  author={Hendrycks, Dan and others}, booktitle={ICLR}, year={2021}
}
@inproceedings{wang2024mmlupro,
  title={MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark},
  author={Wang, Yubo and others}, booktitle={NeurIPS}, year={2024}
}
@article{clark2018arc,
  title={Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge},
  author={Clark, Peter and others}, journal={arXiv:1803.05457}, year={2018}
}
@inproceedings{zellers2019hellaswag,
  title={HellaSwag: Can a Machine Really Finish Your Sentence?},
  author={Zellers, Rowan and others}, booktitle={ACL}, year={2019}
}
@inproceedings{talmor2019commonsenseqa,
  title={CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge},
  author={Talmor, Alon and others}, booktitle={NAACL}, year={2019}
}
@article{sakaguchi2019winogrande,
  title={WinoGrande: An Adversarial Winograd Schema Challenge at Scale},
  author={Sakaguchi, Keisuke and others}, journal={arXiv:1907.10641}, year={2019}
}
@inproceedings{mihaylov2018openbookqa,
  title={Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering},
  author={Mihaylov, Todor and others}, booktitle={EMNLP}, year={2018}
}
@inproceedings{lin2022truthfulqa,
  title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
  author={Lin, Stephanie and Hilton, Jacob and Evans, Owain}, booktitle={ACL}, year={2022}
}
@inproceedings{pal2022medmcqa,
  title={MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering},
  author={Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
  booktitle={CHIL}, year={2022}
}
@inproceedings{jin2019pubmedqa,
  title={PubMedQA: A Dataset for Biomedical Research Question Answering},
  author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and Lu, Xinghua},
  booktitle={EMNLP}, year={2019}
}
@article{jin2020medqa,
  title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
  author={Jin, Di and others}, journal={arXiv:2009.13081}, year={2020}
}
@inproceedings{zuo2025medxpertqa,
  title={MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding},
  author={Zuo, Yuxin and others}, booktitle={ICML}, year={2025}
}
@misc{openai2025healthbench,
  title={HealthBench}, author={OpenAI}, year={2025},
  howpublished={\url{https://openai.com/index/healthbench/}}
}
```

## Reference baselines (`references.json`)

`benchy` ships published **frontier-model scores** for MMLU-Pro, GPQA (Diamond), HumanEval and
MBPP in `references.json`, shown on the Accuracy chart for context. Each entry carries its
source + date (e.g. Artificial Analysis, Epoch AI, the DeepSeek-R1 / Qwen2.5-Coder reports,
Meta Llama model cards, OpenAI/Anthropic launch tables — gathered 2026-06). **These numbers are
eval-setup-dependent** (CoT vs 0-shot, EvalPlus vs plain pass@1, etc. — different from
`benchy`'s own harness) and are **not** size-matched comparisons; verify each at its source.
Add or override baselines per benchmark in the dashboard's **Setup** panel — your (git-ignored)
`config.json` overrides the shipped numbers by label.
