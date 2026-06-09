# Benchmark data — sources, citations, licenses

`benchy` does **not** redistribute any benchmark data. `fetch_benchmarks.py` downloads
the multiple-choice sets from the [Hugging Face datasets server](https://huggingface.co/docs/datasets-server)
and normalises them to `{question, options{A..}, answer_idx}` under `data/` (which is
git-ignored). HealthBench and MedXpertQA come from their upstreams.

**Each dataset is governed by its own license and terms — review them upstream before
using or redistributing.** The notes below are a starting point, not legal advice;
verify the current license on each source. Use of clinical-exam–derived material may be
subject to additional restrictions.

| Benchmark | Source | Paper | License (verify upstream) |
|---|---|---|---|
| MedQA (USMLE) | `bigbio/med_qa` / [jind11/MedQA](https://github.com/jind11/MedQA) | Jin et al. 2020, arXiv:2009.13081 | research use; see source |
| MedMCQA | [openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa) · [medmcqa.github.io](https://medmcqa.github.io) | Pal et al., CHIL 2022 | MIT (verify) |
| MMLU (medical subsets) | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | Hendrycks et al., ICLR 2021, arXiv:2009.03300 | MIT |
| PubMedQA | [pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa) | Jin et al., EMNLP 2019, arXiv:1909.06146 | MIT |
| MedXpertQA | [TsinghuaC3I/MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) | Zuo et al., ICML 2025, arXiv:2501.18362 | see source (verify; may be non-commercial) |
| HealthBench (Hard) | [openai/simple-evals](https://github.com/openai/simple-evals) | OpenAI, 2025 | see OpenAI release terms |

MMLU "medical subsets" = anatomy, clinical_knowledge, college_medicine, college_biology,
medical_genetics, professional_medicine.

## Citations

```bibtex
@article{jin2020medqa,
  title={What Disease does this Patient Have? A Large-scale Open Domain Question
         Answering Dataset from Medical Exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and
          Fang, Hanyi and Szolovits, Peter},
  journal={arXiv:2009.13081}, year={2020}
}
@inproceedings{pal2022medmcqa,
  title={MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical
         domain Question Answering},
  author={Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
  booktitle={CHIL}, year={2022}
}
@inproceedings{hendrycks2021mmlu,
  title={Measuring Massive Multitask Language Understanding},
  author={Hendrycks, Dan and others}, booktitle={ICLR}, year={2021}
}
@inproceedings{jin2019pubmedqa,
  title={PubMedQA: A Dataset for Biomedical Research Question Answering},
  author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and
          Lu, Xinghua},
  booktitle={EMNLP}, year={2019}
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

Reference baselines shown in the dashboard credit their own reports — e.g. the
MedGemma technical report (arXiv:2507.05201) and model system cards. Numbers marked
"≈" / "approx" in-app are not exact measurements.
