# GlassQuill

**Explainable detection of AI-generated text.** A case study project for Applications and Use Cases of Machine Learning (TE7483), AIML Department, Symbiosis Institute of Technology, Pune.

GlassQuill labels a piece of English text as human-written or AI-generated and shows why. It compares four models, tests them on other LLMs and on paraphrased text and explains every prediction with SHAP and LIME.

**Start here:** the project report is [`docs/REPORT.md`](docs/REPORT.md) and the viva sheet is [`docs/VIVA.md`](docs/VIVA.md). The original build guide is [`docs/GlassQuill_Project_Guide.pdf`](docs/GlassQuill_Project_Guide.pdf).

## Models

| Model | Input | Explained with |
|---|---|---|
| Perplexity threshold | GPT-2 log perplexity | the score itself |
| TF-IDF + logistic regression | word and word-pair weights | exact word contributions |
| XGBoost | 18 style and perplexity features, monotone constraints | SHAP |
| DistilBERT | raw text | LIME and SHAP text |
| **App verdict** | mean of XGBoost and TF-IDF + LR | both of the above |

## Data

- **HC3** (`Hello-SimpleAI/HC3`): human and ChatGPT (2022) answers to the same questions.
- **HAP-E** (`browndw/human-ai-parallel-corpus`): human text in 6 genres, continued by GPT-4o, GPT-4o-mini and Llama-3 Instruct.
- **M4GT** (from `Jinyan1/COLING_2025_MGT_en`, M4GT part only): 6 domains, with GPT-4o, GPT-4, GPT-3.5-turbo, Mixtral, Llama-3, Gemma and Cohere.
- **MAGE** (`yaful/MAGE`, test split): 27 generators, 14 domains. Out-of-domain test only, never trained on.

32k texts in total, with equal human and AI counts in each of 17 domains, split by question or document (70/15/15).

## Repo layout

```
app.py               Gradio app, also the Hugging Face Space entry point
streamlit_app.py     optional Streamlit front end
notebook_cells.py    every Colab cell in order (copy them into a notebook)
run_pipeline.py      the same steps as one resumable local script
requirements.txt
docs/                report, viva sheet, project guide (PDF)
src/                 data, features, models, DistilBERT, explanations, detector
models/              saved models go here after training
reports/             result tables, RESULTS.md and the run log
reports/figures/     plots for the report
```

## Quick start (Google Colab)

1. Runtime > Change runtime type > T4 GPU.
2. Add a GitHub token as the Colab secret `GITHUB_TOKEN` (the repo is private, see section 7.3 of the guide).
3. Run:

```python
from google.colab import userdata
token = userdata.get("GITHUB_TOKEN")
!git clone https://{token}@github.com/ThatOneRahul/glassquill.git
%cd glassquill
!pip install -q -r requirements.txt
```

4. Copy the cells from `notebook_cells.py` into the notebook and run them in order.

## Run it locally (any machine with a CUDA GPU, or slowly on CPU)

`run_pipeline.py` runs every step of section 7 of the guide in order. Each stage saves its output to disk, so you can rerun a single stage without starting over.

```bash
python -m venv .venv && .venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt gradio
python run_pipeline.py --quick                        # ~10-minute smoke test on a small sample, writes to */quick/
python run_pipeline.py --stages evaluate ood paraphrase explain   # results only, ~10 min, reuses saved features
python run_pipeline.py                                # full study from scratch, ~1.5 h on an RTX 3050 (feature extraction for 32k texts)
python app.py                                         # Gradio demo at http://127.0.0.1:7860
```

Stages: `data → features → train → bert → evaluate → ood → paraphrase → explain`. Every table ends up in `reports/` (start with [`reports/RESULTS.md`](reports/RESULTS.md)) and every plot in `reports/figures/`. The trained XGBoost, TF-IDF + LR and perplexity threshold are committed in `models/`, so the app runs without retraining. DistilBERT (`models/distilbert/`, ~260 MB) is not committed.

## Results (seed 42)

**In-domain test** (4,829 texts from HC3 + HAP-E + M4GT, never seen in training)

| Model | Accuracy | F1 | ROC-AUC | FPR |
|---|---|---|---|---|
| Perplexity threshold | 0.839 | 0.827 | 0.902 | 0.093 |
| TF-IDF + LR | 0.959 | 0.959 | 0.990 | 0.030 |
| XGBoost | 0.923 | 0.923 | 0.978 | 0.066 |
| DistilBERT | 0.961 | 0.963 | 0.997 | 0.071 |
| **App verdict (XGBoost + LR)** | **0.974** | **0.974** | 0.995 | **0.020** |

The app's rule (the mean of the two explainable models) was chosen on the validation set (0.972 accuracy, 0.021 FPR) before the test set was looked at.

**Out-of-domain, MAGE** (2,000 texts, 27 generators we never trained on)

| Model | Accuracy | F1 | ROC-AUC | FPR |
|---|---|---|---|---|
| Perplexity threshold | 0.656 | 0.551 | 0.681 | 0.111 |
| TF-IDF + LR | 0.606 | 0.426 | 0.648 | 0.079 |
| XGBoost | 0.634 | 0.532 | 0.611 | 0.149 |
| DistilBERT | 0.655 | 0.585 | 0.696 | 0.176 |
| **App verdict** | **0.664** | 0.542 | 0.655 | **0.068** |

The app verdict catches GPT-4 (90%) and GPT-3.5 (80%) on MAGE, but almost never small base models such as OPT, BLOOM and GPT-J (0-4%). Their sampled text has human-like perplexity. See `reports/ood_accuracy_by_generator.csv`.

**Paraphrased AI text** (60 HC3 test AI texts rewritten sentence by sentence with T5): recall before → after

| Perplexity | TF-IDF + LR | XGBoost | DistilBERT | App verdict |
|---|---|---|---|---|
| 1.00 → 0.93 | 0.93 → 0.77 | 1.00 → 0.95 | 1.00 → 1.00 | 0.98 → 0.93 |

**Explanations:** the top SHAP features for XGBoost are log perplexity, `nll_std`, average word length and `top1_frac` (`reports/figures/shap_beeswarm.png`). SHAP and LIME barely agree on DistilBERT (top-10 Jaccard 0.06). LIME recovers 51% of the true top-10 words of the logistic regression, so it is only partly reliable.

## What changed after the first version

1. **Modern AI text.** Trained on HC3 alone, the model labelled a current ChatGPT paragraph as 100% human. It had only seen 2022 ChatGPT. We added HAP-E and M4GT (GPT-4o, GPT-4, Llama-3, Mixtral, Gemma and more), and the same paragraph now scores 99% AI.
2. **Monotone constraints in XGBoost.** A feature may only push in the direction both theory and data agree on (e.g. lower burstiness can only push towards AI). The HC3-only model had learned "very even sentences mean human".
3. **Shortcut removal.** The first run scored 99.3%, which the guide says to distrust. HC3 human answers were word-tokenized (`do n't` in 34% of human vs 0.02% of AI texts, `( x )`, `URL_0`). Sources were also unbalanced (open_qa: 20 human vs 764 AI), and M4GT AI texts carried markdown (`**`, `#`). `clean_text()` removes all of these, and `prepare()` balances human and AI within each of the 17 domains.
4. **Tried and rejected:** a modern reader LM (Qwen2.5-0.5B) and Binoculars instead of GPT-2. They added only +0.01 to +0.04 AUC and were 25× slower.

## Rubric map (ESE, 30 marks)

| Component | Marks | Where |
|---|---|---|
| Problem refinement & use case | 3 | `docs/REPORT.md` §1 |
| Dataset, preprocessing, features | 4 | §3, `reports/eda_source_counts.csv`, `feature_means_by_class.csv`, `figures/length_hist.png` |
| Methodology & pipeline | 5 | §4 (pipeline diagram, model justification, tuning) |
| Implementation | 5 | `src/`, `run_pipeline.py`, `app.py`, run logs in `reports/` |
| Results & comparison | 5 | §6, `reports/RESULTS.md`, confusion matrices, ROC, SHAP figures |
| Report | 3 | `docs/REPORT.md` |
| Presentation | 2 | slides (to do) + live demo |
| Viva | 3 | `docs/VIVA.md` |

## Status

- [x] Data, features, four models, evaluation on three test conditions
- [x] Explanations (SHAP, LIME, word weights) and agreement checks
- [x] App working locally
- [x] Report and viva sheet
- [ ] Slides
- [ ] Hugging Face Space deployment

*Study project. A detector score should never be the only evidence that someone used AI.*
