# GlassQuill

**Explainable detection of AI-generated text.** A case study project for Applications and Use Cases of Machine Learning (TE7483), AIML Department, Symbiosis Institute of Technology, Pune.

GlassQuill labels a piece of English text as human-written or AI-generated and shows why. It compares four models, tests them on other LLMs and on paraphrased text and explains every prediction with SHAP and LIME.

**Start here:** read the full project guide in [`docs/GlassQuill_Project_Guide.pdf`](docs/GlassQuill_Project_Guide.pdf). It covers the background, datasets, method, step-by-step build, deployment, rubric mapping and viva prep.

## Models

| Model | Input | Explained with |
|---|---|---|
| Perplexity threshold | GPT-2 log perplexity | the score itself |
| TF-IDF + logistic regression | word and word-pair weights | exact word contributions |
| XGBoost (main model) | 18 style and perplexity features | SHAP |
| DistilBERT | raw text | LIME and SHAP text |

## Data

- **HC3** (`Hello-SimpleAI/HC3`): human and ChatGPT answers to the same questions. Main train, validation and test data.
- **MAGE** (`yaful/MAGE`, test split): texts from many LLMs and domains. Out-of-domain test.

## Repo layout

```
app.py               Gradio app, also the Hugging Face Space entry point
streamlit_app.py     optional Streamlit front end
notebook_cells.py    every Colab cell in order (copy them into a notebook)
run_pipeline.py      the same steps as one resumable local script
requirements.txt
docs/                the project guide (PDF)
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
python run_pipeline.py --quick                        # 5-minute smoke test, writes to */quick/
python run_pipeline.py                                # full study (about 45 min on an RTX 3050)
python run_pipeline.py --stages ood explain           # rerun only some stages
python app.py                                         # Gradio demo at http://127.0.0.1:7860
```

Stages: `data → features → train → bert → evaluate → ood → paraphrase → explain`. Every table ends up in `reports/` (start with [`reports/RESULTS.md`](reports/RESULTS.md)) and every plot in `reports/figures/`. The trained XGBoost, TF-IDF + LR and perplexity threshold go in `models/`. DistilBERT goes in `models/distilbert/`, which is not committed (about 260 MB).

## Results (full run, 25 October 2026, seed 42)

**In-domain, HC3 test** (1,795 texts, split by question, every source 50/50 human/AI)

| Model | Accuracy | F1 | ROC-AUC | FPR |
|---|---|---|---|---|
| Perplexity threshold | 0.974 | 0.974 | 0.993 | 0.031 |
| TF-IDF + LR | 0.974 | 0.975 | 0.997 | 0.016 |
| **XGBoost (main)** | **0.987** | **0.987** | 0.999 | **0.014** |
| DistilBERT | 0.983 | 0.983 | 1.000 | 0.032 |

**Out-of-domain, MAGE** (1,000 human + 1,000 AI texts, 14 domains, 12 generator families)

| Model | Accuracy | F1 | ROC-AUC | FPR |
|---|---|---|---|---|
| Perplexity threshold | 0.634 | 0.466 | 0.681 | 0.052 |
| TF-IDF + LR | 0.561 | 0.258 | 0.582 | 0.031 |
| **XGBoost** | **0.642** | 0.453 | **0.803** | **0.014** |
| DistilBERT | 0.642 | 0.512 | 0.693 | 0.090 |

**Paraphrased AI text** (150 HC3 test AI texts rewritten with T5): recall before → after

| Perplexity | TF-IDF + LR | XGBoost | DistilBERT |
|---|---|---|---|
| 0.99 → 0.74 | 0.95 → 0.75 | 0.99 → 0.77 | 1.00 → 0.97 |

**Explanations:** SHAP vs LIME top-10 word overlap for DistilBERT: Jaccard 0.14. LIME finds 61% of the true top-10 words of the logistic regression.

What this says, in short:
- **RQ1:** the GPT-2 features separate the classes best (log perplexity, top-10 share, nll_std: effect sizes around 1.7 SD). Style features help less. One surprise: `mean_entropy` pushes towards AI, the opposite of the guide's expectation.
- **RQ2:** the explainable XGBoost matches or beats DistilBERT in-domain and has the lowest FPR on every test set.
- **RQ3 / H3:** all models lose about 35 points on MAGE. They still catch GPT-3.5, GPT-4 and davinci (55–79%) but almost never catch small base models such as OPT, BLOOM and GPT-J (0–6%), whose sampled text has human-like perplexity. See `reports/ood_accuracy_by_generator.csv`.
- **H4:** paraphrasing lowers recall for every model. The perplexity-based models lose 22–25 points and DistilBERT only 3.
- **RQ4:** SHAP and LIME barely agree on DistilBERT. LIME is only partly reliable even when the true answer is known.

## Data issues found and fixed (beyond section 3.1 of the guide)

The guide's rule is "above 99% accuracy, look for a shortcut". The first full run hit 99.3%, and checking turned up these problems:

1. **Source imbalance.** After the 50–400 word filter, `open_qa` had 20 human vs 764 AI texts and finance 530 vs 1,024, so a model could learn topic instead of style. `prepare()` now samples the same number of human and AI texts within every source (`balance_by="source"`; pass `None` for the guide's original sampling).
2. **Tokenizer leftovers in human answers only.** HC3 human answers were word-tokenized and the ChatGPT ones were not. In human texts: `do n't` / `it 's` (34% of human texts vs 0.015% of AI), `( word )` (35% vs 0.01%), `URL_0` link placeholders (11.6% vs 0.07%), ` -- ` and `' word '`. These were free giveaways for TF-IDF and DistilBERT, and they also raised GPT-2 perplexity for human text. `clean_text()` now undoes all of them, which applies to MAGE too.

After both fixes every model is under 99%. The perplexity baseline dropped from 98.7% to 97.4%.

## Status

- [x] Project guide
- [x] Starter code (tested end to end on dummy data, October 2026)
- [x] Run on HC3 and record results
- [x] Robustness tests and explanation analysis
- [ ] Deploy the app (runs locally; the HF Space is not created yet)
- [ ] Report and slides

*Study project. A detector score should never be the only evidence that someone used AI.*
