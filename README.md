# GlassQuill

**Explainable detection of AI-generated text.** Paste an English text and GlassQuill tells you whether it reads as human-written or AI-generated, and shows the evidence behind the verdict: which writing signals pushed it each way, and which words.

![Python 3.11](https://img.shields.io/badge/python-3.11-3776ab)
![React 19](https://img.shields.io/badge/react-19-61dafb)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Most AI-text detectors return a bare score, and a wrong flag can hurt a real person. GlassQuill is built around two goals: a **low false positive rate** (humans wrongly flagged as AI) and a **reason for every verdict** that a person can check.

## Features

- **Verdict with an "unclear" band:** AI, human, or unclear when the evidence is weak.
- **Reasons, not just a score:** SHAP shows how much each of 18 writing signals moved the verdict, compared with typical human and AI values.
- **Word evidence:** colours the words that pushed towards human or AI.
- **Published results:** every model's metrics, the out-of-domain test and the paraphrase test, straight from the evaluation files.
- **Private by design:** pasted text is never stored or logged.

## Results

On 4,829 held-out texts (HC3, HAP-E and M4GT, split by question or document so no topic appears in both training and test):

| Model | Accuracy | F1 | ROC-AUC | False positive rate |
|---|---|---|---|---|
| Perplexity threshold (GPT-2) | 0.839 | 0.827 | 0.902 | 9.3% |
| TF-IDF + logistic regression | 0.959 | 0.959 | 0.990 | 3.0% |
| XGBoost (18 features) | 0.923 | 0.923 | 0.978 | 6.6% |
| DistilBERT (fine-tuned) | 0.961 | 0.963 | 0.997 | 7.1% |
| **GlassQuill** (mean of XGBoost and TF-IDF + LR) | **0.974** | **0.974** | 0.995 | **2.0%** |

- **Unseen generators:** on MAGE (27 generators that were never used in training), GlassQuill reaches 66.4% accuracy at a 6.8% false positive rate and still catches GPT-4 91% of the time. Small open models (OPT, BLOOM, GPT-J) mostly go undetected.
- **Paraphrasing:** after T5 rewrites every sentence, GlassQuill still catches 93% of AI texts (98% before).

All tables are in [`reports/RESULTS.md`](reports/RESULTS.md); the full write-up is in [`docs/REPORT.md`](docs/REPORT.md), and the final report (PDF) and slides are in [`submissions/`](submissions/). The presentation is also [on Canva](https://canva.link/awpmcoj5tqs8av2).

## Quick start

Requirements: Python 3.11 and Node.js 20 or newer. An NVIDIA GPU is optional; on a CPU each text takes about 2 seconds.

```bash
git clone https://github.com/ThatOneRahul/glassquill.git
cd glassquill
python -m venv .venv
.venv/Scripts/activate              # macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CPU only: pip install torch
pip install -r requirements.txt
cd web && npm install && npm run build && cd ..
python server.py                    # open http://127.0.0.1:8000
```

The trained models ship in `models/`, so no training is needed to run the app. GPT-2 downloads from Hugging Face on first start.

For frontend development, run `python server.py` and, in a second terminal, `cd web && npm run dev` (http://localhost:5173, with the API proxied).

## How it works

```
HC3 + HAP-E + M4GT ─► clean ─► balance per domain ─► split by group (70/15/15)
                                                            │
             ┌──────────────────────────────────────────────┴──────────────┐
      18 features per text                                             raw text
 (6 GPT-2 surprise + 12 writing style)                                     │
             │                                                ┌─────────────┴────────────┐
   Perplexity threshold, XGBoost                          TF-IDF + LR                DistilBERT
             └───────── verdict = mean(XGBoost, TF-IDF + LR) ─┘
                                   │
              explained with SHAP (features) and word weights (words)
```

- **Data:** 32,000 texts across 17 domains, with equal human and AI counts in each domain. They cover ChatGPT (2022) and newer models: GPT-4o, GPT-4, Llama-3, Mixtral, Gemma and others.
- **Cleaning:** removes formatting shortcuts that would let a model cheat. For example, HC3's human answers are tokenized (`do n't`, `URL_0`) and its AI answers are not.
- **Features:** GPT-2 measures how predictable each word is (perplexity). AI text is more predictable. Twelve style features cover sentence rhythm (burstiness), vocabulary richness, punctuation and readability.
- **Training:** hyper-parameters are tuned with 5-fold GroupKFold on the training set only. XGBoost uses monotone constraints where theory and data agree, and the test set is used once, at the end.
- **Explanations:** TreeSHAP for XGBoost, exact word weights for the linear model, and LIME for DistilBERT.

A file-by-file guide to the code is in [`docs/CODE_WALKTHROUGH.md`](docs/CODE_WALKTHROUGH.md). Continuing the project? Start with [`HANDOFF.md`](HANDOFF.md): architecture, design decisions, invariants and common tasks, written so you can also hand it to an AI assistant.

## API

The server exposes a small JSON API next to the website.

| Endpoint | Description |
|---|---|
| `POST /api/analyze` | body `{"text": "..."}` (50 words to 12,000 characters); returns the verdict, probabilities, SHAP contributions and word highlights |
| `GET /api/meta` | example texts, typical feature values and the evaluation tables |
| `GET /api/health` | `{"status": "ok"}` |

```bash
curl -X POST http://127.0.0.1:8000/api/analyze -H "Content-Type: application/json" -d '{"text": "..."}'
```

## Project structure

```
src/                 the ML library
  data.py              load, clean, balance and split the datasets
  features.py          the 18 features (GPT-2 perplexity + writing style)
  models.py            perplexity threshold, TF-IDF + LR, XGBoost, metrics
  train_distilbert.py  DistilBERT fine-tuning
  explain.py           SHAP, LIME, word weights
  robustness.py        T5 paraphraser for the robustness test
  detector.py          analyse one text with the saved models
run_pipeline.py      the full study, one function per stage
server.py            FastAPI app: the API, input checks, security, serves the site
web/                 the website (React, TypeScript, Tailwind CSS, Motion)
tests/               API tests (pytest)
models/              trained models used by the app
reports/             result tables, figures and run logs
docs/                technical report and code walkthrough
submissions/         final report (PDF, Word) and presentation
HANDOFF.md           guide for anyone (or any AI assistant) continuing the project
```

## Reproducing the results

```bash
python run_pipeline.py --quick      # every stage on a small sample, about 10 minutes
python run_pipeline.py              # the full study, about 1.5 hours on a laptop GPU
python run_pipeline.py --stages evaluate ood paraphrase explain   # results only, reuses saved work
```

The datasets download automatically on the first run. Every stage saves its output, so any stage can be rerun on its own. The seed is fixed at 42.

## Testing

```bash
python -m pytest tests -q           # 28 API tests
cd web && npm run lint && npm run typecheck
```

## Security and privacy

The API never stores or logs the submitted text. It also:
- accepts 50 words to 12,000 characters, and strips invisible characters that could sway the verdict
- refuses text that isn't English prose, instead of giving a meaningless verdict
- caps the request size and rate-limits each IP (20 analyses a minute)
- sends strict security headers (CSP, no framing, no sniffing)

Each of these has a test in `tests/test_server.py`. When running behind a reverse proxy, set `GLASSQUILL_TRUST_PROXY=1` so the rate limit sees each visitor's real IP.

## Limitations

- A detector score is a second opinion, **never proof** on its own.
- Plain, predictable writing (common among non-native writers) can look AI-like.
- Paraphrasing tools, heavy editing and small open models can fool it.
- English only; it works best on 50 to 400 words.

## Datasets and acknowledgements

GlassQuill is trained and evaluated on public research datasets. Each one keeps its own license:

- [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) (Guo et al., 2023)
- [Human-AI Parallel Corpus (HAP-E)](https://huggingface.co/datasets/browndw/human-ai-parallel-corpus) (Reinhart et al., 2024)
- [M4GT-Bench](https://huggingface.co/datasets/Jinyan1/COLING_2025_MGT_en) (Wang et al., 2024)
- [MAGE](https://huggingface.co/datasets/yaful/MAGE) (Li et al., 2024): out-of-domain test only

It builds on GPT-2, DistilBERT, XGBoost, SHAP and LIME.

## Authors

- Ranjan Verma
- Rahul Narkhede
- Priyansh Johri
- Priyansh Pandey

## License

[MIT](LICENSE)
