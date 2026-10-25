# GlassQuill handoff

Everything someone continuing this project needs to know: what exists, how it fits together, why it was built this way, and where the sharp edges are. It is written so you can read it yourself or paste it into an AI assistant as context before asking it to change the code.

> **For AI assistants:** treat this file as the map and the code as the truth. Before changing anything, read the files named in the section you are working on. Respect the [invariants](#6-invariants-do-not-break-these); most of them exist because breaking them silently produces wrong results rather than errors.

---

## 1. What the project is

GlassQuill classifies an English text (50 to 400 words) as **human-written or AI-generated** and explains the verdict. It has two halves:

| Half | What it does | Entry point |
|---|---|---|
| **The study** | downloads 4 public datasets, cleans and balances them, trains 4 models, evaluates them in-domain, out-of-domain and against paraphrasing, and writes result tables and figures | `run_pipeline.py` |
| **The app** | a FastAPI server that loads 2 of the trained models plus GPT-2, analyses pasted text, and serves a React website that shows the verdict and its evidence | `server.py` + `web/` |

**Status: complete and working.** The full study has been run; its outputs are committed in `models/` and `reports/`, so the app runs without retraining. There are 28 passing API tests. Nothing is half-built.

**Headline result (test set, 4,829 texts):** the app's verdict reaches 97.4% accuracy at a 2.0% false positive rate, against 96.1% / 7.1% for fine-tuned DistilBERT. All numbers are in `reports/RESULTS.md`.

---

## 2. Architecture in one picture

```
                         TRAINING (run once)                                APP (runs all the time)
 ┌──────────────────────────────────────────────────────────┐   ┌─────────────────────────────────────────┐
 │ run_pipeline.py                                          │   │ server.py  (FastAPI)                    │
 │   data      src/data.py      load, clean, balance, split │   │   POST /api/analyze                     │
 │   features  src/features.py  18 features per text        │   │     clean → check_prose → lock          │
 │   train     src/models.py    threshold, TF-IDF+LR, XGB ──┼──►│     → Detector.analyze (src/detector.py)│
 │   bert      src/train_distilbert.py                      │   │   GET /api/meta  ◄── reports/*.csv      │
 │   evaluate / ood / paraphrase / explain  ────────────────┼──►│   GET /*         ◄── web/dist (React)   │
 │                                   writes models/, reports/   └─────────────────────────────────────────┘
 └──────────────────────────────────────────────────────────┘
```

`src/` is a small shared library. The pipeline uses all of it; the app uses only `detector.py`, `features.py` and `explain.py`.

---

## 3. Repository map

```
run_pipeline.py        the study: one function per stage, explicit STAGES dict, --quick mode
server.py              API, input checks, rate limit, security headers, serves web/dist
src/
  data.py              dataset URLs, loaders, clean_text, prepare (balance), group_split
  features.py          stylometric_features (12) + PerplexityScorer (GPT-2, 6 features)
  models.py            evaluate (metrics), perplexity_threshold, train_tfidf_lr, train_xgb
  train_distilbert.py  fine-tuning + predict_proba (also used by LIME)
  explain.py           NICE_NAMES, SHAP plots, lr_word_weights, highlight, LIME
  robustness.py        T5 paraphraser for the paraphrase attack
  detector.py          Detector: loads saved models once, analyze(text) for the app
tests/test_server.py   28 API tests (validation, security, behaviour, site serving)
web/                   React 19 + TypeScript + Vite + Tailwind 4 + Motion
  src/App.tsx            routes, page transitions, scroll-to-hash, error boundary
  src/components/        SiteLayout (header/footer), ui.tsx (shared components)
  src/lib/               api.ts (typed client), hooks.ts, motion.ts (animation tokens), seo.tsx
  src/pages/             Home, Detect, Results, Method, NotFound
  vite.config.ts         SEO plugin (sitemap/robots with a %SITE_URL% placeholder), /api proxy
models/                xgb.joblib, tfidf_lr.joblib, feature_names.json, feature_stats.json,
                       perplexity_threshold.json  (DistilBERT is not committed: 1 GB)
reports/               result CSVs, RESULTS.md, figures/, run logs, local_cases.md
examples.json          4 held-out example texts shown as buttons on the Detector page
docs/REPORT.md         the technical report (method, results, references)
docs/CODE_WALKTHROUGH.md  file-by-file guide with links to each function
```

Git-ignored but used locally: `data/` (downloads and cached features), `models/distilbert/`, `*/quick/` (quick-run outputs), `web/dist/`, `web/node_modules/`, `.venv/`.

---

## 4. How each part works

### 4.1 Data (`src/data.py`, `stage_data`)
- **Sources:** HC3 (ChatGPT 2022 vs human answers to the same questions), HAP-E (GPT-4o and Llama-3 continuations of human text, 6 genres), M4GT (recent chat models, 6 domains). **MAGE** (27 generators) is loaded only for the out-of-domain test.
- **Row format everywhere:** `qid` (group id), `source` (domain), `text`, `label` (0 = human, 1 = AI), `generator`, and later `corpus` and `n_words`.
- **`clean_text`** removes formatting that reveals the *source* rather than the *writing*. HC3's human answers were word-tokenized (`do n't`, `( word )`, `URL_0`); AI answers were not. It also normalises curly quotes, removes markdown and citation markers, and strips giveaway phrases such as "As an AI language model". Without this, accuracy was a misleading 99.3%.
- **`prepare`** drops empty and duplicate texts, keeps 50 to 400 words (HAP-E and M4GT are truncated at a sentence end instead of dropped), then samples **equal human and AI counts within every source**, so the topic cannot predict the label.
- **`group_split`** makes a 70/15/15 split with `GroupShuffleSplit` on `qid`, and asserts that no group appears in both train and test.
- **Result:** 31,984 texts, 17 domains. Per-domain counts are in `reports/eda_source_counts.csv`.

### 4.2 Features (`src/features.py`)
18 numbers per text, in the fixed order of `models/feature_names.json`:
- **12 style features:** sentence length mean, std and burstiness (std / mean), type-token ratio, word length, comma, punctuation, function-word, first-person and contraction rates, Flesch reading ease, and Flesch-Kincaid grade.
- **6 GPT-2 features** (`PerplexityScorer`, first 256 tokens): mean token negative log-likelihood (= log perplexity), its std and max, the share of tokens that were GPT-2's top-1 and top-10 guess, and mean entropy.
- **Main signal:** AI text is more predictable to another language model, so it has lower perplexity (effect size about -1.4 SD).

### 4.3 Models (`src/models.py`, `src/train_distilbert.py`)
| Model | Input | Tuning | Role |
|---|---|---|---|
| Perplexity threshold | `log_ppl` only | cut-off by Youden's J on validation | zero-shot baseline |
| TF-IDF + logistic regression | 1-2 grams, 50k features | C ∈ {0.1, 1, 10} by 5-fold GroupKFold | glass box on words |
| XGBoost | the 18 features | depth ∈ {3, 4, 6}, lr ∈ {0.05, 0.1}, 400 trees, GroupKFold | glass box on style |
| DistilBERT | raw text, 256 tokens | 2 epochs, best epoch by validation F1 | black-box comparison |

- **Monotone constraints:** XGBoost uses them for features where theory (`EXPECTED_DIRECTION`) and the training data agree on the direction (`monotone_directions`).
- **Grid-search workers:** capped at `N_JOBS = 4`; using every core ran a laptop out of memory.
- **The app's verdict** is `p_final = (p_xgb + p_lr) / 2`. The rule was chosen on the **validation** set. The verdict shown is AI at ≥ 0.65, human at ≤ 0.35, and unclear in between (`UNSURE` in `server.py`).

### 4.4 Evaluation (`stage_evaluate`, `stage_ood`, `stage_paraphrase`)
- **One scorer for all:** `score_all` scores every model the same way. `evaluate` reports accuracy, precision, recall, F1, ROC-AUC and FPR (humans wrongly flagged as AI, the error we care most about).
- **In-domain:** the test set, overall, per corpus and per generator.
- **Out-of-domain:** 2,000 balanced MAGE texts, broken down per domain and per generator.
- **Paraphrase:** 60 HC3 AI test texts are rewritten sentence by sentence with T5, then recall is compared before and after.

### 4.5 Explanations (`src/explain.py`, `stage_explain`)
- **XGBoost:** TreeSHAP gives a global beeswarm, a bar chart, a direction table (`shap_global.csv`) and waterfalls for 4 test cases (`local_cases.md`).
- **TF-IDF + LR:** word weights are exact, computed as the TF-IDF value × the coefficient (`lr_word_weights`). These drive the coloured words in the app.
- **Agreement check:** SHAP vs LIME on DistilBERT, and LIME vs the exact LR weights, written to `explanation_agreement.json`. LIME recovers only about half of LR's true top-10 words, which is why SHAP and exact weights are used in the app.

### 4.6 The pipeline script (`run_pipeline.py`)
- **Stages:** `data → features → train → bert → evaluate → ood → paraphrase → explain`. Pick any subset with `--stages`. `--quick` writes to `data/quick`, `models/quick` and `reports/quick` on a small sample (about 10 minutes).
- **Caching:** every stage saves its output. Feature tables are cached as `data/X_<name>.parquet` because GPT-2 is the slow part. A new data split deletes the stale caches.
- **Feature stats:** `stage_features` also writes `models/feature_stats.json`, the per-class **medians** shown as "typical human / AI" in the app.
- **Summary:** `write_summary` builds `reports/RESULTS.md`.
- **Seed:** fixed at 42 for Python, NumPy, torch, the splits and sampling.

### 4.7 The server (`server.py`)
- **Startup:** loads `Detector` once (GPT-2 on CUDA if available, otherwise CPU). A `threading.Lock` lets only one analysis run at a time; a request waits up to 20 s and then gets 503.
- **Endpoints:**

| Endpoint | Returns |
|---|---|
| `POST /api/analyze` | verdict, `p_final`, `p_style`, `p_words`, `models_agree`, `shap_base`, `contributions` (sorted by \|SHAP\|), `highlights`, `words`, `ms` |
| `GET /api/meta` | the examples, typical values, limits, and the result tables. Cached for the whole run |
| `GET /api/health` | a simple status check |

- **Input checks:**
  - pydantic caps the text at 12,000 characters
  - `clean` strips control and invisible format characters, a known trick to sway detectors
  - `check_prose` requires at least 50 words, at least 40% English words and at least 20 distinct words
  - validation errors never echo the text back
- **`guard` middleware:**
  - POST requests to the API need a Content-Length (411 without one) and are capped at 64 KB (413)
  - a sliding-window rate limit of 20 requests a minute per IP (429)
  - security headers: a strict CSP, no framing, nosniff, COOP and CORP, and HSTS over HTTPS
  - Cache-Control: `no-store` for the API, immutable for hashed assets, `no-cache` otherwise
- **Serving the site:**
  - files come from `web/dist`, with path-traversal protection
  - unknown routes get the app with a real 404 status
  - `index.html`, `sitemap.xml` and `robots.txt` contain a `%SITE_URL%` placeholder that the server fills in per request, so one build works on any host
- **Settings (environment variables):**

| Variable | Default | Purpose |
|---|---|---|
| `HOST`, `PORT` | `127.0.0.1`, `8000` | bind address |
| `GLASSQUILL_MODELS` | `models` | model folder (e.g. `models/quick`) |
| `GLASSQUILL_RATE` | `20` | analyses per IP per minute |
| `GLASSQUILL_TRUST_PROXY` | off | `1` behind a reverse proxy: use the last `X-Forwarded-For` entry and allow `X-Forwarded-Proto` |
| `GLASSQUILL_CORS` | `http://localhost:5173` | allowed origins (the Vite dev server) |
| `SITE_URL` | the visited address | fixed public address for canonical links |

### 4.8 The website (`web/`)
- **Stack:** React 19, React Router 7, Tailwind 4 (design tokens in `@theme` in `index.css`), Motion (`motion/react`) and lucide icons. The site is dark-only.
- **Pages:**
  - **Home:** hero, live stats from `/api/meta`, how it works, the false-positive comparison, a results teaser
  - **Detect:** input, verdict, top reasons, SHAP chart, word evidence, the table of all 18 features
  - **Results:** in-domain, MAGE and paraphrase tables, per-corpus and per-generator charts, global SHAP
  - **Method:** pipeline, datasets, feature medians, limits
  - **NotFound**
- **Data flow:** `lib/api.ts` is the only place that calls the server. `getMeta()` is fetched once per page load and shared through `useMeta()`. Every number on the site comes from the API, except the hero example card, which shows a real result written out by hand.
- **Animation:** shared tokens live in `lib/motion.ts`. Only transform and opacity are animated, and `MotionConfig reducedMotion="user"` respects the system setting. `CountUp` needs a **stable** `format` function, defined outside the component.
- **Robustness:**
  - inner pages are lazy-loaded
  - an error boundary replaces a page that fails to load
  - a malformed `#hash` is ignored
  - the Analyse request can be aborted (Clear, or a newer Analyse)
- **SEO:** a per-page `<Seo>` component sets the title, description, canonical and social tags. The Vite plugin emits `sitemap.xml` and `robots.txt`.

---

## 5. Running, testing, rebuilding

```bash
# setup (Python 3.11, Node 20+)
python -m venv .venv && .venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or: pip install torch
pip install -r requirements.txt
cd web && npm install && npm run build && cd ..

python server.py                     # app at http://127.0.0.1:8000
cd web && npm run dev                # frontend dev server at :5173 (needs server.py running)

python -m pytest tests -q            # 28 tests, ~25 s (loads the real models)
cd web && npm run lint && npm run typecheck

python run_pipeline.py --quick       # whole study on a sample, ~10 min
python run_pipeline.py               # full study, ~1.5 h on a laptop GPU (RTX 3050)
python run_pipeline.py --stages evaluate ood paraphrase explain   # results only
GLASSQUILL_MODELS=models/quick python server.py                   # app on quick-run models
```

Versions the results were produced with: torch 2.11 (CUDA 12.8), transformers 5.20, scikit-learn 1.9, xgboost 3.2, shap 0.51, FastAPI 0.162.

---

## 6. Invariants: do not break these

1. **Split by group, tune by group.** Train, validation and test never share a `qid`. All tuning uses `GroupKFold(groups=qid)`. Without this, HC3's paired answers leak the topic and scores inflate.
2. **The test set is used once, at the end.** Thresholds and the verdict rule are chosen on validation. Don't tune anything on `test` or MAGE.
3. **Feature order is `models/feature_names.json`.** `Detector` reorders its input to match it. If you add or rename a feature, retrain XGBoost, add it to `NICE_NAMES`, and regenerate `feature_stats.json` (`stage_features` does this).
4. **`clean_text` must be applied the same way to every source.** It exists to remove source fingerprints. A rule that touches only one class re-creates a shortcut.
5. **Balance per source.** Equal human and AI counts in each `source`, or the domain predicts the label.
6. **Label convention: 1 = AI.** MAGE uses 1 = human and is flipped in `load_mage`.
7. **The verdict rule lives in two places that must match:** `Detector.analyze` (`p_final`) and `score_all` (`"XGBoost + LR (app)"`). The site looks results up by that exact model name (`APP_MODEL` in `web/src/lib/api.ts`).
8. **The models are committed; DistilBERT is not.** The app needs only `xgb.joblib`, `tfidf_lr.joblib` and the three JSON files. Stages that score DistilBERT (`evaluate`, `ood`, `paraphrase`, `explain`) need `models/distilbert/`, so run `--stages bert` first on a fresh clone.
9. **Never return or log the user's text** beyond the analysis response. The response is `no-store`.

---

## 7. Common tasks

| Task | Where and how |
|---|---|
| Add a feature | `stylometric_features` or `PerplexityScorer.features` → add a readable name in `NICE_NAMES` → if you know its direction, add it to `EXPECTED_DIRECTION` → rerun `features train evaluate ood paraphrase explain` |
| Change the unclear band | `UNSURE` in `server.py` and the band strip in `web/src/pages/Detect.tsx` |
| Change the verdict rule | `Detector.analyze`, `score_all`, then re-evaluate. Choose the rule on validation (invariant 2) |
| Add a dataset | write a loader in `src/data.py` that returns the row format → add it to `load_corpus` and `CORPORA` in `run_pipeline.py` → set `SIZES["per_class"]` |
| Add an API field | `analyze` in `server.py` → the `Analysis` type in `web/src/lib/api.ts` → the component that uses it → a test |
| Add a page | `web/src/pages/X.tsx` (default export, with `<Seo>`) → a lazy route in `App.tsx` → `NAV` in `SiteLayout.tsx` → `PAGES` in `server.py` and `vite.config.ts` |
| Update the examples | edit `examples.json` (label + text). Use held-out texts, not training texts |
| Change the rate limit | `GLASSQUILL_RATE` env var |

---

## 8. Gotchas

- **Windows paths and shells:** the project was developed on Windows (Git Bash / PowerShell). Paths in code use `os.path` / `pathlib`, so it also runs on macOS and Linux.
- **GPT-2 reads only the first 256 tokens** (about 190 words). Style features and TF-IDF use the whole text. Texts over 400 words still work but are only partly scored by GPT-2.
- **GPT-2 downloads on first start** (~500 MB, into the Hugging Face cache). The datasets download on the first pipeline run, into `data/`.
- **Memory:** grid search with `n_jobs=-1` exhausted RAM on the development laptop, hence `N_JOBS = 4`.
- **The whole site depends on `/api/meta`.** If the server is down, pages show skeletons and an error message.
- **The site is served from `web/dist`.** After frontend changes, run `npm run build` (or use `npm run dev`), otherwise `server.py` keeps serving the old build. Tests that touch the site are skipped if `web/dist` is missing.
- **A second server on port 8000** fails with "address already in use". Stop the old one or set `PORT`.
- **SHAP vs LIME agreement on DistilBERT is low** (Jaccard 0.06). That's a real finding about LIME's instability, not a bug.

---

## 9. Known limitations and ideas for next steps

**Limitations:**
- 66% accuracy on unseen generators (MAGE).
- Small open base models (OPT, BLOOM, GPT-J) go undetected.
- Paraphrasing lowers recall.
- English only.
- No detection inside mixed human/AI texts.
- Possible bias against plain writing, e.g. by non-native writers.

**Ideas, roughly by value:**
1. Add a stronger reader model, or Binoculars (two-model perplexity ratio), as extra features. This was tried: +0.01 to +0.04 AUC, but about 25× slower.
2. Choose the threshold for a fixed 1% FPR instead of 0.5.
3. Sentence-level scoring, to find AI passages inside human text.
4. Evaluate on the RAID benchmark, and add newer generators to training.
5. Calibrate the probabilities (e.g. isotonic) so "80%" means 80%.
6. Measure fairness on non-native English writing (e.g. the TOEFL essays from Liang et al., 2023).
7. Deployment: the app is a single process with in-memory rate limiting. A multi-instance deployment needs a shared rate limiter and a model server.

---

## 10. Conventions

- **Python:** plain functions, short docstrings that explain *why*, constants at the top of the module. No class unless it holds loaded state (`Detector`, `PerplexityScorer`, `Paraphraser`).
- **TypeScript:** function components, Tailwind classes inline, shared pieces in `components/ui.tsx`, motion values from `lib/motion.ts`. `npm run lint` and `npm run typecheck` must pass.
- **Every security check has a test** in `tests/test_server.py`. Keep it that way.
- **Results are generated, never typed.** Tables come from `run_pipeline.py`, and the site reads them through `/api/meta`.
