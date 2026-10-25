# Code walkthrough

A guide to the code. It goes file by file, says what each part does and why, and ends with common questions about the design. Read it with the files open; each heading links to the code.

## The project in one picture

```
                 run_pipeline.py  (training and evaluation: run once)
 datasets ──► src/data.py ──► src/features.py ──► src/models.py, src/train_distilbert.py
 (HC3, HAP-E,  clean, balance,   18 numbers           4 models + tuning
  M4GT, MAGE)  split by group    per text                 │
                                                          ▼
                              reports/ (tables, figures) and models/ (saved models)

                 server.py  (the website: runs all the time)
 browser ──► POST /api/analyze ──► clean + check text ──► src/detector.py ──► JSON verdict
 (web/src)                                                (loads models/)
```

Two programs, one shared library (`src/`):
- **`run_pipeline.py`** trains and evaluates everything and writes the results. You run it once (or `--quick` for a 10-minute demo).
- **`server.py`** loads the saved models and answers the website.

| Folder / file | What it is | Lines |
|---|---|---|
| `src/data.py` | loading, cleaning, balancing, splitting | 170 |
| `src/features.py` | the 18 features (GPT-2 + writing style) | 84 |
| `src/models.py` | perplexity threshold, TF-IDF + LR, XGBoost, metrics | 74 |
| `src/train_distilbert.py` | DistilBERT fine-tuning | 54 |
| `src/explain.py` | SHAP, LIME, word weights | 75 |
| `src/robustness.py` | the T5 paraphraser for the robustness test | 23 |
| `src/detector.py` | analyse one text with the saved models | 43 |
| `run_pipeline.py` | the whole study, one function per stage | ~430 |
| `server.py` | the web API and its security | ~240 |
| `tests/test_server.py` | 28 automatic checks of the API | |
| `web/src/` | the website (React) | |

---

## 1. Data: [`src/data.py`](../src/data.py)

**Target variable:** `label`, 0 = human, 1 = AI. Every loader returns the same columns: `qid` (group), `source` (domain), `text`, `label`, `generator`.

| Function | What it does |
|---|---|
| [`load_hc3`](../src/data.py#L33) | one row per answer; `qid` = the question, so a question's human and AI answers stay together |
| [`load_hape`](../src/data.py#L46) | human text and its GPT-4o / Llama-3 continuations; `qid` = the source document |
| [`load_m4gt`](../src/data.py#L59) | keeps only human texts and recent chat models (`M4GT_MODELS`), drops its HC3 and MAGE parts |
| [`load_mage`](../src/data.py#L68) | the out-of-domain test set; **flips the labels** (MAGE uses 1 = human) |
| [`clean_text`](../src/data.py#L104) | removes formatting that gives away the source, see below |
| [`truncate_words`](../src/data.py#L129) | cuts long texts at a sentence end at 400 words |
| [`prepare`](../src/data.py#L142) | drop missing and duplicate texts, keep 50-400 words, **balance human/AI per source** |
| [`group_split`](../src/data.py#L161) | 70/15/15 split with `GroupShuffleSplit`, and an `assert` that no group is in both train and test |

**Why `clean_text` matters (data leakage):** HC3's human answers were word-tokenized and the AI answers were not, so `do n't`, `( word )` and `URL_0` appeared in 34%, 35% and 12% of human texts and almost none of the AI ones. A model would learn "space before n't means human". Each `re.sub` line removes one such shortcut, with a comment saying which.

**Missing values:** `dropna(subset=["text"])` in `prepare`. After cleaning, no split has a missing value.

**Why balance per source:** in HC3, `open_qa` had 20 human vs 764 AI texts. Without balancing, the topic alone would predict the label. `prepare` takes, for each source, the smaller of its two class counts.

## 2. Features: [`src/features.py`](../src/features.py)

[`stylometric_features`](../src/features.py#L23) returns **12 style features** using plain Python and regular expressions:
- sentence length mean and spread, and `burstiness` = spread / mean
- vocabulary richness (`type_token_ratio` = unique words / words), average word length
- commas and punctuation per word, share of function words, first-person words and contractions
- two readability scores from `textstat`

[`PerplexityScorer`](../src/features.py#L44) returns **6 GPT-2 features**. The key method is [`token_scores`](../src/features.py#L54):
1. The text is turned into GPT-2 tokens (at most 256).
2. One forward pass gives `logits`: a score for every possible next token at every position.
3. `log_softmax` turns them into log-probabilities.
4. `gather` picks the probability of the token that actually came next. Minus that is the token's **surprise** (`nll`).
5. `ranks`: how many tokens GPT-2 thought were more likely than the real one (1 = its first guess).
6. `entropy`: how unsure GPT-2 was at each step.

[`features`](../src/features.py#L66) averages these: `log_ppl` (mean surprise = log perplexity), `nll_std`, `nll_max`, `top1_frac`, `top10_frac`, `mean_entropy`.

**Why it works:** an LLM picks likely tokens, so another language model finds AI text easy to predict. AI text has lower perplexity and more first-guess tokens.

## 3. Models: [`src/models.py`](../src/models.py)

| Function | Model | Key lines |
|---|---|---|
| [`perplexity_threshold`](../src/models.py#L29) | 1. Perplexity threshold | `roc_curve` on validation, pick the cut-off where TPR − FPR is highest (**Youden's J**) |
| [`train_tfidf_lr`](../src/models.py#L37) | 2. TF-IDF + logistic regression | `TfidfVectorizer(ngram_range=(1, 2))` + `LogisticRegression`, `GridSearchCV` over C ∈ {0.1, 1, 10} |
| [`train_xgb`](../src/models.py#L64) | 3. XGBoost | 400 trees, `GridSearchCV` over depth {3, 4, 6} × learning rate {0.05, 0.1}, **monotone constraints** |
| [`evaluate`](../src/models.py#L16) | metrics for any model | accuracy, precision, recall, F1, ROC-AUC, **FPR** = fp / (fp + tn) |

Both grid searches use `GroupKFold(n_splits=5)` with `groups=qid`, so the validation folds are leak-free too, and they score on F1.

[`monotone_directions`](../src/models.py#L54) compares the class means on the training data with `EXPECTED_DIRECTION` (from theory, e.g. lower burstiness → AI). Only features where both agree get a constraint, and XGBoost then cannot learn the opposite rule.

[`src/train_distilbert.py`](../src/train_distilbert.py), model 4: the Hugging Face `Trainer` fine-tunes `distilbert-base-uncased` with a 2-class head (lr 2e-5, batch 16, 2 epochs, 256 tokens). `load_best_model_at_end=True` with `metric_for_best_model="f1"` keeps the best epoch on validation. [`predict_proba`](../src/train_distilbert.py#L46) returns `[P(human), P(AI)]` per text, which is also the format LIME needs.

## 4. Explanations: [`src/explain.py`](../src/explain.py)

| Function | What it gives |
|---|---|
| [`shap_global`](../src/explain.py#L22) | `TreeExplainer` on the whole test set and the beeswarm plot (global importance) |
| [`shap_local_figure`](../src/explain.py#L29) | a waterfall plot for one text (local explanation) |
| [`plain_reasons`](../src/explain.py#L40) | the 3 largest SHAP values as sentences: "Perplexity = 2.73 pushed towards AI" |
| [`lr_word_weights`](../src/explain.py#L52) | exact word contributions of the LR model: TF-IDF value × coefficient |
| [`highlight`](../src/explain.py#L62) | those weights matched to the words of the text, for the coloured word view |
| [`lime_explain`](../src/explain.py#L72) | LIME: remove random words 500 times, fit a small linear model, return its word weights |

**SHAP property to quote:** base value + sum of all SHAP values = the model's output in log-odds. TreeSHAP computes them exactly for trees.

## 5. The pipeline: [`run_pipeline.py`](../run_pipeline.py)

Read it top to bottom: settings → small helpers → one function per stage → `main`.

- **Settings:** `SEED = 42` everywhere, and [`setup`](../run_pipeline.py#L46) sets the folders and sizes for a full or a `--quick` run.
- **Helpers:**
  - `features()` computes the 18 features once and caches them as a parquet file, since the GPT-2 step is the slow part.
  - `models()` and `bert()` load saved models once (`@cache`).
  - [`score_all`](../run_pipeline.py#L128) returns every model's score and threshold, so all test sets are scored the same way. The app's verdict is `(p_xgb + p_lr) / 2`.
- **Stages:** `stage_data` → `stage_features` → `stage_train` → `stage_bert` → `stage_evaluate` → `stage_ood` → `stage_paraphrase` → `stage_explain`. The [`STAGES`](../run_pipeline.py#L407) table maps names to functions, and `--stages` picks which to run.
- **Outputs:** every result table goes to `reports/`, and `write_summary` collects them into `reports/RESULTS.md`. `stage_features` also writes `models/feature_stats.json`, the per-class training medians that the website shows as "typical human / typical AI".

## 6. Using the models: [`src/detector.py`](../src/detector.py)

`Detector.__init__` loads the two saved models, the feature names and GPT-2 once. [`analyze(text)`](../src/detector.py#L22) then:
1. Computes the 18 features.
2. `p_style` = XGBoost's P(AI), and `p_words` = the LR model's P(AI).
3. Computes the SHAP values for this one text (the reasons).
4. Returns `p_final = (p_style + p_words) / 2`, plus the contributions and word highlights.

## 7. The web server: [`server.py`](../server.py)

| Part | Why |
|---|---|
| [`guard`](../server.py#L77) middleware | runs on every request: requires `Content-Length` and caps the body at 64 KB, applies the rate limit, adds security headers (CSP, frame and sniffing protection) and cache rules |
| [`rate_limited`](../server.py#L62) | sliding window: at most 20 analyses per IP per minute (behind a proxy, `client_ip` trusts only the address the proxy added, so it cannot be faked) |
| `invalid_request` handler | short error messages that never echo the user's text |
| [`clean`](../server.py#L170) | strips invisible characters (zero-width spaces, null bytes), a known trick to fool detectors |
| [`check_prose`](../server.py#L178) | refuses text that isn't English prose instead of giving a meaningless verdict |
| [`analyze`](../server.py#L194) | clean → check → wait for the model (503 if busy) → `Detector.analyze` → verdict band: ≥ 0.65 AI, ≤ 0.35 human, else unclear |
| `/api/meta` | the examples and the result tables from `reports/`, so the website never hard-codes a number |
| `site` | serves the built website; unknown pages get a real 404; fills the site's own address into `index.html`, `sitemap.xml` and `robots.txt` |

## 8. The website: [`web/src/`](../web/src)

| File | What it is |
|---|---|
| `App.tsx` | the routes (/, /detect, /results, /method) and the page transition |
| `components/SiteLayout.tsx` | header, navigation, mobile menu, footer |
| `components/ui.tsx` | shared pieces: Button, PageHeader, Bar, CountUp, toggle, toast |
| `lib/api.ts` | the two API calls (`analyze`, `getMeta`) and their types |
| `lib/motion.ts` | animation timings, so every animation feels the same |
| `pages/*.tsx` | one file per page |

The website only displays what the server returns, and every number on it comes from the models or `reports/`.

## 9. Tests: [`tests/test_server.py`](../tests/test_server.py)

Run them with `python -m pytest tests -q`. They check:
- **validation:** too short, too long, wrong type, not English
- **security:** body cap, chunked bodies, rate limit (including a faked `X-Forwarded-For`), headers, no path traversal
- **behaviour:** a real AI text is flagged and a real human text is not, and invisible characters do not change the verdict

---

## Common questions

**Where do you prevent data leakage?** In three places:
- `group_split` in `src/data.py` splits by question or document and `assert`s that no group crosses from train to test.
- The grid searches use `GroupKFold(groups=qid)`, so tuning is leak-free too.
- `clean_text` removes source-specific formatting.

**How is perplexity computed?** In `PerplexityScorer.token_scores`: `log_softmax` of GPT-2's logits, then `gather` the log-probability of each real next token. The negative mean of those is the log perplexity (`log_ppl`).

**Where are the hyperparameters tuned?** In `GridSearchCV` in `train_tfidf_lr` (C) and `train_xgb` (depth, learning rate), with 5-fold `GroupKFold` and F1 scoring. The perplexity threshold is chosen in `perplexity_threshold` on the validation set.

**How do you handle class imbalance?** We don't train on imbalanced data: `prepare` samples equal human and AI counts per source. That is why accuracy is meaningful here, but we still report precision, recall, F1 and FPR.

**What is `random_state=42` / `SEED`?** It fixes the randomness (sampling, splits, model initialisation), so a rerun gives the same numbers.

**Why `predict_proba` and not `predict`?** We need a score, not just a label: for ROC-AUC, for averaging the two models, and for the "unclear" band.

**How does the app decide?** `Detector.analyze` averages XGBoost's and the LR model's P(AI). `server.py` turns that into AI (≥ 0.65), human (≤ 0.35) or unclear.

**Where are the explanations computed?**
- For one text: `self.explainer(x)` in `Detector.analyze` (SHAP), plus `lr_word_weights` for the words.
- For the report: `stage_explain` in `run_pipeline.py`.

**What does `@cache` do in `run_pipeline.py`?** It loads a model or file once and reuses it, so a stage doesn't reload GPT-2 every time.

**How would you add a new feature?** Add it to the dict in `stylometric_features` (or `PerplexityScorer.features`), delete the cached `data/X_*.parquet`, and rerun `--stages features train evaluate`. SHAP will show whether it helps.

**How is the website protected?** See section 7. Each protection has a test in `tests/test_server.py`.
