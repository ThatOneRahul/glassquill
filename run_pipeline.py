"""GlassQuill: the whole study in one script, one function per stage.

    python run_pipeline.py                        # every stage (about 1.5 h on a laptop GPU)
    python run_pipeline.py --stages evaluate ood  # only some stages (reuses saved work)
    python run_pipeline.py --quick                # small sample in */quick/, about 10 min

Stages, in order:
    data        load HC3 + HAP-E + M4GT, clean, balance, split by group      -> data/*.parquet
    features    18 features per text (GPT-2 surprise + writing style)        -> data/X_*.parquet
    train       perplexity threshold, TF-IDF + LR, XGBoost                    -> models/
    bert        fine-tune DistilBERT                                         -> models/distilbert/
    evaluate    metrics on the test set, confusion matrices, ROC curves      -> reports/
    ood         same metrics on MAGE (27 generators never seen in training)  -> reports/
    paraphrase  recall before and after rewriting AI texts with T5           -> reports/
    explain     SHAP, LIME, top words, explanation agreement                 -> reports/
Every stage saves its output, so a later stage can run on its own.
"""
import argparse
import json
import os
import random
import time
import urllib.request
from functools import cache

import joblib
import matplotlib
matplotlib.use("Agg")                      # draw plots to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data import (HAPE_FILES, HAPE_URL, HC3_URL, M4GT_URL, MAGE_URL, group_split,
                      load_hape, load_hc3, load_m4gt, load_mage, mage_source, prepare)
from src.models import evaluate

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CORPORA = ["hc3", "hape", "m4gt"]
# filled in by setup(): folders and sample sizes for a full or a quick run
DATA = MODELS = REPORTS = FIGS = ""
SIZES = {}


def setup(quick):
    global DATA, MODELS, REPORTS, FIGS, SIZES
    sub = "quick" if quick else ""
    DATA, MODELS, REPORTS = (os.path.join(d, sub) for d in ("data", "models", "reports"))
    FIGS = os.path.join(REPORTS, "figures")
    for d in (DATA, MODELS, FIGS):
        os.makedirs(d, exist_ok=True)
    SIZES = {
        "per_class": {"hc3": 300, "hape": 200, "m4gt": 300} if quick else
                     {"hc3": 6000, "hape": 4000, "m4gt": 6000},   # texts per class, per corpus
        "mage": 100 if quick else 1000,       # MAGE texts per class
        "paraphrase": 20 if quick else 60,    # AI texts to paraphrase
        "agreement": 5 if quick else 10,      # texts for the SHAP vs LIME check
        "bert_epochs": 1 if quick else 2,
    }


# ---- small helpers -------------------------------------------------------------

def path(folder, name):
    return os.path.join(folder, name)


def download(url, target):
    """Download once; raw files are shared by full and quick runs (they live in data/)."""
    if not os.path.exists(target):
        print("  downloading", url)
        urllib.request.urlretrieve(url, target + ".part")
        os.replace(target + ".part", target)
    return target


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(path(FIGS, name), dpi=200)
    plt.close(fig)


@cache
def split(name):                                  # "train", "val" or "test"
    return pd.read_parquet(path(DATA, f"{name}.parquet"))


@cache
def scorer():
    from src.features import PerplexityScorer     # loads GPT-2, so only when needed
    return PerplexityScorer("gpt2", device=DEVICE)


def features(name, texts):
    """The 18 features for a set of texts, computed once and cached as data/X_<name>.parquet."""
    file = path(DATA, f"X_{name}.parquet")
    if not os.path.exists(file):
        from src.features import build_feature_table
        t0 = time.time()
        build_feature_table(list(texts), scorer()).to_parquet(file)
        print(f"  features for {name}: {len(texts)} texts in {time.time() - t0:.0f}s")
    return pd.read_parquet(file)


@cache
def models():
    """The saved classic models: (feature names, perplexity threshold, TF-IDF + LR, XGBoost)."""
    feats = json.load(open(path(MODELS, "feature_names.json")))
    thr = json.load(open(path(MODELS, "perplexity_threshold.json")))["threshold"]
    return feats, thr, joblib.load(path(MODELS, "tfidf_lr.joblib")), joblib.load(path(MODELS, "xgb.joblib"))


@cache
def bert():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    folder = path(MODELS, "distilbert")
    return (AutoModelForSequenceClassification.from_pretrained(folder).to(DEVICE).eval(),
            AutoTokenizer.from_pretrained(folder))


def bert_proba(texts):
    from src.train_distilbert import predict_proba
    model, tok = bert()
    return predict_proba(list(texts), model, tok, device=DEVICE)


def score_all(df, X):
    """Every model's score for every text, plus its threshold. Higher score = more AI-like."""
    feats, thr, lr, xgb = models()
    p_lr = lr.predict_proba(df["text"])[:, 1]
    p_xgb = xgb.predict_proba(X[feats])[:, 1]
    return {
        "Perplexity threshold": (-X["log_ppl"].values, thr),   # low perplexity = AI
        "TF-IDF + LR": (p_lr, 0.5),
        "XGBoost": (p_xgb, 0.5),
        "DistilBERT": (bert_proba(df["text"])[:, 1], 0.5),
        "XGBoost + LR (app)": ((p_xgb + p_lr) / 2, 0.5),       # the verdict the website shows
    }


def metrics_table(labels, scores):
    return pd.DataFrame([evaluate(m, labels, s, t) for m, (s, t) in scores.items()])


def correct(labels, scores):
    """1 where a model got the text right, per model."""
    return pd.DataFrame({m: ((s >= t).astype(int) == labels).astype(float) for m, (s, t) in scores.items()})


# ---- stages ------------------------------------------------------------------

def load_corpus(name):
    """Raw rows of one corpus with columns qid (group), source, text, label, generator."""
    if name == "hc3":
        d = load_hc3(download(HC3_URL, "data/hc3_all.jsonl"))
        return d.assign(generator=np.where(d["label"] == 1, "chatgpt", "human"))
    if name == "hape":
        os.makedirs("data/hape", exist_ok=True)
        return load_hape({f: download(HAPE_URL.format(f), f"data/hape/{f}.parquet") for f in HAPE_FILES})
    return load_m4gt(download(M4GT_URL, "data/coling_dev.parquet"))


def stage_data():
    """Clean and balance each corpus, split it by group (question / document), then join."""
    parts = []
    for name in CORPORA:
        raw = load_corpus(name)
        print(f"  {name}: {len(raw)} raw texts")
        df = prepare(raw, per_class=SIZES["per_class"][name], seed=SEED, truncate=name != "hc3")
        parts.append([d.assign(corpus=name) for d in group_split(df, seed=SEED)])
    train, val, test = (pd.concat(p, ignore_index=True).astype({"qid": str}) for p in zip(*parts))
    for name, d in [("train", train), ("val", val), ("test", test)]:
        d.to_parquet(path(DATA, f"{name}.parquet"))
        print(f"  {name}: {len(d)} texts, share AI {d['label'].mean():.3f}")
    # a new split makes every cached feature table and the paraphrased test texts stale
    for stale in ["X_train", "X_val", "X_test", "X_para_before", "X_para_after", "paraphrased"]:
        if os.path.exists(path(DATA, f"{stale}.parquet")):
            os.remove(path(DATA, f"{stale}.parquet"))

    # exploratory analysis: texts per source and class, and text lengths
    df = pd.concat([train, val, test])
    counts = df.groupby(["source", "label"]).size().unstack().rename(columns={0: "human", 1: "ai"})
    counts.to_csv(path(REPORTS, "eda_source_counts.csv"))
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, name in [(0, "Human"), (1, "AI")]:
        ax.hist(df.loc[df["label"] == label, "n_words"], bins=40, alpha=0.6, label=name)
    ax.set_xlabel("words per text"); ax.set_ylabel("texts"); ax.legend()
    save_fig(fig, "length_hist.png")


def stage_features():
    """Compute the 18 features and compare their averages for human and AI text (RQ1)."""
    for name in ["train", "val", "test"]:
        features(name, split(name)["text"])
    X, y = features("train", None), split("train")["label"].values
    json.dump(list(X.columns), open(path(MODELS, "feature_names.json"), "w"))
    # the "typical human / typical AI" values the website shows next to each feature
    typical = X.groupby(y).median().T.round(4)
    json.dump({f: {"human": r[0], "ai": r[1]} for f, r in typical.iterrows()},
              open(path(MODELS, "feature_stats.json"), "w"), indent=1)
    means = X.groupby(y).mean().T
    means.columns = ["human_mean", "ai_mean"]
    means.index.name = "feature"
    # effect size: how many standard deviations apart the two classes are
    means["effect_size"] = (means["ai_mean"] - means["human_mean"]) / X.std().replace(0, np.nan)
    means.sort_values("effect_size", key=abs, ascending=False).round(4).to_csv(
        path(REPORTS, "feature_means_by_class.csv"))


def stage_train():
    """Fit the three classic models. Tuning uses 5-fold GroupKFold on train only."""
    from src.models import perplexity_threshold, train_tfidf_lr, train_xgb
    train, val = split("train"), split("val")
    X_train, X_val = features("train", None), features("val", None)
    feats = list(X_train.columns)
    thr = float(perplexity_threshold(X_val["log_ppl"].values, val["label"].values))
    json.dump({"threshold": thr, "rule": "AI if -log_ppl >= threshold"},
              open(path(MODELS, "perplexity_threshold.json"), "w"), indent=2)
    print(f"  perplexity rule (chosen on validation): log_ppl <= {-thr:.3f} means AI")
    joblib.dump(train_tfidf_lr(train, seed=SEED), path(MODELS, "tfidf_lr.joblib"))
    joblib.dump(train_xgb(X_train[feats], train["label"], groups=train["qid"], seed=SEED),
                path(MODELS, "xgb.joblib"))
    models.cache_clear()


def stage_bert():
    """Fine-tune DistilBERT; the epoch with the best validation F1 is kept."""
    from src.train_distilbert import train_distilbert
    train_distilbert(split("train"), split("val"), out_dir=path(MODELS, "distilbert"),
                     epochs=SIZES["bert_epochs"], seed=SEED)
    bert.cache_clear()


def stage_evaluate():
    """Test-set metrics overall, per corpus and per generator, plus confusion matrices and ROC."""
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
    test = split("test")
    scores = score_all(test, features("test", None))
    res = metrics_table(test["label"], scores)
    res.round(4).to_csv(path(REPORTS, "results_in_domain.csv"), index=False)
    print(res.round(3).to_string(index=False))

    by_corpus = pd.concat([metrics_table(test["label"][rows], {m: (s[rows], t) for m, (s, t) in scores.items()})
                           .assign(corpus=c) for c, rows in test.groupby("corpus").groups.items()])
    by_corpus.round(4).to_csv(path(REPORTS, "results_in_domain_by_corpus.csv"), index=False)
    hits = correct(test["label"], scores).groupby(test["generator"])
    hits.mean().assign(n=hits.size()).sort_values("XGBoost").round(3).to_csv(
        path(REPORTS, "test_accuracy_by_generator.csv"))

    fig, axes = plt.subplots(1, len(scores), figsize=(4 * len(scores), 4))
    for ax, (m, (s, t)) in zip(axes, scores.items()):
        ConfusionMatrixDisplay.from_predictions(test["label"], (s >= t).astype(int),
                                                display_labels=["Human", "AI"], ax=ax, colorbar=False)
        ax.set_title(m)
    save_fig(fig, "confusion_matrices.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    for m, (s, _) in scores.items():
        RocCurveDisplay.from_predictions(test["label"], s, name=m, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    save_fig(fig, "roc_curves.png")


def stage_ood():
    """Out-of-domain test on MAGE, and which generators and domains fool the models."""
    file = path(DATA, "mage_sample.parquet")
    if not os.path.exists(file):
        load_mage(download(MAGE_URL, "data/mage_test.parquet"), per_class=SIZES["mage"],
                  seed=SEED).to_parquet(file)
    ood = pd.read_parquet(file)
    scores = score_all(ood, features("ood", ood["text"]))
    res = metrics_table(ood["label"], scores)
    res.round(4).to_csv(path(REPORTS, "results_out_of_domain.csv"), index=False)
    print(res.round(3).to_string(index=False))

    domain, generator = zip(*ood["src"].map(mage_source))
    hits = correct(ood["label"], scores).assign(label=ood["label"], domain=domain, generator=generator)
    for key in ["domain", "generator"]:
        g = hits.groupby(key)
        table = g[list(scores)].mean().assign(n=g.size(), share_ai=g["label"].mean())
        table.sort_values("XGBoost").round(3).to_csv(path(REPORTS, f"ood_accuracy_by_{key}.csv"))


def stage_paraphrase():
    """How many AI texts are still caught after every sentence is reworded (H4)."""
    test = split("test")
    ai = test[(test["corpus"] == "hc3") & (test["label"] == 1)]
    ai = ai.sample(n=min(SIZES["paraphrase"], len(ai)), random_state=SEED).reset_index(drop=True)
    file = path(DATA, "paraphrased.parquet")
    if not os.path.exists(file):
        from src.robustness import Paraphraser
        paraphrase = Paraphraser(device=DEVICE)
        ai.assign(text=[paraphrase(t) for t in ai["text"]]).to_parquet(file)
        del paraphrase
        torch.cuda.empty_cache()
    para = pd.read_parquet(file)
    before = score_all(ai, features("para_before", ai["text"]))
    after = score_all(para, features("para_after", para["text"]))
    robust = pd.DataFrame({m: {"recall_before": (before[m][0] >= t).mean(),
                               "recall_after": (after[m][0] >= t).mean()} for m, (_, t) in before.items()}).T
    robust["drop"] = robust["recall_before"] - robust["recall_after"]
    robust.rename_axis("model").round(4).to_csv(path(REPORTS, "results_paraphrase.csv"))
    print(robust.round(3))
    pd.DataFrame({"original": ai["text"], "paraphrased": para["text"]}).head(5).to_csv(
        path(REPORTS, "paraphrase_examples.csv"), index=False)


def stage_explain():
    """Global and local SHAP for XGBoost, top words of the LR model, SHAP vs LIME agreement."""
    import shap
    from src.explain import (NICE_NAMES, lime_explain, lr_word_weights, plain_reasons,
                             shap_global, shap_local_figure)
    feats, _, lr, xgb = models()
    test, X = split("test"), features("test", None)[feats]

    # global: which features matter overall, and which way a higher value pushes (H1, H2)
    sv = shap_global(xgb, X, out_path=path(FIGS, "shap_beeswarm.png"))
    shap.plots.bar(sv, max_display=15, show=False)
    save_fig(plt.gcf(), "shap_bar.png")
    rows = []
    for j, f in enumerate(feats):
        corr = pd.Series(sv.data[:, j]).corr(pd.Series(sv.values[:, j]), method="spearman")
        rows.append({"feature": f, "name": NICE_NAMES.get(f, f), "mean_abs_shap": np.abs(sv.values[:, j]).mean(),
                     "value_vs_shap_spearman": corr, "higher_value_pushes_towards": "AI" if corr > 0 else "human"})
    pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).round(4).to_csv(
        path(REPORTS, "shap_global.csv"), index=False)

    # local: one correct AI, correct human, false positive and false negative
    p = xgb.predict_proba(X)[:, 1]
    y, pred = test["label"].values, (p >= 0.5).astype(int)
    cases = {"true_ai": (y == 1) & (pred == 1), "true_human": (y == 0) & (pred == 0),
             "false_positive": (y == 0) & (pred == 1), "false_negative": (y == 1) & (pred == 0)}
    explainer = shap.TreeExplainer(xgb)
    lines = ["# Local explanations (XGBoost + SHAP)\n"]
    for name, mask in cases.items():
        if not mask.any():
            continue
        i = int(np.argmax(mask))                      # first text of this kind
        fig, row = shap_local_figure(explainer, X.iloc[[i]])
        save_fig(fig, f"waterfall_{name}.png")
        lines += [f"## {name} (P(AI) = {p[i]:.2f}, source: {test['source'][i]})\n",
                  plain_reasons(row), "", f"> {test['text'][i][:500]}...", ""]
    open(path(REPORTS, "local_cases.md"), "w", encoding="utf-8").write("\n".join(lines))

    # glass box: the words with the biggest logistic-regression weights on each side
    vocab = lr.named_steps["tfidfvectorizer"].get_feature_names_out()
    coef = lr.named_steps["logisticregression"].coef_[0]
    order = np.argsort(coef)
    pd.DataFrame({"human_word": vocab[order[:25]], "human_coef": coef[order[:25]],
                  "ai_word": vocab[order[::-1][:25]], "ai_coef": coef[order[::-1][:25]]}).round(4).to_csv(
        path(REPORTS, "lr_top_words.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 7))
    top = np.r_[order[:15], order[::-1][:15]]
    ax.barh(vocab[top], coef[top], color=["#7b4fd6" if c < 0 else "#d64545" for c in coef[top]])
    ax.set_xlabel("logistic regression weight (right = AI, left = human)")
    save_fig(fig, "lr_top_words.png")

    # agreement: SHAP vs LIME on DistilBERT, and LIME vs the exact LR weights
    from transformers import pipeline
    model, tok = bert()
    shap_text = shap.Explainer(pipeline("text-classification", model=model, tokenizer=tok, top_k=None,
                                        device=0 if DEVICE == "cuda" else -1, truncation=True, max_length=256))
    jaccard, recovered = [], []
    for t in test["text"].sample(SIZES["agreement"], random_state=SEED):
        e = shap_text([t])[0, :, "AI"]
        shap_top = {str(e.data[i]).strip().lower() for i in np.argsort(-np.abs(e.values))[:10]}
        lime_top = {w.lower() for w, _ in lime_explain(bert_proba, t, num_features=10)}
        jaccard.append(len(shap_top & lime_top) / max(len(shap_top | lime_top), 1))
        exact = sorted(lr_word_weights(lr, t).items(), key=lambda kv: -abs(kv[1]))
        exact_top = set([w for w, _ in exact if " " not in w][:10])   # top 10 single words, as LIME uses
        lime_lr = {w.lower() for w, _ in lime_explain(lr.predict_proba, t, num_features=10)}
        recovered.append(len(exact_top & lime_lr) / 10)
    agree = {"distilbert_shap_vs_lime_jaccard_top10": round(float(np.mean(jaccard)), 3),
             "lime_recovers_lr_true_top10": round(float(np.mean(recovered)), 3),
             "n_texts": SIZES["agreement"]}
    json.dump(agree, open(path(REPORTS, "explanation_agreement.json"), "w"), indent=2)
    print(" ", agree)


def write_summary():
    """Collect the result tables into reports/RESULTS.md."""
    def table(name, title):
        file = path(REPORTS, name)
        if not os.path.exists(file):
            return f"## {title}\n\n_not run yet_\n"
        d = pd.read_csv(file)
        rows = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
        rows += ["| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |"
                 for r in d.itertuples(index=False)]
        return f"## {title}\n\n" + "\n".join(rows) + "\n"

    parts = ["# GlassQuill results\n",
             "Generated by `run_pipeline.py`. Positive class = AI. FPR = share of human texts wrongly flagged as AI.\n",
             table("results_in_domain.csv", "Test set (HC3 + HAP-E + M4GT)"),
             table("results_in_domain_by_corpus.csv", "Test set, per corpus"),
             table("test_accuracy_by_generator.csv", "Test accuracy per generator"),
             table("results_out_of_domain.csv", "Out-of-domain (MAGE)"),
             table("results_paraphrase.csv", "Paraphrased AI text (recall)"),
             table("feature_means_by_class.csv", "Feature averages by class (RQ1)"),
             table("shap_global.csv", "Global SHAP importance and direction (XGBoost)")]
    agree = path(REPORTS, "explanation_agreement.json")
    if os.path.exists(agree):
        parts.append("## Explanation agreement (RQ4)\n\n```json\n" + open(agree).read() + "\n```\n")
    open(path(REPORTS, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(parts))


STAGES = {"data": stage_data, "features": stage_features, "train": stage_train, "bert": stage_bert,
          "evaluate": stage_evaluate, "ood": stage_ood, "paraphrase": stage_paraphrase, "explain": stage_explain}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--quick", action="store_true", help="small sample, written to */quick/")
    args = parser.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)   # repeatable results
    setup(args.quick)
    print(f"device: {DEVICE}, quick: {args.quick}")
    for name, stage in STAGES.items():
        if name in args.stages:
            print(f"\n=== {name} ===")
            t0 = time.time()
            stage()
            print(f"=== {name} done in {time.time() - t0:.0f}s")
    write_summary()
    print("\nsummary:", path(REPORTS, "RESULTS.md"))


if __name__ == "__main__":
    main()
