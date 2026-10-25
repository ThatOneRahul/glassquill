"""Run the whole GlassQuill study locally, step by step (guide sections 7.4 to 7.10).

Same steps as notebook_cells.py, but every stage saves its output and the next
stage loads it from disk, so a crash or a rerun never starts from zero.

    python run_pipeline.py                      # every stage
    python run_pipeline.py --stages ood explain # only these stages
    python run_pipeline.py --quick              # tiny sample, separate folders, for a smoke test
"""
import argparse
import json
import os
import random
import re
import time
import urllib.request

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data import (HAPE_FILES, HAPE_URL, HC3_URL, M4GT_URL, MAGE_URL, group_split,
                      load_hape, load_hc3, load_m4gt, load_mage, prepare)

STAGES = ["data", "features", "train", "bert", "evaluate", "ood", "paraphrase", "explain"]
SEED = 42


class Run:
    """Paths, settings and lazily loaded artefacts shared by the stages."""

    def __init__(self, quick=False):
        sub = "quick" if quick else ""
        self.data = os.path.join("data", sub)
        self.models = os.path.join("models", sub)
        self.reports = os.path.join("reports", sub)
        self.figs = os.path.join(self.reports, "figures")
        for d in [self.data, self.models, self.figs]:
            os.makedirs(d, exist_ok=True)
        self.cache = "data"                         # raw downloads are shared by both modes
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.corpora = ["hc3", "hape", "m4gt"]
        # texts per class and corpus before the split (HC3 as in the guide)
        self.per_class = ({"hc3": 300, "hape": 200, "m4gt": 300} if quick else
                          {"hc3": 6000, "hape": 4000, "m4gt": 6000})
        self.ood_per_class = 100 if quick else 1000
        # kept small so the robustness and explanation stages take minutes, not an hour
        self.n_para = 20 if quick else 60
        self.n_agree = 5 if quick else 10
        self.bert_epochs = 1 if quick else 2
        self._mem = {}

    # ---- small helpers -------------------------------------------------
    def p(self, *parts):
        return os.path.join(*parts)

    def split(self, name):
        return self._load(name, lambda: pd.read_parquet(self.p(self.data, f"{name}.parquet")))

    def X(self, name):
        return self._load("X_" + name,
                          lambda: pd.read_parquet(self.p(self.data, f"X_{name}.parquet")))

    def scorer(self):
        from src.features import PerplexityScorer
        return self._load("scorer", lambda: PerplexityScorer("gpt2", device=self.dev))

    def feats(self):
        with open(self.p(self.models, "feature_names.json")) as f:
            return json.load(f)

    def thr(self):
        with open(self.p(self.models, "perplexity_threshold.json")) as f:
            return json.load(f)["threshold"]

    def lr(self):
        return self._load("lr", lambda: joblib.load(self.p(self.models, "tfidf_lr.joblib")))

    def xgb(self):
        return self._load("xgb", lambda: joblib.load(self.p(self.models, "xgb.joblib")))

    def bert(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = self.p(self.models, "distilbert")
        return self._load("bert", lambda: (
            AutoModelForSequenceClassification.from_pretrained(path).to(self.dev).eval(),
            AutoTokenizer.from_pretrained(path)))

    def bert_proba(self, texts):
        from src.train_distilbert import predict_proba
        model, tok = self.bert()
        return predict_proba(list(texts), model, tok, device=self.dev)

    def features_of(self, name, texts):
        """Feature table for any text set, cached as data/X_<name>.parquet."""
        path = self.p(self.data, f"X_{name}.parquet")
        if not os.path.exists(path):
            from src.features import build_feature_table
            t0 = time.time()
            build_feature_table(list(texts), self.scorer()).to_parquet(path)
            print(f"  features for {name}: {len(texts)} texts in {time.time() - t0:.0f}s")
        return self.X(name)

    def score_all(self, d, Xd):
        """Score and threshold for every model. Higher score = more AI-like."""
        lr = self.lr().predict_proba(d["text"])[:, 1]
        xgb = self.xgb().predict_proba(Xd[self.feats()])[:, 1]
        return {
            "Perplexity threshold": (-Xd["log_ppl"].values, self.thr()),
            "TF-IDF + LR": (lr, 0.5),
            "XGBoost": (xgb, 0.5),
            "DistilBERT": (self.bert_proba(d["text"])[:, 1], 0.5),
            "XGBoost + LR (app)": ((xgb + lr) / 2, 0.5),   # the verdict the app shows
        }

    def _load(self, key, fn):
        if key not in self._mem:
            self._mem[key] = fn()
        return self._mem[key]


def download(url, path):
    if not os.path.exists(path):
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


# ---- stages ------------------------------------------------------------
def load_corpus(r, name):
    """Raw rows of one training corpus: qid (group), source, text, label, generator."""
    if name == "hc3":
        d = load_hc3(download(HC3_URL, r.p(r.cache, "hc3_all.jsonl")))
        return d.assign(generator=np.where(d["label"] == 1, "chatgpt", "human"))
    if name == "hape":
        os.makedirs(r.p(r.cache, "hape"), exist_ok=True)
        return load_hape({f: download(HAPE_URL.format(f), r.p(r.cache, "hape", f + ".parquet"))
                          for f in HAPE_FILES})
    return load_m4gt(download(M4GT_URL, r.p(r.cache, "coling_dev.parquet")))


def stage_data(r):
    """Step 3: load, clean, balance and split every corpus by its groups, then EDA.

    Each corpus is split on its own (HC3 by question, HAP-E by source document,
    M4GT by text), then the three train / val / test parts are joined.
    """
    parts = []
    for name in r.corpora:
        raw = load_corpus(r, name)
        print(f"  {name} raw:", raw.shape, raw["label"].value_counts().to_dict())
        df = prepare(raw, per_class=r.per_class[name], seed=SEED, truncate=name != "hc3")
        parts.append([d.assign(corpus=name) for d in group_split(df, seed=SEED)])
    train, val, test = [pd.concat(p, ignore_index=True).astype({"qid": str})
                        for p in zip(*parts)]
    df = pd.concat([train, val, test], ignore_index=True)
    for name, d in [("train", train), ("val", val), ("test", test)]:
        d.to_parquet(r.p(r.data, f"{name}.parquet"))
        print(f"  {name}: {len(d)} texts, share AI {d['label'].mean():.3f}")
        old = r.p(r.data, f"X_{name}.parquet")      # a new split makes old features stale
        if os.path.exists(old):
            os.remove(old)

    counts = df.groupby(["source", "label"]).size().unstack().rename(columns={0: "human", 1: "ai"})
    counts.to_csv(r.p(r.reports, "eda_source_counts.csv"))
    print(counts)
    fig, ax = plt.subplots(figsize=(6, 4))
    for lab, name in [(0, "Human"), (1, "AI")]:
        ax.hist(df.loc[df["label"] == lab, "n_words"], bins=40, alpha=0.6, label=name)
    ax.set_xlabel("words per text"); ax.set_ylabel("texts"); ax.legend()
    plt.tight_layout(); fig.savefig(r.p(r.figs, "length_hist.png"), dpi=200); plt.close(fig)


def stage_features(r):
    """Step 4: 12 style + 6 GPT-2 features for every split (RQ1 table)."""
    for name in ["train", "val", "test"]:
        r.features_of(name, r.split(name)["text"])
    X, y = r.X("train"), r.split("train")["label"]
    with open(r.p(r.models, "feature_names.json"), "w") as f:
        json.dump(list(X.columns), f)
    means = X.assign(label=y.values).groupby("label").mean().T
    means.columns = ["human_mean", "ai_mean"]
    means.index.name = "feature"
    # standardised gap: how many pooled standard deviations apart the two classes are
    means["effect_size"] = (means["ai_mean"] - means["human_mean"]) / X.std().replace(0, np.nan)
    means = means.reindex(means["effect_size"].abs().sort_values(ascending=False).index)
    means.round(4).to_csv(r.p(r.reports, "feature_means_by_class.csv"))
    print(means.round(3))


def stage_train(r):
    """Step 5: perplexity threshold, TF-IDF + LR and XGBoost, tuned with GroupKFold."""
    from src.models import perplexity_threshold, train_tfidf_lr, train_xgb
    train, val = r.split("train"), r.split("val")
    feats = r.feats()
    thr = float(perplexity_threshold(r.X("val")["log_ppl"].values, val["label"].values))
    with open(r.p(r.models, "perplexity_threshold.json"), "w") as f:
        json.dump({"threshold": thr, "rule": "AI if -log_ppl >= threshold"}, f, indent=2)
    print(f"  perplexity threshold (Youden J on val): log_ppl <= {-thr:.3f} means AI")
    lr = train_tfidf_lr(train, seed=SEED)
    xgb = train_xgb(r.X("train")[feats], train["label"], groups=train["qid"], seed=SEED)
    joblib.dump(lr, r.p(r.models, "tfidf_lr.joblib"))
    joblib.dump(xgb, r.p(r.models, "xgb.joblib"))
    r._mem.pop("lr", None); r._mem.pop("xgb", None)


def stage_bert(r):
    """Step 6: fine-tune DistilBERT, keep the epoch with the best validation F1."""
    from src.train_distilbert import train_distilbert
    out = r.p(r.models, "distilbert")
    train_distilbert(r.split("train"), r.split("val"), out_dir=out,
                     epochs=r.bert_epochs, seed=SEED)
    r._mem.pop("bert", None)


def stage_evaluate(r):
    """Step 7a: in-domain metrics, confusion matrices and ROC curves on HC3 test."""
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
    from src.models import evaluate
    test = r.split("test")
    scores = r.score_all(test, r.X("test"))
    np.savez(r.p(r.data, "scores_test.npz"), **{m: s for m, (s, _) in scores.items()})
    res = pd.DataFrame([evaluate(m, test["label"], s, t) for m, (s, t) in scores.items()])
    res.round(4).to_csv(r.p(r.reports, "results_in_domain.csv"), index=False)
    print(res.round(3).to_string(index=False))

    # the same metrics inside each corpus, and accuracy for each generator
    by_corpus = pd.DataFrame([{"corpus": c, **evaluate(m, test["label"][k], s[k], t)}
                              for c, k in test.groupby("corpus").groups.items()
                              for m, (s, t) in scores.items()])
    by_corpus.round(4).to_csv(r.p(r.reports, "results_in_domain_by_corpus.csv"), index=False)
    print(by_corpus.pivot(index="model", columns="corpus", values="f1").round(3))
    hits = pd.DataFrame({m: ((s >= t).astype(int) == test["label"]).astype(float)
                         for m, (s, t) in scores.items()})
    gen = hits.assign(generator=test["generator"]).groupby("generator")
    gen.mean().assign(n=gen.size()).sort_values("XGBoost").round(3).to_csv(
        r.p(r.reports, "test_accuracy_by_generator.csv"))

    fig, axes = plt.subplots(1, len(scores), figsize=(4 * len(scores), 4))
    for ax, (m, (s, t)) in zip(axes, scores.items()):
        ConfusionMatrixDisplay.from_predictions(
            test["label"], (s >= t).astype(int), display_labels=["Human", "AI"],
            ax=ax, colorbar=False)
        ax.set_title(m)
    plt.tight_layout(); fig.savefig(r.p(r.figs, "confusion_matrices.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for m, (s, _) in scores.items():
        RocCurveDisplay.from_predictions(test["label"], s, name=m, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.tight_layout(); fig.savefig(r.p(r.figs, "roc_curves.png"), dpi=200); plt.close(fig)


def stage_ood(r):
    """Step 7b: out-of-domain test on MAGE (other LLMs, other domains)."""
    from src.models import evaluate
    path = r.p(r.data, "mage_sample.parquet")
    if not os.path.exists(path):
        load_mage(download(MAGE_URL, r.p(r.cache, "mage_test.parquet")),
                  per_class=r.ood_per_class, seed=SEED).to_parquet(path)
    ood = pd.read_parquet(path)
    scores = r.score_all(ood, r.features_of("ood", ood["text"]))
    res = pd.DataFrame([evaluate(m, ood["label"], s, t) for m, (s, t) in scores.items()])
    res.round(4).to_csv(r.p(r.reports, "results_out_of_domain.csv"), index=False)
    print(res.round(3).to_string(index=False))

    # error analysis: which generators and domains fool each model most?
    per_src = ood[["src", "label"]].copy()
    per_src["domain"], per_src["generator"] = zip(*ood["src"].map(mage_source))
    for m, (s, t) in scores.items():
        per_src[m] = ((s >= t).astype(int) == ood["label"]).astype(float)
    for key in ["src", "domain", "generator"]:
        table = per_src.groupby(key).agg(
            n=("label", "size"), share_ai=("label", "mean"),
            **{m: (m, "mean") for m in scores}).sort_values("XGBoost")
        table.round(3).to_csv(r.p(r.reports, f"ood_accuracy_by_{key}.csv"))
    print("  accuracy by generator:\n", table.round(3).to_string())


def mage_source(src):
    """'cmv_machine_continuation_opt_6.7b' -> ('cmv', 'OPT'). Domain and model family."""
    domain = re.split(r"_(?=human|machine|gpt)", src)[0]
    rest = src[len(domain) + 1:]
    if rest == "human":
        return domain, "human"
    if rest.endswith("_para"):
        return domain, "GPT-4 paraphrase"              # MAGE's paraphrased subsets
    families = [("opt", "OPT"), ("bloom", "BLOOM"), ("glm", "GLM-130B"),
                ("davinci", "GPT-3 (davinci)"), ("gpt-3.5", "GPT-3.5"), ("gpt4", "GPT-4"),
                ("t0", "T0"), ("flan", "Flan-T5"), ("gpt_j", "GPT-J"), ("neox", "GPT-NeoX"),
                ("b$", "LLaMA")]
    low = rest.lower()
    return domain, next((name for k, name in families if re.search(k, low)), rest)


def stage_paraphrase(r):
    """Step 7c: recall on AI texts before and after T5 paraphrasing (H4)."""
    test = r.split("test")
    test = test[test["corpus"] == "hc3"]            # same 150 HC3 texts as in the guide
    ai = test[test["label"] == 1].sample(n=min(r.n_para, int(test["label"].sum())),
                                         random_state=SEED).reset_index(drop=True)
    path = r.p(r.data, "paraphrased.parquet")
    if not os.path.exists(path):
        from src.robustness import Paraphraser
        pp = Paraphraser(device=r.dev)
        t0 = time.time()
        para = ai.assign(text=[pp(t) for t in ai["text"]])
        para.to_parquet(path)
        print(f"  paraphrased {len(para)} texts in {time.time() - t0:.0f}s")
        del pp; torch.cuda.empty_cache()
    para = pd.read_parquet(path)
    before = r.score_all(ai, r.features_of("para_before", ai["text"]))
    after = r.score_all(para, r.features_of("para_after", para["text"]))
    robust = pd.DataFrame({m: {"recall_before": (before[m][0] >= t).mean(),
                               "recall_after": (after[m][0] >= t).mean()}
                           for m, (_, t) in before.items()}).T
    robust["drop"] = robust["recall_before"] - robust["recall_after"]
    robust.index.name = "model"
    robust.round(4).to_csv(r.p(r.reports, "results_paraphrase.csv"))
    print(robust.round(3))
    pd.DataFrame({"original": ai["text"], "paraphrased": para["text"]}).head(5).to_csv(
        r.p(r.reports, "paraphrase_examples.csv"), index=False)


def stage_explain(r):
    """Step 8: global and local SHAP, LR top words and the two agreement checks."""
    import shap
    from src.explain import (NICE_NAMES, lime_explain, lr_word_weights, plain_reasons,
                             shap_global, shap_local_figure)
    test, Xt, feats, xgb, lr = r.split("test"), r.X("test"), r.feats(), r.xgb(), r.lr()

    # global SHAP (beeswarm + bar) and the direction each feature pushes (H1, H2)
    sv = shap_global(xgb, Xt[feats], out_path=r.p(r.figs, "shap_beeswarm.png"))
    shap.plots.bar(sv, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(r.p(r.figs, "shap_bar.png"), dpi=200); plt.close("all")
    rows = []
    for j, f in enumerate(feats):
        corr = pd.Series(sv.data[:, j]).corr(pd.Series(sv.values[:, j]), method="spearman")
        rows.append({"feature": f, "name": NICE_NAMES.get(f, f),
                     "mean_abs_shap": np.abs(sv.values[:, j]).mean(),
                     "value_vs_shap_spearman": corr,
                     "higher_value_pushes_towards": "AI" if corr > 0 else "human"})
    pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).round(4).to_csv(
        r.p(r.reports, "shap_global.csv"), index=False)

    # four local cases: correct AI, correct human, false positive, false negative
    p = xgb.predict_proba(Xt[feats])[:, 1]
    y, pred = test["label"].values, (p >= 0.5).astype(int)
    masks = {"true_ai": (y == 1) & (pred == 1), "true_human": (y == 0) & (pred == 0),
             "false_positive": (y == 0) & (pred == 1), "false_negative": (y == 1) & (pred == 0)}
    explainer = shap.TreeExplainer(xgb)
    lines = ["# Local explanations (XGBoost + SHAP)\n"]
    for name, mask in masks.items():
        idx = np.where(mask)[0]
        if not len(idx):
            continue
        i = int(idx[0])
        fig, row = shap_local_figure(explainer, Xt[feats].iloc[[i]])
        fig.savefig(r.p(r.figs, f"waterfall_{name}.png"), dpi=200); plt.close(fig)
        lines += [f"## {name} (P(AI) = {p[i]:.2f}, source: {test['source'][i]})\n",
                  plain_reasons(row), "", f"> {test['text'][i][:500]}...", ""]
    with open(r.p(r.reports, "local_cases.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # glass box: the words with the largest LR coefficients on each side
    vec = lr.named_steps["tfidfvectorizer"]
    coef = lr.named_steps["logisticregression"].coef_[0]
    vocab = vec.get_feature_names_out()
    order = np.argsort(coef)
    top = pd.DataFrame({"human_word": vocab[order[:25]], "human_coef": coef[order[:25]],
                        "ai_word": vocab[order[::-1][:25]], "ai_coef": coef[order[::-1][:25]]})
    top.round(4).to_csv(r.p(r.reports, "lr_top_words.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 7))
    sel = np.r_[order[:15], order[::-1][:15]]
    ax.barh(vocab[sel], coef[sel], color=["#7b4fd6" if c < 0 else "#d64545" for c in coef[sel]])
    ax.set_xlabel("logistic regression coefficient (right = AI, left = human)")
    plt.tight_layout(); fig.savefig(r.p(r.figs, "lr_top_words.png"), dpi=200); plt.close(fig)

    # check 1: SHAP vs LIME top-10 words for DistilBERT; check 2: LIME vs exact LR weights
    from transformers import pipeline
    model, tok = r.bert()
    clf = pipeline("text-classification", model=model, tokenizer=tok, top_k=None,
                   device=0 if r.dev == "cuda" else -1, truncation=True, max_length=256)
    shap_text = shap.Explainer(clf)
    sample = test["text"].sample(r.n_agree, random_state=SEED).tolist()
    jac, hits = [], []
    for t in sample:
        e = shap_text([t])[0, :, "AI"]
        a = {str(e.data[i]).strip().lower() for i in np.argsort(-np.abs(e.values))[:10]}
        b = {w.lower() for w, _ in lime_explain(r.bert_proba, t, num_features=10)}
        jac.append(len(a & b) / max(len(a | b), 1))
        ranked = sorted(lr_word_weights(lr, t).items(), key=lambda kv: -abs(kv[1]))
        true_top = [w for w, _ in ranked if " " not in w][:10]   # single words only
        lime_top = [w.lower() for w, _ in lime_explain(lr.predict_proba, t, num_features=10)]
        hits.append(len(set(true_top) & set(lime_top)) / 10)
    agree = {"distilbert_shap_vs_lime_jaccard_top10": round(float(np.mean(jac)), 3),
             "lime_recovers_lr_true_top10": round(float(np.mean(hits)), 3),
             "n_texts": len(sample)}
    with open(r.p(r.reports, "explanation_agreement.json"), "w") as f:
        json.dump(agree, f, indent=2)
    print(" ", agree)

    # coloured SHAP text view of one AI text, as HTML for the report
    i = int(np.where(masks["true_ai"])[0][0])
    html = shap.plots.text(shap_text([test["text"][i]])[0, :, "AI"], display=False)
    with open(r.p(r.figs, "shap_text_true_ai.html"), "w", encoding="utf-8") as f:
        f.write(html)


def write_summary(r):
    """Collect every result table into one markdown file for the report."""
    def md(path, title):
        if not os.path.exists(path):
            return f"## {title}\n\n_not run yet_\n"
        d = pd.read_csv(path)
        head = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
        body = "".join("| " + " | ".join(
            f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |\n"
            for row in d.itertuples(index=False))
        return f"## {title}\n\n{head}{body}\n"

    rp = lambda n: r.p(r.reports, n)
    parts = ["# GlassQuill results\n",
             "Generated by `run_pipeline.py`. Positive class = AI. "
             "FPR = share of human texts wrongly flagged as AI.\n",
             md(rp("results_in_domain.csv"), "In-domain (HC3 + HAP-E + M4GT test)"),
             md(rp("results_in_domain_by_corpus.csv"), "In-domain, per corpus"),
             md(rp("test_accuracy_by_generator.csv"), "Test accuracy per generator"),
             md(rp("results_out_of_domain.csv"), "Out-of-domain (MAGE test sample)"),
             md(rp("results_paraphrase.csv"), "Paraphrased AI text (recall)"),
             md(rp("feature_means_by_class.csv"), "Feature means by class (RQ1)"),
             md(rp("shap_global.csv"), "Global SHAP importance and direction (XGBoost)")]
    if os.path.exists(rp("explanation_agreement.json")):
        with open(rp("explanation_agreement.json")) as f:
            parts.append("## Explanation agreement (RQ4)\n\n```json\n" + f.read() + "\n```\n")
    with open(rp("RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    ap.add_argument("--quick", action="store_true", help="small sample, writes to */quick/")
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    r = Run(quick=args.quick)
    print(f"device: {r.dev}  quick: {args.quick}")
    for name in STAGES:
        if name in args.stages:
            print(f"\n=== {name} ===")
            t0 = time.time()
            globals()[f"stage_{name}"](r)
            print(f"=== {name} done in {time.time() - t0:.0f}s")
    write_summary(r)
    print(f"\nsummary: {r.p(r.reports, 'RESULTS.md')}")


if __name__ == "__main__":
    main()
