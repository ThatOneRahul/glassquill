# %% CELL setup
import os, json, random
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
for d in ["data", "models", "reports/figures"]:
    os.makedirs(d, exist_ok=True)
print("device:", dev)

# %% CELL data
from src.data import load_hc3, prepare, group_split

raw = load_hc3()
print(raw.shape, raw["label"].value_counts().to_dict())
df = prepare(raw, per_class=6000)
train, val, test = group_split(df, seed=SEED)
for name, d in [("train", train), ("val", val), ("test", test)]:
    d.to_parquet(f"data/{name}.parquet")
    print(name, len(d), "share AI:", round(d["label"].mean(), 3))

# %% CELL eda
print(df.groupby(["source", "label"]).size().unstack())       # texts per domain
fig, ax = plt.subplots(figsize=(6, 4))
for lab, name in [(0, "Human"), (1, "AI")]:
    ax.hist(df.loc[df["label"] == lab, "n_words"], bins=40, alpha=0.6, label=name)
ax.set_xlabel("words per text"); ax.legend()
fig.savefig("reports/figures/length_hist.png", dpi=200)

# %% CELL features
from src.features import PerplexityScorer, build_feature_table

scorer = PerplexityScorer("gpt2")                 # uses the Colab GPU if present
X = {}
for name, d in [("train", train), ("val", val), ("test", test)]:
    X[name] = build_feature_table(d["text"], scorer)
    X[name].to_parquet(f"data/X_{name}.parquet")
FEATS = list(X["train"].columns)
print(X["train"].assign(label=train["label"]).groupby("label").mean().T.round(3))

# %% CELL train
from src.models import evaluate, perplexity_threshold, train_tfidf_lr, train_xgb

thr = perplexity_threshold(X["val"]["log_ppl"].values, val["label"].values)
lr = train_tfidf_lr(train, seed=SEED)
xgb = train_xgb(X["train"][FEATS], train["label"], groups=train["qid"], seed=SEED)

# %% CELL bert
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.train_distilbert import train_distilbert, predict_proba

train_distilbert(train, val, out_dir="models/distilbert", seed=SEED)
bert = AutoModelForSequenceClassification.from_pretrained("models/distilbert")
bert_tok = AutoTokenizer.from_pretrained("models/distilbert")

# %% CELL evaluate
def score_all(d, Xd):
    """Score and threshold for every model. Higher score = more AI-like."""
    return {
        "Perplexity threshold": (-Xd["log_ppl"].values, thr),
        "TF-IDF + LR": (lr.predict_proba(d["text"])[:, 1], 0.5),
        "XGBoost": (xgb.predict_proba(Xd[FEATS])[:, 1], 0.5),
        "DistilBERT": (predict_proba(list(d["text"]), bert, bert_tok, device=dev)[:, 1], 0.5),
    }

scores_test = score_all(test, X["test"])
results = pd.DataFrame([evaluate(m, test["label"], s, t) for m, (s, t) in scores_test.items()])
results.round(3).to_csv("reports/results_in_domain.csv", index=False)
print(results.round(3).to_string(index=False))

# %% CELL plots
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (m, (s, t)) in zip(axes, scores_test.items()):
    ConfusionMatrixDisplay.from_predictions(
        test["label"], (s >= t).astype(int), display_labels=["Human", "AI"], ax=ax, colorbar=False)
    ax.set_title(m)
plt.tight_layout(); fig.savefig("reports/figures/confusion_matrices.png", dpi=200)

fig, ax = plt.subplots(figsize=(6, 5))
for m, (s, t) in scores_test.items():
    RocCurveDisplay.from_predictions(test["label"], s, name=m, ax=ax)
fig.savefig("reports/figures/roc_curves.png", dpi=200)

# %% CELL ood
from src.data import clean_text

MAGE = ("https://huggingface.co/datasets/yaful/MAGE/resolve/"
        "refs%2Fconvert%2Fparquet/default/test/0000.parquet")
ood = pd.read_parquet(MAGE)
ood["label"] = 1 - ood["label"]                  # MAGE uses 1 = human. We use 1 = AI
ood["text"] = (ood["text"].str.replace(r"(?s)Hello, users of CMV!.*$", "", regex=True)
               .map(clean_text))                 # drop the Reddit moderator footer
ood["n_words"] = ood["text"].str.split().str.len()
ood = (ood[ood["n_words"].between(50, 400)]
       .groupby("label").sample(n=1000, random_state=SEED).reset_index(drop=True))

X_ood = build_feature_table(ood["text"], scorer)
scores_ood = score_all(ood, X_ood)
ood_results = pd.DataFrame([evaluate(m, ood["label"], s, t)
                            for m, (s, t) in scores_ood.items()])
ood_results.round(3).to_csv("reports/results_out_of_domain.csv", index=False)
print(ood_results.round(3).to_string(index=False))

# which generators fool the main model most?
correct = (scores_ood["XGBoost"][0] >= 0.5).astype(int) == ood["label"]
print(ood.assign(correct=correct).groupby("src")["correct"].agg(["mean", "size"])
      .sort_values("mean").head(10))

# %% CELL paraphrase
from transformers import AutoModelForSeq2SeqLM
from src.features import SENT_SPLIT

PP = "humarin/chatgpt_paraphraser_on_T5_base"
pp_tok = AutoTokenizer.from_pretrained(PP)
pp = AutoModelForSeq2SeqLM.from_pretrained(PP).to(dev)

def paraphrase(text):
    out = []
    for s in SENT_SPLIT.split(text):
        enc = pp_tok("paraphrase: " + s, return_tensors="pt", truncation=True,
                     max_length=128).to(dev)
        ids = pp.generate(**enc, num_beams=4, max_length=128, no_repeat_ngram_size=3)
        out.append(pp_tok.decode(ids[0], skip_special_tokens=True))
    return " ".join(out)

ai_test = test[test["label"] == 1].sample(n=150, random_state=SEED).reset_index(drop=True)
para = ai_test.assign(text=ai_test["text"].map(paraphrase))
before = score_all(ai_test, build_feature_table(ai_test["text"], scorer))
after = score_all(para, build_feature_table(para["text"], scorer))
robust = pd.DataFrame({m: {"recall_before": (before[m][0] >= t).mean(),
                           "recall_after": (after[m][0] >= t).mean()}
                       for m, (_, t) in before.items()}).T
print(robust.round(3))

# %% CELL shap
import shap
from src.explain import shap_global, shap_local_figure, plain_reasons

sv = shap_global(xgb, X["test"][FEATS])               # saves the beeswarm plot
shap.plots.bar(sv, max_display=15, show=False)
plt.tight_layout(); plt.savefig("reports/figures/shap_bar.png", dpi=200); plt.close()

p = scores_test["XGBoost"][0]; y = test["label"].values; pred = (p >= 0.5).astype(int)
def first(mask):
    idx = np.where(mask)[0]
    return int(idx[0]) if len(idx) else None

cases = {"true_ai": first((y == 1) & (pred == 1)),
         "true_human": first((y == 0) & (pred == 0)),
         "false_positive": first((y == 0) & (pred == 1)),
         "false_negative": first((y == 1) & (pred == 0))}
explainer = shap.TreeExplainer(xgb)
for name, i in cases.items():
    if i is None:
        continue
    fig, row = shap_local_figure(explainer, X["test"][FEATS].iloc[[i]])
    fig.savefig(f"reports/figures/waterfall_{name}.png", dpi=200); plt.close(fig)
    print(f"--- {name}  P(AI) = {p[i]:.2f}\n{plain_reasons(row)}\n{test['text'][i][:300]}\n")

# %% CELL agreement
from transformers import pipeline
from src.explain import lime_explain, lr_word_weights

clf = pipeline("text-classification", model=bert, tokenizer=bert_tok, top_k=None,
               device=0 if dev == "cuda" else -1)
shap_text = shap.Explainer(clf)

def top_shap(text, k=10):
    e = shap_text([text])[0, :, "AI"]
    order = np.argsort(-np.abs(e.values))[:k]
    return {str(e.data[i]).strip().lower() for i in order}

def top_lime(text, k=10):
    f = lambda t: predict_proba(t, bert, bert_tok, device=dev)
    return {w.lower() for w, _ in lime_explain(f, text, num_features=k)}

sample = test["text"].sample(20, random_state=SEED)
jac = []
for t in sample:
    a, b = top_shap(t), top_lime(t)
    jac.append(len(a & b) / len(a | b))
print("DistilBERT: SHAP vs LIME top-10 overlap (Jaccard):", round(float(np.mean(jac)), 2))

# LIME sanity check on the glass-box model, where the true word weights are known
hits = []
for t in sample:
    exact = lr_word_weights(lr, t)
    ranked = sorted(exact.items(), key=lambda kv: -abs(kv[1]))
    true_top = [w for w, _ in ranked if " " not in w][:10]        # single words only
    lime_top = [w.lower() for w, _ in lime_explain(lr.predict_proba, t, num_features=10)]
    hits.append(len(set(true_top) & set(lime_top)) / 10)
print("LIME recovers the true top words of LR:", round(float(np.mean(hits)), 2))

# coloured text view of one AI text (shows in the notebook)
shap.plots.text(shap_text([test["text"][cases["true_ai"]]])[0, :, "AI"])

# %% CELL save
import joblib

joblib.dump(xgb, "models/xgb.joblib")
joblib.dump(lr, "models/tfidf_lr.joblib")
with open("models/feature_names.json", "w") as f:
    json.dump(FEATS, f)
