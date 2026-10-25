"""Explanations: SHAP for XGBoost, word weights for TF-IDF LR, LIME for DistilBERT."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from lime.lime_text import LimeTextExplainer

NICE_NAMES = {
    "log_ppl": "Perplexity (log)", "nll_std": "Unevenness of surprise",
    "nll_max": "Most surprising token", "top1_frac": "Share of top-1 guessable words",
    "top10_frac": "Share of top-10 guessable words", "mean_entropy": "GPT-2 uncertainty",
    "avg_sent_len": "Average sentence length", "std_sent_len": "Sentence length spread",
    "burstiness": "Burstiness", "type_token_ratio": "Vocabulary richness",
    "avg_word_len": "Average word length", "comma_rate": "Commas per word",
    "punct_rate": "Punctuation per word", "function_word_rate": "Function word rate",
    "first_person_rate": "First-person words", "contraction_rate": "Contractions",
    "flesch_reading_ease": "Reading ease", "fk_grade": "Grade level",
}


def shap_global(model, X, out_path="reports/figures/shap_beeswarm.png"):
    sv = shap.TreeExplainer(model)(X)
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
    return sv


def shap_local_figure(explainer, x_row):
    """Waterfall plot for one text. x_row is a one-row DataFrame."""
    sv = explainer(x_row)
    shown = shap.Explanation(                       # readable names on the plot only
        values=sv[0].values, base_values=sv[0].base_values, data=sv[0].data,
        feature_names=[NICE_NAMES.get(n, n) for n in sv[0].feature_names])
    shap.plots.waterfall(shown, max_display=10, show=False)
    fig = plt.gcf(); fig.set_size_inches(8, 5); plt.tight_layout()
    return fig, sv[0]


def plain_reasons(sv_row, top_k=3):
    """Turn the biggest SHAP values into short sentences for the user."""
    order = np.argsort(-np.abs(sv_row.values))[:top_k]
    lines = []
    for i in order:
        name = sv_row.feature_names[i]
        side = "AI" if sv_row.values[i] > 0 else "human"
        lines.append(f"- {NICE_NAMES.get(name, name)} = {sv_row.data[i]:.2f} "
                     f"pushed the score towards {side}")
    return "\n".join(lines)


def lr_word_weights(pipe, text):
    """Contribution of each word in the text for the TF-IDF + LR model."""
    vec = pipe.named_steps["tfidfvectorizer"]
    lr = pipe.named_steps["logisticregression"]
    x = vec.transform([text]).multiply(lr.coef_[0]).tocsr()
    vocab = vec.get_feature_names_out()
    return {vocab[j]: v for j, v in zip(x.indices, x.data)}


def highlight(text, weights):
    """(word, score) pairs for the coloured word view. Positive = towards AI."""
    top = max((abs(v) for v in weights.values()), default=1.0) or 1.0
    out = []
    for w in text.split():
        key = w.lower().strip(".,!?;:\"'()")
        out.append((w + " ", float(weights[key] / top) if key in weights else None))
    return out


def lime_explain(predict_proba, text, num_features=10, num_samples=500):
    explainer = LimeTextExplainer(class_names=["Human", "AI"], random_state=42)
    exp = explainer.explain_instance(text, predict_proba, num_features=num_features,
                                     num_samples=num_samples)
    return exp.as_list()   # [(word, weight), ...]
