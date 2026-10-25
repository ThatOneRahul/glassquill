"""What the website uses: load the saved models once, then analyse one text."""
import json

import joblib
import pandas as pd
import shap

from src.explain import NICE_NAMES, highlight, lr_word_weights
from src.features import PerplexityScorer, stylometric_features


class Detector:
    def __init__(self, model_dir="models", device=None):
        self.xgb = joblib.load(f"{model_dir}/xgb.joblib")          # style model
        self.lr = joblib.load(f"{model_dir}/tfidf_lr.joblib")      # word model
        with open(f"{model_dir}/feature_names.json") as f:
            self.feature_names = json.load(f)
        self.explainer = shap.TreeExplainer(self.xgb)
        self.scorer = PerplexityScorer("gpt2", device=device)

    def analyze(self, text):
        # the 18 features: 12 about writing style + 6 about how predictable GPT-2 finds the text
        feats = stylometric_features(text)
        feats.update(self.scorer.features(text))
        x = pd.DataFrame([feats])[self.feature_names]

        p_style = float(self.xgb.predict_proba(x)[0, 1])
        p_words = float(self.lr.predict_proba([text])[0, 1])
        sv = self.explainer(x)[0]                                  # SHAP: why the style model decided
        contributions = sorted(
            ({"feature": n, "name": NICE_NAMES.get(n, n), "value": float(v), "shap": float(s)}
             for n, v, s in zip(sv.feature_names, sv.data, sv.values)),
            key=lambda c: -abs(c["shap"]))
        return {
            # verdict = mean of the two explainable models: on the validation set this beat
            # either model alone and DistilBERT (accuracy 0.972, false positive rate 0.021)
            "p_final": (p_style + p_words) / 2,
            "p_ai": p_style,
            "p_ai_lr": p_words,
            "shap_base": float(sv.base_values),   # log-odds; base + sum of SHAP values = output
            "contributions": contributions,
            "highlights": highlight(text, lr_word_weights(self.lr, text)),
        }
