"""Everything the app needs: load saved models once, then analyse a text."""
import json
import joblib
import pandas as pd
import shap

from src.explain import (NICE_NAMES, highlight, lr_word_weights, plain_reasons,
                         shap_local_figure)
from src.features import PerplexityScorer, stylometric_features


class Detector:
    def __init__(self, model_dir="models", device=None, lm_name="gpt2"):
        self.xgb = joblib.load(f"{model_dir}/xgb.joblib")
        self.lr = joblib.load(f"{model_dir}/tfidf_lr.joblib")
        with open(f"{model_dir}/feature_names.json") as f:
            self.feature_names = json.load(f)
        self.explainer = shap.TreeExplainer(self.xgb)
        self.scorer = PerplexityScorer(lm_name, device=device)

    def neural_features(self, text):
        """The only part that needs GPT-2 (and a GPU if one is there)."""
        return self.scorer.features(text)

    def analyze(self, text, neural=None, figure=True):
        """figure=False skips the matplotlib waterfall (the web API sends the numbers)."""
        feats = stylometric_features(text)
        feats.update(neural if neural is not None else self.neural_features(text))
        x = pd.DataFrame([feats])[self.feature_names]
        p_ai = float(self.xgb.predict_proba(x)[0, 1])
        if figure:
            fig, sv_row = shap_local_figure(self.explainer, x)
        else:
            fig, sv_row = None, self.explainer(x)[0]
        contributions = sorted(                      # same numbers as the plot, as plain data
            ({"feature": n, "name": NICE_NAMES.get(n, n), "value": float(v), "shap": float(s)}
             for n, v, s in zip(sv_row.feature_names, sv_row.data, sv_row.values)),
            key=lambda c: -abs(c["shap"]))
        p_ai_lr = float(self.lr.predict_proba([text])[0, 1])
        return {
            # verdict = mean of the two explainable models; on the validation set this
            # beats either one alone and DistilBERT (accuracy 0.972, FPR 0.021)
            "p_final": (p_ai + p_ai_lr) / 2,
            "p_ai": p_ai,
            "p_ai_lr": p_ai_lr,
            "shap_fig": fig,
            "shap_base": float(sv_row.base_values),   # log-odds; base + sum(shap) = output
            "contributions": contributions,
            "reasons": plain_reasons(sv_row),
            "highlights": highlight(text, lr_word_weights(self.lr, text)),
        }
