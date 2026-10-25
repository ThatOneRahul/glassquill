"""Train the three classic models and score them with the same metrics."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

# Parallel grid-search workers. Each worker holds its own copy of the data, so "all cores"
# (n_jobs=-1) can run a laptop out of memory; 4 is a safe default.
N_JOBS = 4


def evaluate(name, y_true, score, threshold=0.5):
    pred = (np.asarray(score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"model": name,
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred),
            "f1": f1_score(y_true, pred),
            "roc_auc": roc_auc_score(y_true, score),
            "fpr": fp / (fp + tn)}          # humans wrongly flagged as AI


# Model 1: zero-shot perplexity threshold (no training at all)
def perplexity_threshold(val_logppl, y_val):
    """Lower perplexity = more AI-like, so the score is -log_ppl.
    Pick the cut-off on the validation set with Youden's J."""
    fpr, tpr, thr = roc_curve(y_val, -val_logppl)
    return thr[np.argmax(tpr - fpr)]


# Model 2: TF-IDF + logistic regression (interpretable by design)
def train_tfidf_lr(train, seed=42):
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, random_state=seed))
    grid = GridSearchCV(pipe, {"logisticregression__C": [0.1, 1, 10]},
                        cv=GroupKFold(n_splits=5), scoring="f1", n_jobs=N_JOBS)
    grid.fit(train["text"], train["label"], groups=train["qid"])
    print("best C:", grid.best_params_)
    return grid.best_estimator_


# Expected direction from the guide (section 4.3): +1 = higher value means more AI-like.
EXPECTED_DIRECTION = {"log_ppl": -1, "nll_std": -1, "nll_max": -1, "top1_frac": 1,
                      "top10_frac": 1, "mean_entropy": -1, "std_sent_len": -1,
                      "burstiness": -1, "first_person_rate": -1, "contraction_rate": -1}


def monotone_directions(X, y):
    """Constrain a feature only when the theory and the training data agree on its
    direction. Without this the trees can learn odd rules, e.g. 'very even sentence
    lengths mean human', which break on text unlike the training data."""
    gap = X[y == 1].mean() - X[y == 0].mean()
    return {f: d for f, d in EXPECTED_DIRECTION.items()
            if f in X.columns and np.sign(gap[f]) == d}


# Model 3: XGBoost on style + perplexity features (explained with SHAP)
def train_xgb(X_train, y_train, groups, seed=42, monotone=True):
    cons = monotone_directions(X_train, np.asarray(y_train)) if monotone else {}
    print("monotone constraints:", cons)
    base = XGBClassifier(n_estimators=400, subsample=0.8, colsample_bytree=0.8,
                         eval_metric="logloss", random_state=seed,
                         monotone_constraints=cons or None)
    grid = GridSearchCV(base, {"max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1]},
                        cv=GroupKFold(n_splits=5), scoring="f1", n_jobs=N_JOBS)
    grid.fit(X_train, y_train, groups=groups)
    print("best params:", grid.best_params_)
    return grid.best_estimator_
