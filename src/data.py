"""Load HC3, clean the text, balance classes and make a leak-free split."""
import re
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

HC3_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"

# Phrases that give the answer away. If we keep them the model learns a shortcut.
GIVEAWAYS = [
    r"as an ai language model[^.]*\.?",
    r"i'?m sorry,? but i[^.]*\.?",
    r"i am not able to[^.]*\.?",
    r"i hope this helps[^.]*\.?",
]


def load_hc3(url=HC3_URL):
    """One row per answer: qid, source, text, label (0 = human, 1 = AI)."""
    raw = pd.read_json(url, lines=True)
    rows = []
    for qid, r in raw.iterrows():
        for t in r["human_answers"]:
            rows.append({"qid": qid, "source": r["source"], "text": t, "label": 0})
        for t in r["chatgpt_answers"]:
            rows.append({"qid": qid, "source": r["source"], "text": t, "label": 1})
    return pd.DataFrame(rows)


def clean_text(t):
    t = str(t).replace("\\n", " ").replace("\n", " ")   # HC3 stores literal "\n"
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)              # "word ," -> "word,"
    for p in GIVEAWAYS:
        t = re.sub(p, "", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


def prepare(df, min_words=50, max_words=400, per_class=6000, seed=42):
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].map(clean_text)
    df = df.drop_duplicates("text")
    df["n_words"] = df["text"].str.split().str.len()
    df = df[df["n_words"].between(min_words, max_words)]
    n = min(per_class, df["label"].value_counts().min())   # same count per class
    df = df.groupby("label").sample(n=n, random_state=seed)
    return df.reset_index(drop=True)


def group_split(df, seed=42):
    """70/15/15 split where one question never appears in two splits."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr, rest = next(gss.split(df, groups=df["qid"]))
    train, rest = df.iloc[tr], df.iloc[rest]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    va, te = next(gss2.split(rest, groups=rest["qid"]))
    val, test = rest.iloc[va], rest.iloc[te]
    assert not set(train.qid) & set(test.qid), "question leak between train and test"
    return [d.reset_index(drop=True) for d in (train, val, test)]
