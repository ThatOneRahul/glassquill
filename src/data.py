"""Load HC3, clean the text, balance classes and make a leak-free split."""
import re
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

HC3_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"
MAGE_URL = ("https://huggingface.co/datasets/yaful/MAGE/resolve/"
            "refs%2Fconvert%2Fparquet/default/test/0000.parquet")

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
    # HC3 human answers were run through a word tokenizer and AI answers were not.
    # Each pattern below is in 2-35% of human texts and almost no AI texts.
    t = re.sub(r"\bURL_\d+\b", "", t)                   # removed links: "see URL_0"
    t = re.sub(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", t, flags=re.IGNORECASE)  # "do n't"
    t = re.sub(r"\(\s+", "(", t)                        # "( word )" -> "(word)"
    t = re.sub(r"\s+\)", ")", t)
    t = re.sub(r"(?<!\w)' ([^'\n]{1,60}?) '(?!\w)", r"'\1'", t)   # "' word '" -> "'word'"
    t = re.sub(r"(\w) ' ", r"\1' ", t)                  # "students ' books"
    t = t.replace(" -- ", " - ")
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)              # "word ," -> "word,"
    for p in GIVEAWAYS:
        t = re.sub(p, "", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


def prepare(df, min_words=50, max_words=400, per_class=6000, balance_by="source", seed=42):
    """Clean, filter by length and sample the same number of texts per class.

    With balance_by="source" every source also gets as many human as AI texts.
    Without it, open_qa ends up almost all AI and reddit_eli5 mostly human, so a
    model can score well by spotting the topic instead of the writing style.
    """
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].map(clean_text)
    df = df.drop_duplicates("text")
    df["n_words"] = df["text"].str.split().str.len()
    df = df[df["n_words"].between(min_words, max_words)]
    if not balance_by:
        n = min(per_class, df["label"].value_counts().min())   # same count per class
        return df.groupby("label").sample(n=n, random_state=seed).reset_index(drop=True)
    pairs = df.groupby([balance_by, "label"]).size().unstack(fill_value=0).min(axis=1)
    scale = min(1.0, per_class / pairs.sum())                 # shrink every group alike
    parts = [g.sample(n=int(pairs[key] * scale), random_state=seed)
             for (key, _), g in df.groupby([balance_by, "label"])]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_mage(url=MAGE_URL, per_class=1000, min_words=50, max_words=400, seed=42):
    """MAGE test split, relabelled to our convention and balanced (section 3.2)."""
    ood = pd.read_parquet(url)
    ood["label"] = 1 - ood["label"]                  # MAGE uses 1 = human. We use 1 = AI
    ood["text"] = (ood["text"].str.replace(r"(?s)Hello, users of CMV!.*$", "", regex=True)
                   .map(clean_text))                 # drop the Reddit moderator footer
    ood["n_words"] = ood["text"].str.split().str.len()
    ood = ood[ood["n_words"].between(min_words, max_words)]
    n = min(per_class, ood["label"].value_counts().min())
    return ood.groupby("label").sample(n=n, random_state=seed).reset_index(drop=True)


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
