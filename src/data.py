"""Load HC3, clean the text, balance classes and make a leak-free split."""
import re
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

HC3_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"
MAGE_URL = ("https://huggingface.co/datasets/yaful/MAGE/resolve/"
            "refs%2Fconvert%2Fparquet/default/test/0000.parquet")

# Modern LLM text, so the detector is not stuck on 2022 ChatGPT.
# HAP-E (MIT): human text continued by GPT-4o / Llama-3 in 6 genres. Chunk 2 is the
# human continuation; chunk 1 was the prompt and is not used.
HAPE_URL = ("https://huggingface.co/datasets/browndw/human-ai-parallel-corpus/"
            "resolve/main/text_data/hape-text_{}.parquet")
HAPE_FILES = ["human-chunk-2", "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18",
              "llama-3-70B-Instruct", "llama-3-8B-Instruct"]
# M4GT part of the COLING 2025 shared task data. Its HC3 and MAGE parts are skipped
# so that MAGE stays a clean out-of-domain test.
M4GT_URL = ("https://huggingface.co/datasets/Jinyan1/COLING_2025_MGT_en/"
            "resolve/main/data/dev-00000-of-00001.parquet")
M4GT_MODELS = {"gpt4o", "gpt4", "gpt-3.5-turbo", "mixtral-8x7b", "llama3-8b", "llama3-70b",
               "gemma-7b-it", "gemma2-9b-it", "cohere"}

# Phrases that give the answer away. If we keep them the model learns a shortcut.
GIVEAWAYS = [
    r"as an ai language model[^.]*\.?",
    r"i'?m sorry,? but i[^.]*\.?",
    r"i am not able to[^.]*\.?",
    r"i hope this helps[^.]*\.?",
]


TYPOGRAPHY = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                            "—": " - ", "–": "-", "…": "...", " ": " "})


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
    # Typography and markdown differ by source, not by author: curly quotes are
    # commoner in scraped human text, ** and # headings in raw LLM output (and
    # they vanish when someone copies from a chat window). Normalise both.
    t = t.translate(TYPOGRAPHY)
    t = re.sub(r"\*\*|__", "", t)                       # **bold**
    t = re.sub(r"(^|\s)#{1,6}\s", r"\1", t)             # "## Heading"
    t = re.sub(r"\[\d+(?:[,-]\s*\d+)*\]", "", t)        # citation markers [1], [2, 3]
    # HC3 human answers were run through a word tokenizer and AI answers were not.
    # Each pattern below is in 2-35% of human texts and almost no AI texts.
    t = re.sub(r"\[?_?URL_\d+_?\]?", "", t)             # removed links: "see URL_0"
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


def prepare(df, min_words=50, max_words=400, per_class=6000, balance_by="source", seed=42,
            truncate=False):
    """Clean, filter by length and sample the same number of texts per class.

    With balance_by="source" every source also gets as many human as AI texts.
    Without it, open_qa ends up almost all AI and reddit_eli5 mostly human, so a
    model can score well by spotting the topic instead of the writing style.
    """
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].map(clean_text)
    if truncate:
        df["text"] = df["text"].map(lambda t: truncate_words(t, max_words))
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


def load_hape(paths):
    """paths: {file name in HAPE_FILES: local path}. Group = the source document."""
    rows = []
    for name, path in paths.items():
        d = pd.read_parquet(path)
        doc = d["doc_id"].str.split("@").str[0]                 # "acad_0001"
        rows.append(pd.DataFrame({
            "qid": "hape:" + doc, "source": "hape/" + doc.str.split("_").str[0],
            "text": d["text"], "label": int(not name.startswith("human")),
            "generator": "human" if name.startswith("human") else name}))
    return pd.DataFrame(pd.concat(rows, ignore_index=True))


def load_m4gt(path):
    """Human texts and texts from recent instruction-tuned LLMs, six domains."""
    d = pd.read_parquet(path, columns=["id", "source", "sub_source", "model", "label", "text"])
    d = d[(d["source"] == "m4gt") & ((d["label"] == 0) | d["model"].isin(M4GT_MODELS))]
    return pd.DataFrame({"qid": "m4gt:" + d["id"].astype(str),
                         "source": "m4gt/" + d["sub_source"], "text": d["text"],
                         "label": d["label"].astype(int),
                         "generator": d["model"].where(d["label"] == 1, "human")})


def truncate_words(t, max_words=400):
    """Cut long texts at the last sentence end before max_words, so long human and
    AI texts both end up in the 50-400 word window instead of being dropped."""
    words = t.split()
    if len(words) <= max_words:
        return t
    cut = " ".join(words[:max_words])
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[:end + 1] if end > len(cut) // 2 else cut


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
