"""Hand-made features: writing style (stylometry) and GPT-2 perplexity."""
import re
import string
import numpy as np
import pandas as pd
import nltk
import textstat
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:                                    # textstat needs this word list for syllables
    nltk.data.find("corpora/cmudict")
except LookupError:
    nltk.download("cmudict", quiet=True)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z']+")
FUNCTION_WORDS = {"the", "of", "and", "to", "a", "in", "that", "is", "it", "for",
                  "as", "with", "be", "on", "not", "this", "but", "by", "or", "which"}
FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our"}


def stylometric_features(text):
    sents = [s for s in SENT_SPLIT.split(text) if s.strip()]
    words = WORD_RE.findall(text.lower())
    n = max(len(words), 1)
    lens = np.array([len(WORD_RE.findall(s)) for s in sents] or [n])
    return {
        "avg_sent_len": lens.mean(),
        "std_sent_len": lens.std(),
        "burstiness": lens.std() / (lens.mean() + 1e-9),   # spread of sentence length
        "type_token_ratio": len(set(words)) / n,           # vocabulary richness
        "avg_word_len": np.mean([len(w) for w in words]) if words else 0.0,
        "comma_rate": text.count(",") / n,
        "punct_rate": sum(c in string.punctuation for c in text) / n,
        "function_word_rate": sum(w in FUNCTION_WORDS for w in words) / n,
        "first_person_rate": sum(w in FIRST_PERSON for w in words) / n,
        "contraction_rate": sum("'" in w for w in words) / n,
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "fk_grade": textstat.flesch_kincaid_grade(text),
    }


class PerplexityScorer:
    """How surprised GPT-2 is by each token. AI text is usually less surprising."""

    def __init__(self, model_name="gpt2", device=None, max_tokens=256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        self.max_tokens = max_tokens

    @torch.no_grad()
    def token_scores(self, text):
        enc = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=self.max_tokens).to(self.device)
        ids = enc["input_ids"]
        logp = torch.log_softmax(self.model(**enc).logits[0, :-1].float(), dim=-1)
        target = ids[0, 1:]
        tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        ranks = (logp > tok_lp.unsqueeze(-1)).sum(-1) + 1      # 1 = model's top guess
        entropy = -(logp.exp() * logp).sum(-1)
        tokens = self.tok.convert_ids_to_tokens(target.tolist())
        return tokens, (-tok_lp).cpu().numpy(), ranks.cpu().numpy(), entropy.cpu().numpy()

    def features(self, text):
        _, nll, ranks, ent = self.token_scores(text)
        return {
            "log_ppl": float(nll.mean()),          # log perplexity
            "nll_std": float(nll.std()),           # how uneven the surprise is
            "nll_max": float(nll.max()),
            "top1_frac": float((ranks == 1).mean()),
            "top10_frac": float((ranks <= 10).mean()),
            "mean_entropy": float(ent.mean()),
        }


def build_feature_table(texts, scorer):
    rows = []
    for t in texts:
        f = stylometric_features(t)
        f.update(scorer.features(t))
        rows.append(f)
    return pd.DataFrame(rows)
