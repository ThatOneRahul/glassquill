"""Paraphrase attack (section 3.3): rewrite every sentence of an AI text with T5."""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.features import SENT_SPLIT

PARAPHRASER = "humarin/chatgpt_paraphraser_on_T5_base"


class Paraphraser:
    def __init__(self, model_name=PARAPHRASER, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, text):
        """All sentences of one text go through the model as one batch."""
        sents = [s for s in SENT_SPLIT.split(text) if s.strip()]
        enc = self.tok(["paraphrase: " + s for s in sents], return_tensors="pt",
                       truncation=True, max_length=128, padding=True).to(self.device)
        ids = self.model.generate(**enc, num_beams=4, max_length=128, no_repeat_ngram_size=3)
        return " ".join(self.tok.batch_decode(ids, skip_special_tokens=True))
