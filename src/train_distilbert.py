"""Model 4: fine-tune DistilBERT (most accurate, but a black box)."""
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

MODEL = "distilbert-base-uncased"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}


def train_distilbert(train, val, out_dir="models/distilbert", model_name=MODEL,
                     epochs=2, seed=42):
    tok = AutoTokenizer.from_pretrained(model_name)

    def to_ds(df):
        ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=256), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
        id2label={0: "Human", 1: "AI"}, label2id={"Human": 0, "AI": 1})
    args = TrainingArguments(
        output_dir=out_dir, eval_strategy="epoch", save_strategy="epoch",
        learning_rate=2e-5, per_device_train_batch_size=16, per_device_eval_batch_size=32,
        num_train_epochs=epochs, weight_decay=0.01, load_best_model_at_end=True,
        metric_for_best_model="f1", save_total_limit=1,
        fp16=torch.cuda.is_available(), report_to="none", seed=seed)
    trainer = Trainer(model=model, args=args, train_dataset=to_ds(train),
                      eval_dataset=to_ds(val), processing_class=tok,
                      data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    return trainer


@torch.no_grad()
def predict_proba(texts, model, tok, device="cpu", batch_size=32):
    """Returns an (n, 2) array of [P(human), P(AI)]. Also used by LIME."""
    model.to(device).eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(list(texts[i:i + batch_size]), truncation=True, max_length=256,
                  padding=True, return_tensors="pt").to(device)
        out.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
    return np.vstack(out)
