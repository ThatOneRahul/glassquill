---
title: GlassQuill
emoji: 🪶
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Explainable detection of AI-generated text
---

# GlassQuill

Paste an English text and GlassQuill tells you whether it reads as human-written or AI-generated, and shows the evidence behind the verdict: which writing signals pushed it each way, and which words.

- **Verdict with an "unclear" band:** AI, human, or unclear when the evidence is weak.
- **Reasons, not just a score:** SHAP shows how much each of 18 writing signals moved the verdict.
- **Word evidence:** colours the words that pushed towards human or AI.
- **Private:** pasted text is never stored or logged.

On 4,829 held-out texts it reaches 97.4% accuracy at a 2.0% false positive rate. A detector score is a second opinion, never proof on its own.

Made by Ranjan Verma, Rahul Narkhede, Priyansh Johri and Priyansh Pandey as an Applications and Use Cases of Machine Learning case study.

The app's files are in `glassquill-app.tar.gz`. `app.py` unpacks them and runs GlassQuill's FastAPI server (GPT-2, XGBoost and TF-IDF + logistic regression) with its React site.
