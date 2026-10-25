# Viva preparation

The viva is individual: **each member must be able to answer every question here**, not only the ones about the part they built. The answers are short on purpose, so say them in your own words.

## Problem and use case
**Q. What problem does GlassQuill solve?**
AI-text detectors give a score without a reason, and a wrong flag can hurt a student. We classify text as human or AI and show *why*, so a teacher can check the reasoning.

**Q. Who uses it?**
Teachers (a second opinion), students (a way to question a flag), platforms (screening with an audit trail) and researchers (a transparent baseline).

**Q. Why focus on the false positive rate?**
A false positive accuses a human of cheating. That is the harmful error, so we report FPR next to F1 everywhere. Our app verdict has a 2% FPR on the test set.

**Q. Why is accuracy not enough?**
Two models with the same accuracy can make very different errors. We also report precision, recall, F1, ROC-AUC, FPR and confusion matrices.

## Data
**Q. Which datasets, and why three?**
HC3 has 2022 ChatGPT answers next to human answers to the same questions. HAP-E and M4GT add 2024 models (GPT-4o, Llama-3, Mixtral, Gemma) and 12 more genres. With HC3 alone, the model called a modern ChatGPT paragraph 100% human.

**Q. Target variable? Missing values?**
`label`: 0 = human, 1 = AI. We drop empty and duplicate texts, so there are 0 missing values after cleaning. The features cannot be missing because rates divide by max(words, 1).

**Q. Why did you split by question or document?**
HC3 has human and AI answers to the same question. If that question is in both train and test, the model can learn the topic instead of the style, and the score is too high. We use GroupShuffleSplit and GroupKFold.

**Q. What shortcuts did you find?**
HC3 human answers were tokenized (`do n't`, `( x )`, `URL_0`) and ChatGPT answers were not. 34% of human texts had `do n't` against 0.02% of AI texts. Also: literal `\n` in AI texts, markdown in M4GT AI texts, and a source imbalance (open_qa: 20 human vs 764 AI). We removed all of them. Accuracy fell from 99.3% to a believable level, which shows the shortcuts were real.

**Q. Why balance per domain?**
Otherwise the topic predicts the label. Each of our 17 domains has equal human and AI counts.

## Features and models
**Q. What is perplexity? Why is it lower for AI text?**
PPL = exp of the mean negative log-probability of each token under a language model (GPT-2). LLMs choose likely tokens, so another LM finds their text easy to predict. It is our strongest feature (effect size -1.4 SD).

**Q. Why GPT-2 and not a newer model?**
We tested Qwen2.5-0.5B and Binoculars. They added only +0.01 to +0.04 AUC and were 25× slower on our GPU. GPT-2 is fast, free and well studied.

**Q. What is burstiness?**
Std of sentence length / mean sentence length. Humans mix short and long sentences. AI is more even: 0.38 vs 0.53.

**Q. Why these four models?**
They go from simple and transparent to powerful and opaque: a perplexity threshold (one signal, zero-shot), TF-IDF + LR (glass box on words), XGBoost (glass box on 18 meaningful features) and DistilBERT (black box). This lets us measure what explainability costs.

**Q. What are monotone constraints and why use them?**
They force a feature to push in one direction only, e.g. lower burstiness can only increase the AI score. We applied them only where theory and data agree (10 features). Without them the HC3-only model learned "very even sentences mean human", which is backwards.

**Q. Why is the app verdict the average of XGBoost and LR?**
They look at different things (style vs words) and make different mistakes. We picked the rule on the **validation** set: 97.2% accuracy and 2.1% FPR, vs 92.9% for XGBoost alone. It also beats DistilBERT on the test set (97.4% vs 96.1%, FPR 2.0% vs 7.1%) and stays explainable.

**Q. How did you tune?**
5-fold GroupKFold on train, scored on F1. LR: C ∈ {0.1, 1, 10}, best 10. XGBoost: depth ∈ {3, 4, 6}, learning rate ∈ {0.05, 0.1}, 400 trees. Perplexity threshold: Youden's J on validation. DistilBERT: 2 epochs, best epoch by validation F1.

## Explainability
**Q. How are SHAP values calculated?**
Shapley values from game theory: the average change in the prediction when a feature is added, over all feature orders. TreeSHAP computes them exactly for trees. Base value + sum of SHAP = model output in log-odds.

**Q. SHAP vs LIME?**
SHAP has theory behind it and adds up exactly. LIME perturbs the text and fits a local linear model, so it is approximate and random. On our LR model, where the true word weights are known, LIME recovers only part of the top 10, so we trust SHAP more.

**Q. What does XGBoost's top feature tell you?**
`log_ppl`: low perplexity pushes strongly towards AI (see `shap_beeswarm.png`). Next come `top10_frac` and `nll_std`.

## Results and limits
**Q. Main result?**
On the test set the app verdict gets 97.4% accuracy, 0.974 F1, 0.995 ROC-AUC and 2.0% FPR. DistilBERT gets 96.1% and 7.1% FPR.

**Q. Why does performance drop on MAGE?**
MAGE has 27 generators and 14 domains we never trained on. We still catch GPT-4 (90%) and GPT-3.5 (80%). Small base models (OPT, BLOOM, GPT-J) sample "surprising" words, so their perplexity looks human, and nearly all detectors miss them.

**Q. How can someone fool the detector?**
Paraphrasing (we measured the recall drop), mixing human and AI sentences, asking the LLM to write casually, or adding typos.

**Q. Is it fair to non-native writers?**
Not proven. Simple wording has low perplexity, so it risks being flagged. That is why the app shows reasons and a disclaimer and never gives a bare verdict.

**Q. What would you do with more time?**
Add a modern reader LM or Binoculars as an extra feature, set the threshold for a fixed 1% FPR, do sentence-level detection for mixed texts, cover Indian languages and test on RAID.
