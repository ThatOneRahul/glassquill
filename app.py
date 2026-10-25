"""GlassQuill demo: paste text, get a verdict and the reasons behind it."""
import html
import json
import os

import gradio as gr
import matplotlib.pyplot as plt

try:                                  # present on Hugging Face Spaces
    import spaces
    gpu = spaces.GPU
except ImportError:                   # laptop or Colab: do nothing
    def gpu(*args, **kwargs):
        return args[0] if args and callable(args[0]) else (lambda f: f)

from src.detector import Detector

MODELS = os.getenv("GLASSQUILL_MODELS", "models")
DEVICE = "cuda" if os.getenv("SPACES_ZERO_GPU") else None   # None = auto
det = Detector(MODELS, device=DEVICE)
with open(os.path.join(MODELS, "feature_stats.json")) as f:
    TYPICAL = json.load(f)            # median of each feature for human and AI training texts
with open("examples.json", encoding="utf-8") as f:
    EXAMPLES = json.load(f)

MIN_WORDS, MAX_WORDS = 50, 400
UNSURE = (0.35, 0.65)                 # between these the app says "unclear" instead of guessing


@gpu(duration=15)                     # seconds of GPU time asked for per call
def neural_features(text):
    return det.neural_features(text)


def word_count(text):
    n = len(text.split())
    if n == 0:
        return "<span class='gq-count'>0 words</span>"
    hint = ("add a few more" if n < MIN_WORDS else
            "only the first ~256 tokens are scored" if n > MAX_WORDS else "good length")
    cls = "warn" if n < MIN_WORDS or n > MAX_WORDS else "ok"
    return f"<span class='gq-count {cls}'>{n} words · {hint}</span>"


def verdict_card(r):
    p = r["p_final"]
    if p >= UNSURE[1]:
        kind, title = "ai", "Likely AI-generated"
    elif p <= UNSURE[0]:
        kind, title = "human", "Likely human-written"
    else:
        kind, title = "unsure", "Unclear: could be either"
    agree = (r["p_ai"] >= 0.5) == (r["p_ai_lr"] >= 0.5)
    note = ("Both models agree." if agree else
            "The two models disagree. Treat this result with extra caution.")

    def bar(name, sub, v):
        return (f"<div class='gq-model'><div class='gq-model-head'><span>{name}"
                f"<small>{sub}</small></span><b>{v:.0%}</b></div>"
                f"<div class='gq-track'><div class='gq-fill' style='width:{v * 100:.1f}%'></div>"
                f"</div></div>")

    return (f"<div class='gq-card gq-{kind}'>"
            f"<span class='gq-badge'>{title}</span>"
            f"<div class='gq-score'>{p * 100:.0f}<span>%</span></div>"
            f"<div class='gq-caption'>chance the text is AI-generated</div>"
            f"<div class='gq-meter'><div style='width:{p * 100:.1f}%'></div></div>"
            f"{bar('Style model', 'XGBoost on 18 features', r['p_ai'])}"
            f"{bar('Word model', 'TF-IDF + logistic regression', r['p_ai_lr'])}"
            f"<div class='gq-note'>{note}</div></div>")


def fmt(feature, v):
    return f"{v:.1%}" if feature.endswith(("_frac", "_rate", "_ratio")) else f"{v:.2f}"


def reasons_html(contributions, k=5):
    items = []
    for c in contributions[:k]:
        t = TYPICAL.get(c["feature"], {})
        side = "ai" if c["shap"] > 0 else "human"
        typical = (f"humans ~{fmt(c['feature'], t['human'])}, AI ~{fmt(c['feature'], t['ai'])}"
                   if t else "")
        items.append(
            f"<li class='gq-reason gq-{side}'><span class='gq-dir'>"
            f"{'→ AI' if side == 'ai' else '→ human'}</span>"
            f"<span><b>{html.escape(c['name'])}</b> is {fmt(c['feature'], c['value'])}"
            f"<small>{typical}</small></span></li>")
    return ("<div class='gq-reasons'><h4>Top reasons from the style model</h4><ul>"
            + "".join(items) + "</ul></div>")


def feature_table(contributions):
    rows = []
    for c in contributions:                  # already sorted by size of the SHAP value
        t = TYPICAL.get(c["feature"], {})
        side = "ai" if c["shap"] > 0 else "human"
        rows.append(
            f"<tr><td>{html.escape(c['name'])}<small>{c['feature']}</small></td>"
            f"<td><b>{fmt(c['feature'], c['value'])}</b></td>"
            f"<td>{fmt(c['feature'], t['human']) if t else ''}</td>"
            f"<td>{fmt(c['feature'], t['ai']) if t else ''}</td>"
            f"<td class='gq-{side}'>{'AI' if side == 'ai' else 'human'} "
            f"<small>{c['shap']:+.2f}</small></td></tr>")
    return ("<table class='gq-table'><thead><tr><th>Feature</th><th>This text</th>"
            "<th>Typical human</th><th>Typical AI</th><th>Pushes towards (SHAP)</th></tr>"
            "</thead><tbody>" + "".join(rows) + "</tbody></table>")


def run(text):
    text = (text or "").strip()
    if len(text.split()) < MIN_WORDS:
        raise gr.Error(f"Please paste at least {MIN_WORDS} words. Short texts do not give "
                       "the models enough to go on.")
    plt.close("all")
    r = det.analyze(text, neural=neural_features(text))
    return (verdict_card(r), reasons_html(r["contributions"]), r["shap_fig"],
            r["highlights"], feature_table(r["contributions"]))


def clear():
    return "", word_count(""), EMPTY_CARD, "", None, None, ""


EMPTY_CARD = ("<div class='gq-card gq-empty'><div class='gq-empty-title'>No text analysed yet"
              "</div><p>Paste 50 to 400 words of English, or pick an example, then press "
              "<b>Analyse</b>.</p></div>")

ABOUT = """
### How GlassQuill decides
The verdict is the **average of two explainable models**:

| | Looks at | Explained by |
|---|---|---|
| **Style model** (XGBoost) | 18 numbers: how predictable the text is to GPT-2 (perplexity), sentence rhythm, vocabulary, punctuation, readability | SHAP: exactly how much each number moved the score |
| **Word model** (TF-IDF + logistic regression) | which words and word pairs appear | the weight of every word (the coloured text) |

Trained on 32,000 texts from HC3, HAP-E and M4GT: human writing and text from ChatGPT, GPT-4o, GPT-4, Llama-3, Mixtral, Gemma and others, across 17 domains.

| On texts it never saw | Accuracy | Humans wrongly flagged |
|---|---|---|
| Test set (same 17 domains) | 97.4% | 2.0% |
| MAGE (27 unseen generators) | 66.4% | 6.8% |

### Limits
- **Never use this score alone as proof.** It is a second opinion with reasons.
- Plain, predictable writing (common for non-native writers) can look AI-like.
- Paraphrasing tools, heavy editing and small open models can fool it.
- English only; best on 50-400 words; mixed human + AI texts are not handled.

Study project for Applications and Use Cases of ML (TE7483), SIT Pune.
"""

THEME = gr.themes.Soft(
    primary_hue="indigo", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"])

CSS = """
:root { --gq-ai: #e5484d; --gq-human: #6e56cf; --gq-unsure: #d97706; }
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
#gq-hero { padding: 8px 0 4px; }
#gq-hero h1 { font-size: 2.1rem; font-weight: 750; letter-spacing: -0.02em; margin: 0; }
#gq-hero h1 span { color: var(--color-accent); }
#gq-hero p { margin: 4px 0 0; opacity: .75; font-size: 1rem; }
.gq-pill { display: inline-block; margin-top: 10px; padding: 4px 12px; border-radius: 999px;
  font-size: .8rem; background: color-mix(in srgb, var(--gq-unsure) 14%, transparent);
  color: var(--gq-unsure); border: 1px solid color-mix(in srgb, var(--gq-unsure) 35%, transparent); }
.gq-count { font-size: .82rem; opacity: .8; }
.gq-count.warn { color: var(--gq-unsure); opacity: 1; }
.gq-count.ok { color: #16a34a; opacity: 1; }
.gq-card { border-radius: 16px; padding: 22px 22px 18px; border: 1px solid var(--border-color-primary);
  background: var(--block-background-fill); }
.gq-empty { text-align: center; padding: 48px 24px; opacity: .8; }
.gq-empty-title { font-size: 1.15rem; font-weight: 650; margin-bottom: 6px; }
.gq-card { --tone: var(--gq-unsure); }
.gq-card.gq-ai { --tone: var(--gq-ai); }
.gq-card.gq-human { --tone: var(--gq-human); }
.gq-badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 650;
  font-size: .85rem; color: var(--tone); background: color-mix(in srgb, var(--tone) 13%, transparent); }
.gq-score { font-size: 3.6rem; font-weight: 800; line-height: 1; margin-top: 12px; color: var(--tone);
  letter-spacing: -0.03em; }
.gq-score span { font-size: 1.6rem; font-weight: 700; margin-left: 2px; }
.gq-caption { opacity: .7; font-size: .9rem; margin-top: 2px; }
.gq-meter { height: 10px; border-radius: 99px; margin: 14px 0 18px; overflow: hidden;
  background: linear-gradient(90deg, color-mix(in srgb, var(--gq-human) 25%, transparent),
  color-mix(in srgb, var(--gq-unsure) 25%, transparent), color-mix(in srgb, var(--gq-ai) 25%, transparent)); }
.gq-meter div { height: 100%; background: var(--tone); border-radius: 99px; }
.gq-model { margin-top: 10px; }
.gq-model-head { display: flex; justify-content: space-between; font-size: .9rem; }
.gq-model-head small { opacity: .6; margin-left: 8px; }
.gq-track { height: 6px; border-radius: 99px; background: var(--border-color-primary); margin-top: 5px; }
.gq-fill { height: 100%; border-radius: 99px; background: var(--color-accent); }
.gq-note { margin-top: 16px; font-size: .85rem; opacity: .75; }
.gq-reasons h4 { margin: 4px 0 8px; font-size: .95rem; }
.gq-reasons ul { list-style: none; padding: 0; margin: 0; }
.gq-reason { display: flex; gap: 12px; align-items: baseline; padding: 9px 0;
  border-bottom: 1px solid var(--border-color-primary); }
.gq-reason:last-child { border-bottom: 0; }
.gq-reason small { display: block; opacity: .6; font-size: .8rem; }
.gq-dir { flex: 0 0 74px; font-weight: 700; font-size: .8rem; }
.gq-reason.gq-ai .gq-dir { color: var(--gq-ai); }
.gq-reason.gq-human .gq-dir { color: var(--gq-human); }
.gq-legend { font-size: .85rem; opacity: .8; }
.gq-legend b.ai { color: var(--gq-ai); } .gq-legend b.human { color: var(--gq-human); }
.gq-table { width: 100%; border-collapse: collapse; font-size: .9rem; }
.gq-table th { text-align: left; font-weight: 600; opacity: .7; font-size: .8rem;
  padding: 8px 10px; border-bottom: 1px solid var(--border-color-primary); }
.gq-table td { padding: 8px 10px; border-bottom: 1px solid var(--border-color-primary); }
.gq-table td small { display: block; opacity: .5; font-size: .75rem; font-family: var(--font-mono); }
.gq-table td.gq-ai { color: var(--gq-ai); font-weight: 600; }
.gq-table td.gq-human { color: var(--gq-human); font-weight: 600; }
footer { display: none !important; }
"""

with gr.Blocks(title="GlassQuill · explainable AI-text detector") as demo:
    gr.HTML("<div id='gq-hero'><h1>Glass<span>Quill</span></h1>"
            "<p>Is this text written by a person or an AI? GlassQuill gives a verdict "
            "and shows you why.</p>"
            "<span class='gq-pill'>Study project: a second opinion, never proof on its own"
            "</span></div>")
    with gr.Row(equal_height=False):
        with gr.Column(scale=6):
            text = gr.Textbox(lines=14, max_lines=22, show_label=False,
                              placeholder="Paste an English text of 50 to 400 words here…")
            count = gr.HTML(word_count(""))
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                btn = gr.Button("Analyse", variant="primary", scale=3)
            gr.Examples(examples=[[e["text"]] for e in EXAMPLES], inputs=text,
                        example_labels=[e["label"] for e in EXAMPLES],
                        label="Try an example (real texts the model never saw in training)")
        with gr.Column(scale=5):
            card = gr.HTML(EMPTY_CARD)
            reasons = gr.HTML()

    with gr.Tabs():
        with gr.Tab("Why: feature contributions"):
            gr.Markdown("Each bar shows how much one feature pushed the style model "
                        "**towards AI (red, right)** or **towards human (blue, left)**, "
                        "starting from the average score.")
            plot = gr.Plot(show_label=False)
        with gr.Tab("Word evidence"):
            gr.HTML("<p class='gq-legend'>Words the word model weighs: <b class='ai'>red</b> "
                    "pushes towards AI, <b class='human'>purple</b> towards human. Uncoloured "
                    "words carry no weight.</p>")
            words = gr.HighlightedText(show_label=False, combine_adjacent=False,
                                       show_legend=False)
        with gr.Tab("All 18 features"):
            table = gr.HTML("<p class='gq-legend'>Analyse a text to see all 18 features.</p>")
        with gr.Tab("About and limits"):
            gr.Markdown(ABOUT)

    text.change(word_count, text, count, show_progress="hidden")
    outputs = [card, reasons, plot, words, table]
    btn.click(run, text, outputs)
    text.submit(run, text, outputs)
    clear_btn.click(clear, None, [text, count] + outputs)


def launch(**kwargs):
    """Start the app with the GlassQuill look. In Colab: launch(share=True)."""
    return demo.launch(theme=THEME, css=CSS, **kwargs)


if __name__ == "__main__":
    launch()
