"""GlassQuill demo: paste text, get a verdict and the reasons behind it."""
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

DEVICE = "cuda" if os.getenv("SPACES_ZERO_GPU") else None   # None = auto
det = Detector(os.getenv("GLASSQUILL_MODELS", "models"), device=DEVICE)


@gpu(duration=15)                     # seconds of GPU time asked for per call
def neural_features(text):
    return det.neural_features(text)


def run(text):
    if len(text.split()) < 50:
        raise gr.Error("Please paste at least 50 words.")
    plt.close("all")
    r = det.analyze(text, neural=neural_features(text))
    verdict = {"AI-generated": r["p_ai"], "Human-written": 1 - r["p_ai"]}
    reasons = (f"**Main model (XGBoost) says {r['p_ai']:.0%} AI.** "
               f"Word model (TF-IDF + LR) says {r['p_ai_lr']:.0%} AI.\n\n"
               f"Top reasons:\n{r['reasons']}")
    return verdict, reasons, r["shap_fig"], r["highlights"]


with gr.Blocks(title="GlassQuill") as demo:
    gr.Markdown("# GlassQuill\nExplainable detection of AI-generated text. "
                "This is a study project. Do not use it alone to accuse anyone.")
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(lines=12, label="Paste text (50 to 400 words works best)")
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column():
            label = gr.Label(label="Verdict", num_top_classes=2)
            reasons = gr.Markdown()
    plot = gr.Plot(label="Why: SHAP contribution of each feature")
    words = gr.HighlightedText(label="Words that pushed towards AI (red) or human (purple)")
    btn.click(run, inputs=text, outputs=[label, reasons, plot, words])

if __name__ == "__main__":
    demo.launch()
