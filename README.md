# GlassQuill

**Explainable detection of AI-generated text.** A case study project for Applications and Use Cases of Machine Learning (TE7483), AIML Department, Symbiosis Institute of Technology, Pune.

GlassQuill labels a piece of English text as human-written or AI-generated and shows why. It compares four models, tests them on other LLMs and on paraphrased text and explains every prediction with SHAP and LIME.

**Start here:** read the full project guide in [`docs/GlassQuill_Project_Guide.pdf`](docs/GlassQuill_Project_Guide.pdf). It covers the background, datasets, method, step-by-step build, deployment, rubric mapping and viva prep.

## Models

| Model | Input | Explained with |
|---|---|---|
| Perplexity threshold | GPT-2 log perplexity | the score itself |
| TF-IDF + logistic regression | word and word-pair weights | exact word contributions |
| XGBoost (main model) | 18 style and perplexity features | SHAP |
| DistilBERT | raw text | LIME and SHAP text |

## Data

- **HC3** (`Hello-SimpleAI/HC3`): human and ChatGPT answers to the same questions. Main train, validation and test data.
- **MAGE** (`yaful/MAGE`, test split): texts from many LLMs and domains. Out-of-domain test.

## Repo layout

```
app.py               Gradio app, also the Hugging Face Space entry point
streamlit_app.py     optional Streamlit front end
notebook_cells.py    every Colab cell in order (copy them into a notebook)
requirements.txt
docs/                the project guide (PDF)
src/                 data, features, models, DistilBERT, explanations, detector
models/              saved models go here after training
reports/figures/     plots for the report
```

## Quick start (Google Colab)

1. Runtime > Change runtime type > T4 GPU.
2. Add a GitHub token as the Colab secret `GITHUB_TOKEN` (the repo is private, see section 7.3 of the guide).
3. Run:

```python
from google.colab import userdata
token = userdata.get("GITHUB_TOKEN")
!git clone https://{token}@github.com/ThatOneRahul/glassquill.git
%cd glassquill
!pip install -q -r requirements.txt
```

4. Copy the cells from `notebook_cells.py` into the notebook and run them in order.

## Status

- [x] Project guide
- [x] Starter code (tested end to end on dummy data, October 2026)
- [ ] Run on HC3 and record results
- [ ] Robustness tests and explanation analysis
- [ ] Deploy the app
- [ ] Report and slides

*Study project. A detector score should never be the only evidence that someone used AI.*
