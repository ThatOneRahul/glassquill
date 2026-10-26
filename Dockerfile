# GlassQuill in one container: the built React site + the FastAPI server.
#
#   docker build -t glassquill .
#   docker run -p 7860:7860 glassquill        # http://localhost:7860
#
# Needs about 1.5 GB of RAM (GPT-2 stays in memory). The Hugging Face Space is built from
# deploy/hf-space, which runs these same steps on an archive of the app (deploy/make_space.sh).

# ---- 1. build the website (same as `cd web && npm ci && npm run build`) -----------------
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---- 2. the server ----------------------------------------------------------------------
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    NLTK_DATA=/app/.cache/nltk_data \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    HOST=0.0.0.0 \
    PORT=7860 \
    GLASSQUILL_TRUST_PROXY=1
WORKDIR /app

# CPU build of torch: the hosting has no GPU, and the CUDA build is ~3 GB larger
RUN pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu
COPY requirements-server.txt .
RUN pip install -r requirements-server.txt

# Bake GPT-2, the textstat word list and matplotlib's font cache into the image, so the
# server starts without downloading anything
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained('gpt2'); AutoModelForCausalLM.from_pretrained('gpt2')" \
 && NLTK_ALLOW_PROXIED_URLOPEN=1 python -c "import nltk; \
nltk.download('cmudict', download_dir='/app/.cache/nltk_data', quiet=True); \
nltk.data.find('corpora/cmudict')" \
 && python -c "import matplotlib.pyplot"
ENV HF_HUB_OFFLINE=1

COPY server.py examples.json ./
COPY src/ src/
COPY models/ models/
COPY reports/*.csv reports/
COPY --from=web /web/dist web/dist

# Hugging Face runs containers as user 1000; caches must stay writable for it
RUN useradd -m -u 1000 user && chown -R user /app/.cache
USER user
EXPOSE 7860
CMD ["python", "server.py"]
