# GlassQuill on Hugging Face Spaces. Same steps as the Dockerfile at the root of the
# GitHub repo, but the app comes from glassquill-app.tar.gz (made by deploy/make_space.sh).

# ---- 0. unpack the app ------------------------------------------------------------------
FROM busybox:stable AS app
ADD glassquill-app.tar.gz /src/

# ---- 1. build the website ---------------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY --from=app /src/web/package.json /src/web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY --from=app /src/web/ ./
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
    GLASSQUILL_TRUST_PROXY=1 \
    GLASSQUILL_FRAME_ANCESTORS=https://huggingface.co \
    SITE_URL=https://thatonerahul-glassquill.hf.space
WORKDIR /app

RUN pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu
COPY --from=app /src/requirements-server.txt .
RUN pip install -r requirements-server.txt

RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained('gpt2'); AutoModelForCausalLM.from_pretrained('gpt2')" \
 && NLTK_ALLOW_PROXIED_URLOPEN=1 python -c "import nltk; \
nltk.download('cmudict', download_dir='/app/.cache/nltk_data', quiet=True); \
nltk.data.find('corpora/cmudict')" \
 && python -c "import matplotlib.pyplot"
ENV HF_HUB_OFFLINE=1

COPY --from=app /src/server.py /src/examples.json ./
COPY --from=app /src/src/ src/
COPY --from=app /src/models/ models/
COPY --from=app /src/reports/ reports/
COPY --from=web /web/dist web/dist

RUN useradd -m -u 1000 user && chown -R user /app/.cache
USER user
EXPOSE 7860
CMD ["python", "server.py"]
