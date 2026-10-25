"""GlassQuill web server: JSON API for the detector + the built React site (web/dist).

    python server.py                      # http://127.0.0.1:8000  (API + site)
    cd web && npm run dev                 # frontend dev server on :5173, proxies /api here

Everything the site shows comes from the trained models and the CSVs in reports/.
"""
import csv
import json
import os
import re
import threading
import unicodedata
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.detector import Detector

ROOT = Path(__file__).parent
MODELS = ROOT / os.getenv("GLASSQUILL_MODELS", "models")
REPORTS = ROOT / "reports"
DIST = ROOT / "web" / "dist"
MIN_WORDS, MAX_CHARS, MAX_BODY = 50, 12_000, 64_000
RATE = (int(os.getenv("GLASSQUILL_RATE", "20")), 60)       # requests per window (seconds)
UNSURE = (0.35, 0.65)                                       # band where the verdict is "unclear"
PAGES = {"/", "/detect", "/results", "/method"}             # routes the React app owns

TRUST_PROXY = os.getenv("GLASSQUILL_TRUST_PROXY") == "1"     # behind nginx etc.: use X-Forwarded-For
BUSY_WAIT = 20                                              # seconds a request may wait for the model

det = Detector(str(MODELS))
lock = threading.Lock()                                     # one GPU, one model: no overlap
TYPICAL = json.loads((MODELS / "feature_stats.json").read_text())
EXAMPLES = json.loads((ROOT / "examples.json").read_text(encoding="utf-8"))

app = FastAPI(title="GlassQuill API", docs_url=None, redoc_url=None, openapi_url=None)

CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
       "script-src 'self'; font-src 'self'; connect-src 'self'; object-src 'none'; "
       "base-uri 'self'; frame-ancestors 'none'; form-action 'self'")
hits = defaultdict(deque)


def client_ip(request: Request) -> str:
    if TRUST_PROXY and (fwd := request.headers.get("x-forwarded-for")):
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "?"


def rate_limited(ip: str) -> bool:
    """Sliding window per IP. Idle IPs are dropped so the table cannot grow forever."""
    now = time.monotonic()
    for key in [k for k, q in hits.items() if not q or now - q[-1] > RATE[1]]:
        del hits[key]
    q = hits[ip]
    while q and now - q[0] > RATE[1]:
        q.popleft()
    if len(q) >= RATE[0]:
        return True
    q.append(now)
    return False


@app.middleware("http")
async def guard(request: Request, call_next):
    """Body size cap, per-IP rate limit on the API, and security and cache headers."""
    path = request.url.path
    if path.startswith("/api/") and request.method == "POST":
        length = request.headers.get("content-length")
        if length is None:          # chunked bodies would be read in full before any check
            return JSONResponse({"detail": "Content-Length is required."}, status_code=411)
        if not length.isdigit() or int(length) > MAX_BODY:
            return JSONResponse({"detail": "Request too large."}, status_code=413)
        if rate_limited(client_ip(request)):
            return JSONResponse({"detail": "Too many requests. Please wait a minute."},
                                status_code=429, headers={"Retry-After": str(RATE[1])})
    response = await call_next(request)
    response.headers.update({
        "Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY", "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Resource-Policy": "same-origin"})
    if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    if path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"          # results contain the pasted text
    elif path.startswith("/assets/"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"  # hashed names
    else:
        response.headers["Cache-Control"] = "no-cache"           # index.html, icons: revalidate
    return response


# added last so it wraps everything: 413/429 replies carry CORS headers too (dev server on :5173)
app.add_middleware(CORSMiddleware, allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
                   allow_origins=os.getenv("GLASSQUILL_CORS", "http://localhost:5173").split(","))


@app.exception_handler(RequestValidationError)
async def invalid_request(_request: Request, exc: RequestValidationError):
    """Short, readable messages, and never echo the submitted text back."""
    kinds = {e.get("type") for e in exc.errors()}
    if "string_too_long" in kinds:
        msg = f"That text is too long. Please keep it under {MAX_CHARS:,} characters."
    else:
        msg = 'Send JSON like {"text": "..."} with the text to analyse.'
    return JSONResponse({"detail": msg}, status_code=422)


class AnalyzeIn(BaseModel):
    text: str = Field(..., max_length=MAX_CHARS)


def read_csv(name):
    """Report table as a list of dicts with numbers parsed; [] if the file is missing."""
    path = REPORTS / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    """Examples, typical feature values and the real evaluation results for the site."""
    return {
        "examples": EXAMPLES,
        "typical": TYPICAL,
        "limits": {"min_words": MIN_WORDS, "max_chars": MAX_CHARS, "unsure": UNSURE},
        "results": {
            "in_domain": read_csv("results_in_domain.csv"),
            "by_corpus": read_csv("results_in_domain_by_corpus.csv"),
            "out_of_domain": read_csv("results_out_of_domain.csv"),
            "paraphrase": read_csv("results_paraphrase.csv"),
            "ood_by_generator": read_csv("ood_accuracy_by_generator.csv"),
            "shap_global": read_csv("shap_global.csv"),
            "sources": read_csv("eda_source_counts.csv"),
        },
    }


ENGLISH_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")      # "a", "I" and "don't" all count


def clean(text: str) -> str:
    """Drop control and invisible format characters (null bytes, zero-width spaces, RTL
    overrides): they carry no meaning, but they change the model's input and are a known
    trick for slipping AI text past detectors. Then normalise whitespace."""
    kept = "".join(ch for ch in text if ch.isspace() or unicodedata.category(ch) not in ("Cc", "Cf"))
    return " ".join(kept.split())


def check_prose(text: str) -> None:
    """The models were trained on English prose. Refuse input they cannot judge, instead of
    returning a confident but meaningless verdict (emoji, numbers, other scripts, spam)."""
    tokens = text.split()
    words = ENGLISH_WORD.findall(text)
    if len(tokens) < MIN_WORDS:
        raise HTTPException(422, f"Please paste at least {MIN_WORDS} words (you have {len(tokens)}).")
    # real prose with numbers or code still clears 40%; emoji, digits or other scripts score ~0%
    if len(words) < 0.4 * len(tokens):
        raise HTTPException(422, "GlassQuill only reads English prose. This text has too few "
                                 "English words to judge.")
    if len({w.lower() for w in words}) < 20:
        raise HTTPException(422, "This text repeats the same few words. Please paste real prose.")


@app.post("/api/analyze")
def analyze(body: AnalyzeIn):
    text = clean(body.text)
    check_prose(text)
    words = len(text.split())
    t0 = time.perf_counter()
    if not lock.acquire(timeout=BUSY_WAIT):             # don't let a queue tie up every worker
        raise HTTPException(503, "GlassQuill is busy right now. Please try again in a moment.")
    try:
        r = det.analyze(text, figure=False)
    finally:
        lock.release()
    p = r["p_final"]
    verdict = "ai" if p >= UNSURE[1] else "human" if p <= UNSURE[0] else "unsure"
    return {
        "verdict": verdict, "p_final": p, "p_style": r["p_ai"], "p_words": r["p_ai_lr"],
        "models_agree": (r["p_ai"] >= 0.5) == (r["p_ai_lr"] >= 0.5),
        "shap_base": r["shap_base"],
        "contributions": [{**c, "typical": TYPICAL.get(c["feature"])} for c in r["contributions"]],
        "highlights": [{"token": t, "score": s} for t, s in r["highlights"]],
        "words": words, "ms": round((time.perf_counter() - t0) * 1000),
    }


# ---- the built React site -------------------------------------------------------
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    def site(path: str):
        target = (DIST / path).resolve()
        if path and target.is_file() and DIST.resolve() in target.parents:   # no path escape
            return FileResponse(target)
        if path.startswith("api/"):
            raise HTTPException(404, "Unknown API route.")
        # unknown pages still get the app (it renders its 404 page) but with a real 404 status
        status = 200 if "/" + path.rstrip("/") in PAGES or path == "" else 404
        return FileResponse(DIST / "index.html", status_code=status)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
