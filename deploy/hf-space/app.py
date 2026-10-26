"""Hugging Face Space entry point (Gradio SDK on ZeroGPU, the free hardware for new accounts).

Unpacks GlassQuill and serves its own server.py: the same FastAPI app (API + React site) as
`python server.py` on localhost. GPT-2 scoring, the part that uses a GPU on a local CUDA
machine, runs inside a @spaces.GPU call, which is how ZeroGPU lends a GPU to a request."""
try:
    import spaces       # first: ZeroGPU has to patch torch before anything touches CUDA
except ImportError:     # running somewhere without ZeroGPU: plain CPU/GPU like localhost
    spaces = None

import os
import sys
import tarfile
import traceback
from pathlib import Path

HERE = Path(__file__).parent
APP = Path("/tmp/glassquill")

if not (APP / "server.py").exists():
    with tarfile.open(HERE / "glassquill-app.tar.gz") as tar:
        tar.extractall(APP, filter="data")

# ZeroGPU sets PORT=7861 for its own runtime; visitors reach the Space on GRADIO_SERVER_PORT
os.environ["HOST"] = "0.0.0.0"
os.environ["PORT"] = os.environ.get("GRADIO_SERVER_PORT", "7860")
os.environ.setdefault("GLASSQUILL_TRUST_PROXY", "1")                 # behind the Space proxy
os.environ.setdefault("GLASSQUILL_FRAME_ANCESTORS", "https://huggingface.co")  # Space page iframe
if os.environ.get("SPACE_HOST"):                                     # e.g. user-glassquill.hf.space
    os.environ.setdefault("SITE_URL", f"https://{os.environ['SPACE_HOST']}")

os.chdir(APP)
sys.path.insert(0, str(APP))

import uvicorn                      # noqa: E402
from fastapi import HTTPException   # noqa: E402

import server                       # noqa: E402  loads the models (GPT-2 onto the GPU device)

score_tokens = server.det.scorer.token_scores      # GPT-2 surprise per token, as on localhost

if spaces is not None:
    score_tokens = spaces.GPU(duration=20)(score_tokens)


def score_tokens_or_busy(text):
    """Same result as before; if ZeroGPU can't lend a GPU right now (queue or daily quota),
    the site shows a clear message instead of a generic error."""
    try:
        return score_tokens(text)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(503, "GlassQuill's free GPU is busy right now. "
                                 "Please try again in a minute.") from e


server.det.scorer.token_scores = score_tokens_or_busy

if __name__ == "__main__":
    try:                            # only exists on ZeroGPU: register the GPU function
        from spaces.zero import startup
    except ImportError:
        startup = None
    if startup is not None:
        startup()
    uvicorn.run(server.app, host=os.environ["HOST"], port=int(os.environ["PORT"]))
