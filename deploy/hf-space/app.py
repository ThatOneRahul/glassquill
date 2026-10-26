"""Hugging Face Space entry point. Unpacks GlassQuill and runs its own server.py, the same
FastAPI app (API + React site) as `python server.py` on localhost, on the Space's port.
Gradio is only the Space's runtime here; the site is GlassQuill's own React app."""
import os
import runpy
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).parent
APP = Path("/tmp/glassquill")

if not (APP / "server.py").exists():
    with tarfile.open(HERE / "glassquill-app.tar.gz") as tar:
        tar.extractall(APP, filter="data")

os.environ.setdefault("HOST", "0.0.0.0")
os.environ.setdefault("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860"))
os.environ.setdefault("GLASSQUILL_TRUST_PROXY", "1")                 # behind the Space proxy
os.environ.setdefault("GLASSQUILL_FRAME_ANCESTORS", "https://huggingface.co")  # Space page iframe
if os.environ.get("SPACE_HOST"):                                     # e.g. user-glassquill.hf.space
    os.environ.setdefault("SITE_URL", f"https://{os.environ['SPACE_HOST']}")

os.chdir(APP)
sys.path.insert(0, str(APP))
runpy.run_path(str(APP / "server.py"), run_name="__main__")
