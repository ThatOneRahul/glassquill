"""API and site-serving checks for server.py (loads the real models, ~20 s).

    .venv/Scripts/python.exe -m pytest tests -q
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GLASSQUILL_RATE"] = "1000"          # rate limit tested separately below

from fastapi.testclient import TestClient      # noqa: E402

import server                                   # noqa: E402

client = TestClient(server.app)
EXAMPLES = json.loads((ROOT / "examples.json").read_text(encoding="utf-8"))
HUMAN = ("My grandfather never trusted banks. When he died we found nearly eleven thousand "
         "rupees rolled up in old film canisters behind the water heater, plus a note in his "
         "awful handwriting that just said for the roof. The roof had been replaced twice since "
         "he wrote it. My mother laughed for about a minute and then cried for an hour. We used "
         "the money for his memorial lunch, which he would have hated.")


def test_health_and_headers():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    for h in ["Content-Security-Policy", "X-Content-Type-Options", "X-Frame-Options", "Referrer-Policy"]:
        assert h in r.headers


def test_meta_has_real_results():
    m = client.get("/api/meta").json()
    assert len(m["examples"]) >= 3
    assert len(m["typical"]) == 18
    models = {r["model"] for r in m["results"]["in_domain"]}
    assert "XGBoost + LR (app)" in models and "DistilBERT" in models


@pytest.mark.parametrize("text", ["", "too short to analyse"])
def test_short_text_rejected(text):
    r = client.post("/api/analyze", json={"text": text})
    assert r.status_code == 422


def test_too_long_rejected():
    r = client.post("/api/analyze", json={"text": "word " * 3000})   # 15,000 chars > 12,000
    assert r.status_code == 422


def test_body_size_capped():
    r = client.post("/api/analyze", content=b'{"text": "' + b"a" * 70_000 + b'"}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_wrong_type_rejected():
    assert client.post("/api/analyze", json={"text": 123}).status_code == 422
    assert client.post("/api/analyze", json={}).status_code == 422


def test_analyze_ai_example():
    chatgpt = next(e["text"] for e in EXAMPLES if e["label"].startswith("ChatGPT"))
    r = client.post("/api/analyze", json={"text": chatgpt})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "ai" and body["p_final"] > 0.65
    assert len(body["contributions"]) == 18
    assert all(c["typical"] for c in body["contributions"])
    assert body["highlights"] and {"token", "score"} <= body["highlights"][0].keys()


def test_analyze_human_text():
    body = client.post("/api/analyze", json={"text": HUMAN}).json()
    assert body["verdict"] == "human", body["p_final"]


def test_markup_is_returned_as_plain_text():
    """Scripts in the input come back as plain tokens; the frontend renders them as text."""
    text = HUMAN + " <script>alert(1)</script>"
    body = client.post("/api/analyze", json={"text": text}).json()
    assert any("<script>" in h["token"] for h in body["highlights"])


def test_rate_limit(monkeypatch):
    monkeypatch.setattr(server, "RATE", (2, 60))
    server.hits.clear()
    codes = [client.post("/api/analyze", json={"text": "x"}).status_code for _ in range(3)]
    assert codes == [422, 422, 429]
    server.hits.clear()


@pytest.mark.skipif(not server.DIST.exists(), reason="run `npm run build` in web/ first")
def test_site_routes_and_404():
    for page in ["/", "/detect", "/results", "/method"]:
        r = client.get(page)
        assert r.status_code == 200 and '<div id="root">' in r.text, page
    assert client.get("/no-such-page").status_code == 404
    assert client.get("/api/nope").status_code == 404
    for f in ["/favicon.ico", "/favicon.svg", "/robots.txt", "/sitemap.xml", "/site.webmanifest"]:
        assert client.get(f).status_code == 200, f
    assert "Sitemap:" in client.get("/robots.txt").text


@pytest.mark.skipif(not server.DIST.exists(), reason="needs web/dist")
def test_no_path_traversal():
    for p in ["/../server.py", "/..%2Fserver.py", "/%2e%2e/models/xgb.joblib"]:
        r = client.get(p)
        assert "import" not in r.text and r.headers["content-type"].startswith("text/html")
