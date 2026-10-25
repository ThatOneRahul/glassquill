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


def test_chunked_body_rejected():
    """Without Content-Length the size cap could not be checked before reading the body."""
    def body():
        yield b'{"text": "' + b"a " * 50_000
        yield b'"}'
    r = client.post("/api/analyze", content=body(), headers={"Content-Type": "application/json"})
    assert r.status_code == 411


def test_bad_content_length():
    r = client.post("/api/analyze", content=b"{}", headers={"Content-Type": "application/json", "Content-Length": "abc"})
    assert r.status_code in (400, 413)


def test_validation_error_does_not_echo_input():
    secret = "SECRETWORD " * 1200                      # > 12,000 characters
    r = client.post("/api/analyze", json={"text": secret})
    assert r.status_code == 422 and "SECRETWORD" not in r.text
    assert isinstance(r.json()["detail"], str)


@pytest.mark.parametrize("junk", [
    "😀 " * 60,
    " ".join(str(i) for i in range(60)),
    "यह एक परीक्षण वाक्य है " * 12,
    "!!! " * 60,
    "a " * 60,
])
def test_non_prose_rejected(junk):
    r = client.post("/api/analyze", json={"text": junk})
    assert r.status_code == 422 and "detail" in r.json()


def test_invisible_characters_are_ignored():
    """Zero-width spaces, RTL overrides and null bytes must not change the verdict."""
    clean = client.post("/api/analyze", json={"text": HUMAN}).json()
    dirty_text = HUMAN.replace(" ", " ​").replace("e", "e‮", 3) + "\x00"
    dirty = client.post("/api/analyze", json={"text": dirty_text}).json()
    assert abs(clean["p_final"] - dirty["p_final"]) < 1e-6


def test_busy_model_returns_503(monkeypatch):
    monkeypatch.setattr(server, "BUSY_WAIT", 0.1)
    server.lock.acquire()
    try:
        r = client.post("/api/analyze", json={"text": HUMAN})
    finally:
        server.lock.release()
    assert r.status_code == 503


def test_cache_headers():
    assert client.post("/api/analyze", json={"text": HUMAN}).headers["cache-control"] == "no-store"
    assert client.get("/api/meta").headers["cache-control"] == "no-store"


def test_rate_table_forgets_idle_ips():
    server.hits.clear()
    server.hits["203.0.113.9"].append(0.0)              # last seen long ago
    server.rate_limited("198.51.100.1")
    assert "203.0.113.9" not in server.hits
    server.hits.clear()


def test_rate_limit(monkeypatch):
    monkeypatch.setattr(server, "RATE", (2, 60))
    server.hits.clear()
    codes = [client.post("/api/analyze", json={"text": "x"}).status_code for _ in range(3)]
    assert codes == [422, 422, 429]
    server.hits.clear()


def test_faked_forwarded_for_cannot_dodge_rate_limit(monkeypatch):
    monkeypatch.setattr(server, "RATE", (2, 60))
    monkeypatch.setattr(server, "TRUST_PROXY", True)
    server.hits.clear()
    codes = [client.post("/api/analyze", json={"text": "x"},
                         headers={"X-Forwarded-For": f"10.0.0.{i}, 203.0.113.7"}).status_code
             for i in range(3)]
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


@pytest.mark.skipif(not server.DIST.exists(), reason="run `npm run build` in web/ first")
def test_site_url_filled_in_for_any_host(monkeypatch):
    monkeypatch.setattr(server, "TRUST_PROXY", True)
    proxied = {"X-Forwarded-Proto": "https"}
    assert "<loc>https://testserver/detect</loc>" in client.get("/sitemap.xml", headers=proxied).text
    assert 'href="https://testserver/"' in client.get("/", headers=proxied).text
    monkeypatch.setattr(server, "SITE_URL", "https://glassquill.example")
    assert "Sitemap: https://glassquill.example/sitemap.xml" in client.get("/robots.txt").text
    assert "%SITE_URL%" not in client.get("/no-such-page").text


@pytest.mark.skipif(not server.DIST.exists(), reason="run `npm run build` in web/ first")
def test_forwarded_proto_cannot_inject_markup(monkeypatch):
    monkeypatch.setattr(server, "TRUST_PROXY", True)
    page = client.get("/", headers={"X-Forwarded-Proto": '"><script>x()</script>'}).text
    assert "<script>x()" not in page and 'href="http://testserver/"' in page


@pytest.mark.skipif(not server.DIST.exists(), reason="needs web/dist")
def test_no_path_traversal():
    for p in ["/../server.py", "/..%2Fserver.py", "/%2e%2e/models/xgb.joblib"]:
        r = client.get(p)
        assert "import" not in r.text and r.headers["content-type"].startswith("text/html")
