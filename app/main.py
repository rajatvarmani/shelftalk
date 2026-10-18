"""ShelfTalk web server (Flask). Run locally:  python -m app.main"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from . import agent
from .store import get_store

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("shelftalk")

STATIC = Path(__file__).resolve().parent.parent / "static"
MAX_IMAGES = 4
GEMINI_AUDIO = {"audio/wav", "audio/mp3", "audio/mpeg", "audio/aiff", "audio/aac",
                "audio/ogg", "audio/flac"}

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


def to_wav(data: bytes, mime: str) -> tuple[bytes, str]:
    """Browsers record webm/mp4; normalise to 16 kHz mono WAV for Gemini."""
    base = mime.split(";")[0].strip().lower()
    if base in GEMINI_AUDIO or not shutil.which("ffmpeg"):
        return data, base or "audio/webm"
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in"), Path(d, "out.wav")
        src.write_bytes(data)
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src),
                            "-ac", "1", "-ar", "16000", str(dst)], capture_output=True, timeout=30)
        if r.returncode != 0 or not dst.exists():
            log.warning("ffmpeg failed: %s", r.stderr[:300])
            return data, base
        return dst.read_bytes(), "audio/wav"


def snapshot() -> dict:
    store = get_store()
    state = store.state()
    return {
        "shop": state["shop"],
        "inventory": store.inventory(),
        "drafts": state["drafts"][:10],
        "report": store.report(7),
        "today": store.report(1),
        "recent": list(reversed(state["ledger"][-25:])),
    }


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:p>")
def static_files(p):
    return send_from_directory(STATIC, p)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/state")
def api_state():
    return jsonify(snapshot())


@app.post("/api/turn")
def api_turn():
    text = request.form.get("text", "")
    try:
        history = json.loads(request.form.get("history", "[]"))
    except json.JSONDecodeError:
        history = []
    images = []
    for f in request.files.getlist("images")[:MAX_IMAGES]:
        data = f.read()
        if data:
            images.append((data, f.mimetype or "image/jpeg"))
    audio = None
    a = request.files.get("audio")
    if a:
        data = a.read()
        if data:
            audio = to_wav(data, a.mimetype or "audio/webm")
    if not (text.strip() or images or audio):
        return jsonify({"error": "Send a photo, a voice note or a message."}), 400
    try:
        result = agent.run_turn(text=text, images=images, audio=audio, history=history)
    except Exception as e:
        log.exception("agent failed")
        return jsonify({"error": friendly_error(e)}), 502
    result["state"] = snapshot()
    return jsonify(result)


@app.post("/api/brief")
def api_brief():
    try:
        result = agent.run_turn(text=agent.BRIEF_PROMPT)
    except Exception as e:
        log.exception("brief failed")
        return jsonify({"error": friendly_error(e)}), 502
    result["state"] = snapshot()
    return jsonify(result)


@app.post("/api/undo")
def api_undo():
    rid = (request.get_json(silent=True) or {}).get("request_id", "")
    n = get_store().undo(rid) if rid else 0
    return jsonify({"undone": n, "state": snapshot()})


@app.post("/api/reset")
def api_reset():
    get_store().reset()
    return jsonify({"state": snapshot()})


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "api key" in low or "api_key" in low or "credentials" in low or "default credentials" in low:
        return "Gemini is not configured. Set GEMINI_API_KEY, or GOOGLE_GENAI_USE_VERTEXAI=true on Cloud Run."
    if "429" in low or "quota" in low or "resource_exhausted" in low:
        return "Gemini is busy (rate limit). Wait a few seconds and send again."
    return f"Gemini call failed: {msg[:200]}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
