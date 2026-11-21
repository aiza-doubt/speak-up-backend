"""
Speak Up backend (Flask).

What it does:
  * (the website itself is hosted separately, on Vercel)
  * /api/chat        -> AI Conversation partner + AI Feedback (OpenAI chat model)
  * /api/transcribe  -> microphone recording -> text (OpenAI speech-to-text)
  * /api/tts         -> optional OpenAI voice (off by default; the browser voice is used)
  * /api/health      -> tells the website which features are switched on

Your OpenAI key lives ONLY here, in an environment variable. It is never sent to the browser.
Audio and text are processed in memory and are NOT saved or logged by this server.
"""
import os
import time
import threading
from collections import defaultdict, deque

from flask import Flask, jsonify, request, send_from_directory, Response
from werkzeug.middleware.proxy_fix import ProxyFix
from openai import OpenAI
import openai

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def env(name, default=""):
    return (os.environ.get(name) or default).strip()


# ---------- settings (all can be changed in the hosting dashboard, no code edits) ----------
API_KEY = env("OPENAI_API_KEY")
BASE_URL = env("OPENAI_BASE_URL")                       # leave empty for OpenAI itself
MODEL_DEFAULT = env("OPENAI_MODEL_DEFAULT", "gpt-6.1-terra")   # conversation + feedback reports
MODEL_QUICK = env("OPENAI_MODEL_QUICK", "gpt-6-luna")          # short live tips + word meanings
STT_MODEL = env("OPENAI_STT_MODEL", "gpt-transcribe")          # microphone -> text
STT_PROMPT = env(
    "OPENAI_STT_PROMPT",
    "Transcribe exactly what is said, including hesitations such as um, uh and er, and repeated words.",
)
TTS_MODEL = env("OPENAI_TTS_MODEL")                     # empty = use the free browser voice instead
TTS_VOICE = env("OPENAI_TTS_VOICE", "alloy")

# Websites allowed to call this backend, e.g. https://speak-up.vercel.app,https://www.yourdomain.com
ALLOWED_ORIGINS = {o.strip().rstrip("/") for o in env("ALLOWED_ORIGINS").split(",") if o.strip()}

DAILY_LIMIT = int(env("DAILY_REQUEST_LIMIT", "1500"))   # total AI calls per day for the whole site
LIMIT_CHAT_PER_MIN = int(env("LIMIT_CHAT_PER_MIN", "30"))     # per visitor
LIMIT_STT_PER_MIN = int(env("LIMIT_STT_PER_MIN", "15"))
LIMIT_TTS_PER_MIN = int(env("LIMIT_TTS_PER_MIN", "30"))

MAX_AUDIO_BYTES = 24 * 1024 * 1024                      # OpenAI accepts up to 25 MB per file
MAX_MESSAGES = 60
MAX_CHARS = 60000

app = Flask(__name__, static_folder=None)               # nothing is served unless listed below
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
app.config["MAX_CONTENT_LENGTH"] = MAX_AUDIO_BYTES + 512 * 1024

client = OpenAI(api_key=API_KEY, base_url=BASE_URL or None, timeout=90.0, max_retries=1) if API_KEY else None


# ---------- simple abuse protection (protects your wallet) ----------
_lock = threading.Lock()
_hits = defaultdict(deque)
_day = {"date": "", "count": 0}


def too_many(kind, per_min):
    """True if this visitor is going too fast, or the whole site hit today's cap."""
    now = time.time()
    ip = request.remote_addr or "unknown"
    with _lock:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if _day["date"] != today:
            _day["date"], _day["count"] = today, 0
        if _day["count"] >= DAILY_LIMIT:
            return True
        q = _hits[(kind, ip)]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= per_min:
            return True
        q.append(now)
        _day["count"] += 1
    return False


def fail(code, status):
    return jsonify({"error": code}), status


@app.after_request
def headers(resp):
    origin = (request.headers.get("Origin") or "").rstrip("/")
    if origin and origin in ALLOWED_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Max-Age"] = "600"
        resp.headers["Vary"] = "Origin"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "same-origin"
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/")
def home():
    return jsonify({"service": "speak-up-api", "ok": True})


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "ai": bool(client), "stt": bool(client), "tts": bool(client and TTS_MODEL)})


# ---------- AI conversation + feedback ----------
@app.post("/api/chat")
def chat():
    if not client:
        return fail("ai_not_configured", 503)
    if too_many("chat", LIMIT_CHAT_PER_MIN):
        return fail("rate_limited", 429)
    body = request.get_json(silent=True) or {}
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs or len(msgs) > MAX_MESSAGES:
        return fail("bad_request", 400)
    clean, total = [], 0
    for m in msgs:
        role, content = m.get("role"), m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            return fail("bad_request", 400)
        total += len(content)
        clean.append({"role": role, "content": content})
    if total > MAX_CHARS:
        return fail("bad_request", 400)
    model = MODEL_QUICK if body.get("tier") == "quick" else MODEL_DEFAULT
    try:
        out = client.chat.completions.create(model=model, messages=clean)
        return jsonify({"text": out.choices[0].message.content or ""})
    except openai.RateLimitError:
        return fail("rate_limited", 429)
    except openai.AuthenticationError:
        app.logger.error("OpenAI rejected the API key")
        return fail("ai_not_configured", 503)
    except Exception as e:  # never leak details to the browser
        app.logger.error("chat failed: %s", type(e).__name__)
        return fail("ai_unavailable", 502)


# ---------- microphone recording -> text ----------
@app.post("/api/transcribe")
def transcribe():
    if not client:
        return fail("ai_not_configured", 503)
    if too_many("stt", LIMIT_STT_PER_MIN):
        return fail("rate_limited", 429)
    f = request.files.get("audio")
    if not f:
        return fail("bad_request", 400)
    data = f.read()
    if len(data) < 1500:                                  # too short to contain speech
        return jsonify({"text": ""})
    name = os.path.basename(f.filename or "speech.webm")
    if not name.lower().endswith((".webm", ".mp4", ".m4a", ".ogg", ".wav", ".mp3")):
        name = "speech.webm"
    kwargs = {"model": STT_MODEL, "file": (name, data, f.mimetype or "audio/webm"), "language": "en"}
    try:
        try:
            out = client.audio.transcriptions.create(prompt=STT_PROMPT, **kwargs) if STT_PROMPT else client.audio.transcriptions.create(**kwargs)
        except openai.BadRequestError:
            out = client.audio.transcriptions.create(**kwargs)   # this model may not accept a prompt
        return jsonify({"text": (getattr(out, "text", "") or "").strip()})
    except openai.RateLimitError:
        return fail("rate_limited", 429)
    except openai.AuthenticationError:
        app.logger.error("OpenAI rejected the API key")
        return fail("ai_not_configured", 503)
    except Exception as e:
        app.logger.error("transcribe failed: %s", type(e).__name__)
        return fail("stt_unavailable", 502)


# ---------- optional OpenAI voice ----------
@app.post("/api/tts")
def tts():
    if not (client and TTS_MODEL):
        return fail("tts_off", 404)
    if too_many("tts", LIMIT_TTS_PER_MIN):
        return fail("rate_limited", 429)
    text = ((request.get_json(silent=True) or {}).get("text") or "").strip()[:1900]
    if not text:
        return fail("bad_request", 400)
    try:
        out = client.audio.speech.create(model=TTS_MODEL, voice=TTS_VOICE, input=text, response_format="mp3")
        return Response(out.content, mimetype="audio/mpeg")
    except Exception as e:
        app.logger.error("tts failed: %s", type(e).__name__)
        return fail("tts_unavailable", 502)


@app.errorhandler(413)
def too_big(_e):
    return fail("bad_request", 413)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(env("PORT", "8000")))
