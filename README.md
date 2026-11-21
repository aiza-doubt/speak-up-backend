# Speak Up - backend (deploy on Render)

Flask server that talks to OpenAI. Your OpenAI key is stored ONLY as an environment variable on Render.

Files in this repository: server.py, requirements.txt, render.yaml, env.example, README.md

Render settings (Web Service):
- Runtime: Python 3
- Build command: pip install -r requirements.txt
- Start command: gunicorn server:app --workers 1 --threads 8 --timeout 120
- Environment variables: OPENAI_API_KEY (required), ALLOWED_ORIGINS (your Vercel address), PYTHON_VERSION=3.12.3

Endpoints: GET /api/health, POST /api/chat, POST /api/transcribe, POST /api/tts (optional)
