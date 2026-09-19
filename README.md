# Operator — Browser Agent Control Room

## What's in here
- `index.html` — the frontend. Pure HTML/CSS/JS, no build step. Open it directly in any browser, or host it anywhere static (Vercel, Netlify, GitHub Pages).
- `server.py` — FastAPI backend. Runs the real agent, streams progress + screenshots over a WebSocket.
- `agent_core.py` — the agent logic itself, adapted from `cast_fixed_v4.py` (Browser Use + Gemini).
- `requirements.txt`, `render.yaml` — deploy config for Render.

## Why not Vercel for everything
`cast_fixed_v4.py` drives a real Chromium browser with Playwright. Vercel's serverless functions have no persistent process and can't launch a browser, so the backend has to run somewhere with a real, always-on process — Render, Railway, Fly.io, or your own machine. The **frontend** (`index.html`) is a static file and can go on Vercel, or anywhere else, or just be opened locally.

## Fastest way to test right now (local)
```bash
# 1. Backend
python -m venv venv && source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install --with-deps chromium
uvicorn server:app --reload --port 8000

# 2. Frontend
# Just open index.html in your browser (double-click it, or:)
python -m http.server 5500 -d .    # then visit http://localhost:5500/index.html
```
In the page: paste your Gemini API key, click **Verify** (it'll ask for your backend URL the first time — enter `http://localhost:8000`), type a task, hit **Run agent**.

Get a Gemini key at https://aistudio.google.com/apikey — `gemini-3.1-flash-lite` is Google's current low-latency, cost-efficient model built for high-volume agentic workflows like this one ($0.25/1M input tokens, $1.50/1M output tokens).

## Deploying the backend to Render
1. Push `server.py`, `agent_core.py`, `requirements.txt`, `render.yaml` to a GitHub repo.
2. In Render: **New → Blueprint**, point it at the repo. It reads `render.yaml` automatically.
3. **Use at least the `Standard` instance type.** Chromium is memory-hungry; the free/starter tier will likely OOM or time out mid-task. This is a real constraint of running a browser, not a config mistake.
4. Once deployed, copy your Render URL (e.g. `https://agent-control-server.onrender.com`).

## Deploying the frontend to Vercel
1. Push just `index.html` to a repo (or a folder containing it).
2. In Vercel: **New Project** → import the repo → deploy. No build command needed (static file).
3. Open the deployed site with `?server=https://your-backend.onrender.com` once, e.g.:
   `https://your-frontend.vercel.app/?server=https://agent-control-server.onrender.com`
   The frontend saves that URL in `localStorage`, so you only need the `?server=` param the first visit — or just paste the backend URL when the app prompts you.

## Security notes
- The Gemini API key never touches the server's disk. It lives in the browser's `localStorage` and is sent once per WebSocket connection, held in memory for that single run, then discarded.
- `ALLOWED_ORIGINS` in `render.yaml` defaults to `*` for easy testing — lock it down to your actual frontend's origin before sharing this publicly.
- `agent_core.py`'s `run_python` tool executes code the agent writes with no sandboxing beyond the process itself. Treat this deployment as single-tenant / trusted-use, not a public multi-tenant service, unless you add real sandboxing (e.g. a container-per-run).

## What changed vs. your original script
- Removed the self-bootstrapping venv (Render's build step installs pinned deps once instead).
- Removed the global `GOOGLE_API_KEY` env var — every run takes a key passed in from the browser, so multiple people can use one deployment without sharing credentials.
- Replaced the Rich terminal UI with structured events sent over a WebSocket (`state`, `log`, `action`, `screenshot`, `result`, `error`).
- Added live screenshot capture from the agent's browser session for the frontend's "live viewport" panel.
- Kept: the task prompt shape, the Google Sheets specialist auto-detection (static + live), retry/backoff policy, and all the Toolbox actions (fetch_url, read/write files, pandas table tools, run_python, system_info).
