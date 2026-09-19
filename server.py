#!/usr/bin/env python3
"""
Agent Control Server
=====================

FastAPI backend for the browser automation frontend. Wraps the
Browser Use + Gemini agent (see agent_core.py, adapted from
cast_fixed_v4.py) behind:

  - POST /api/session/verify   -> validate a Gemini API key
  - WS   /ws/run                -> start a task, stream live events
  - GET  /api/history            -> recent task history
  - GET  /api/health              -> liveness probe

Design decisions
-----------------
- The Gemini API key is NEVER read from a server-side .env or
  stored on disk. It arrives per-session from the browser (the
  frontend keeps it in localStorage) and is held only in memory
  for the lifetime of one WebSocket connection / one task run.
- Real browser automation (Playwright/Chromium) needs a real,
  persistent process — this is designed to run on Render (or any
  VM/container host), not on serverless/edge platforms.
- Each task runs in its own isolated Agent + browser session so
  concurrent users never share state.
- Screenshots are pulled from Browser Use's live browser session
  on a timer and pushed down the same WebSocket as small JPEG
  data URLs, so the frontend can render a live "viewport".

Run locally:
    pip install -r requirements.txt
    playwright install --with-deps chromium
    uvicorn server:app --host 0.0.0.0 --port 8000

Deploy:
    See render.yaml / README.md for the Render.com deployment.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("agent-server")

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path(os.getenv("AGENT_DATA_DIR", "./agent_data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_FILE = DATA_DIR / "history.jsonl"

MODEL_NAME = os.getenv("AGENT_MODEL", "gemini-3.1-flash-lite")
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "60"))
MAX_FAILURES = int(os.getenv("AGENT_MAX_FAILURES", "5"))
STEP_TIMEOUT = int(os.getenv("AGENT_STEP_TIMEOUT", "180"))
HEADLESS = os.getenv("HEADLESS", "true").lower() in {"1", "true", "yes", "on"}
SCREENSHOT_INTERVAL = float(os.getenv("SCREENSHOT_INTERVAL", "1.2"))

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if o.strip()
]

# ============================================================
# APP
# ============================================================

app = FastAPI(title="Agent Control Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HISTORY PERSISTENCE
# ============================================================

@dataclass
class TaskRecord:
    id: str
    task: str
    started_at: str
    finished_at: Optional[str] = None
    status: str = "running"
    steps: int = 0
    error: Optional[str] = None
    result: Optional[str] = None


def save_history(record: TaskRecord) -> None:
    try:
        with HISTORY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("Failed writing history")


def read_history(limit: int = 30) -> list[dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    lines = HISTORY_FILE.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    # Keep only the latest record per id (later lines overwrite status).
    by_id: dict[str, dict[str, Any]] = {}
    for line in lines:
        try:
            rec = json.loads(line)
            by_id[rec["id"]] = rec
        except Exception:
            continue
    out = list(by_id.values())
    out.sort(key=lambda r: r.get("started_at", ""), reverse=True)
    return out[:limit]


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class VerifyKeyRequest(BaseModel):
    api_key: str = Field(..., min_length=10)


class VerifyKeyResponse(BaseModel):
    ok: bool
    model: str
    message: str


# ============================================================
# GEMINI KEY VERIFICATION (no agent, just a cheap round trip)
# ============================================================

@app.post("/api/session/verify", response_model=VerifyKeyResponse)
async def verify_key(payload: VerifyKeyRequest) -> VerifyKeyResponse:
    api_key = payload.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is empty.")

    try:
        from google import genai
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="google-genai is not installed on the server.",
        ) from exc

    try:
        client = genai.Client(api_key=api_key)
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL_NAME,
            contents="Reply with exactly READY",
        )
        text = getattr(response, "text", None)
        if not text:
            return VerifyKeyResponse(
                ok=False, model=MODEL_NAME, message="Gemini returned an empty response."
            )
        return VerifyKeyResponse(
            ok=True, model=MODEL_NAME, message=f"Connected to {MODEL_NAME}."
        )
    except Exception as exc:  # noqa: BLE001
        return VerifyKeyResponse(
            ok=False,
            model=MODEL_NAME,
            message=f"{type(exc).__name__}: {exc}",
        )


@app.get("/api/history")
async def get_history(limit: int = 30) -> JSONResponse:
    return JSONResponse(read_history(limit=limit))


@app.get("/")
async def root() -> dict[str, Any]:
    """Friendly landing response — this backend has no UI of its own.
    Open index.html separately and point it at this server's URL."""
    return {
        "service": "Agent Control Server",
        "status": "running",
        "message": "This is the API backend, not the app itself. Open index.html in your browser and point it at this URL.",
        "endpoints": {
            "health": "/api/health",
            "verify_key": "POST /api/session/verify",
            "history": "/api/history",
            "run_agent": "WS /ws/run",
        },
    }


@app.get("/api/health")
async def health() -> dict[str, Any]:
    try:
        import browser_use  # noqa: F401
        bu_ok = True
    except Exception:
        bu_ok = False
    return {
        "status": "ok",
        "browser_use_available": bu_ok,
        "headless": HEADLESS,
        "model": MODEL_NAME,
        "time": datetime.now().isoformat(),
    }


# ============================================================
# LIVE AGENT RUN — WebSocket
# ============================================================
#
# Protocol (server -> client), one JSON object per message:
#   {"type": "state",  "state": "planning"|"executing"|"verifying"
#                                |"completed"|"failed"|"cancelled",
#    "elapsed": 12.3}
#   {"type": "log",    "message": "..."}
#   {"type": "screenshot", "data": "data:image/jpeg;base64,...", "url": "..."}
#   {"type": "action", "name": "click_element", "detail": "..."}
#   {"type": "result", "text": "..."}
#   {"type": "error",  "message": "..."}
#
# Protocol (client -> server), sent once immediately after connect:
#   {"api_key": "...", "task": "...", "max_steps": 60, "headless": true}
#
# Client may also send {"type": "cancel"} at any time.

class RunSession:
    """One isolated agent run bound to one WebSocket connection."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.cancelled = False
        self.started = time.monotonic()
        self._send_lock = asyncio.Lock()

    async def send(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            try:
                await self.ws.send_text(json.dumps(payload))
            except Exception:
                # Connection likely closed; caller loops will notice.
                pass

    async def log(self, message: str) -> None:
        logger.info(message)
        await self.send({"type": "log", "message": message, "t": time.monotonic() - self.started})

    async def state(self, state: str) -> None:
        await self.send({"type": "state", "state": state, "elapsed": round(time.monotonic() - self.started, 1)})

    async def action(self, name: str, detail: str = "") -> None:
        await self.send({"type": "action", "name": name, "detail": detail})

    async def screenshot(self, data_url: str, url: str = "") -> None:
        await self.send({"type": "screenshot", "data": data_url, "url": url})


@app.websocket("/ws/run")
async def ws_run(ws: WebSocket) -> None:
    await ws.accept()
    session = RunSession(ws)
    record: Optional[TaskRecord] = None
    agent = None
    screenshot_task: Optional[asyncio.Task] = None

    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=30)
        init = json.loads(raw)
    except (asyncio.TimeoutError, json.JSONDecodeError, WebSocketDisconnect):
        await session.send({"type": "error", "message": "No valid init payload received."})
        await ws.close()
        return

    api_key = (init.get("api_key") or "").strip()
    task_text = (init.get("task") or "").strip()
    max_steps = int(init.get("max_steps") or MAX_STEPS)
    headless = bool(init.get("headless", HEADLESS))
    model_name = (init.get("model") or MODEL_NAME).strip()

    if not api_key:
        await session.send({"type": "error", "message": "Missing Gemini API key."})
        await ws.close()
        return
    if not task_text:
        await session.send({"type": "error", "message": "Missing task description."})
        await ws.close()
        return

    record = TaskRecord(
        id=str(uuid.uuid4())[:8],
        task=task_text,
        started_at=datetime.now().isoformat(),
    )
    save_history(record)

    try:
        from agent_core import AgentRunner

        await session.state("planning")
        await session.log("Validating Gemini connection and preparing browser session...")

        runner = AgentRunner(
            api_key=api_key,
            model=model_name,
            max_steps=max_steps,
            max_failures=MAX_FAILURES,
            step_timeout=STEP_TIMEOUT,
            headless=headless,
        )

        # Background loop: periodically grab a screenshot from the
        # live browser session while the agent runs.
        async def screenshot_loop():
            last_url = ""
            while not session.cancelled:
                await asyncio.sleep(SCREENSHOT_INTERVAL)
                try:
                    shot = await runner.capture_screenshot()
                    if shot:
                        b64, url = shot
                        last_url = url or last_url
                        await session.screenshot(f"data:image/jpeg;base64,{b64}", last_url)
                except Exception:
                    continue

        screenshot_task = asyncio.create_task(screenshot_loop())

        async def on_step(event: dict[str, Any]) -> None:
            record.steps += 1
            kind = event.get("kind", "step")
            if kind == "action":
                await session.action(event.get("name", "action"), event.get("detail", ""))
            else:
                await session.log(event.get("message", str(event)))

        await session.state("executing")
        await session.log(f"Agent live — model {model_name}, max {max_steps} steps.")

        result_text = await runner.run(task_text, on_step=on_step)

        session.cancelled = True  # stop screenshot loop
        if screenshot_task:
            screenshot_task.cancel()

        await session.state("verifying")
        await session.log("Run finished — verifying final state.")

        record.status = "completed"
        record.finished_at = datetime.now().isoformat()
        record.result = result_text
        save_history(record)

        await session.state("completed")
        await session.send({"type": "result", "text": result_text})

    except WebSocketDisconnect:
        session.cancelled = True
        if record:
            record.status = "cancelled"
            record.finished_at = datetime.now().isoformat()
            save_history(record)
    except Exception as exc:  # noqa: BLE001
        logger.error("Agent run failed:\n%s", traceback.format_exc())
        session.cancelled = True
        if record:
            record.status = "failed"
            record.error = str(exc)
            record.finished_at = datetime.now().isoformat()
            save_history(record)
        await session.state("failed")
        await session.send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        session.cancelled = True
        if screenshot_task:
            screenshot_task.cancel()
        try:
            if agent is not None:
                await agent.close()
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
