#!/usr/bin/env python3
"""
Agent Core
==========

This module adapts the autonomous browser agent from
`cast_fixed_v4.py` for use behind a web server instead of a CLI.

What changed from the original script, and why
------------------------------------------------
- No self-bootstrapping virtualenv. On Render, dependencies are
  installed once at build time (requirements.txt); relaunching a
  subprocess per request would be slow and fragile in a server
  context, so that logic (originally at the top of the script)
  is removed here. Pin the same versions in requirements.txt
  instead.
- No global GOOGLE_API_KEY read from the environment. Every run
  is created via `AgentRunner(api_key=...)`, so multiple users'
  keys are never mixed and nothing sensitive touches disk.
- No Rich console / Live UI. State and log lines are pushed
  through an `on_step` async callback instead, which server.py
  forwards to the browser over a WebSocket.
- Added `capture_screenshot()`, using Browser Use's live
  `agent.browser_session` to grab a JPEG frame the frontend can
  show as a "live viewport" while the agent works.
- The custom Toolbox actions (fetch_url, read/write/list files,
  the pandas table tools, run_python, system_info) are preserved
  as-is, since they are useful, general-purpose agent tools and
  do not depend on any local-only assumptions beyond a writable
  working directory, which Render provides.

Everything else — the task prompt shape, the Google Sheets
specialist auto-detection, the retry/backoff policy, and the
tool set — mirrors the original script.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import io
import json
import os
import random
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from browser_use import Agent, ChatGoogle, Tools as BrowserUseTools

WORKSPACE_DIR = Path(os.getenv("AGENT_WORKSPACE_DIR", "./agent_data/workspace"))
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    import numpy as np  # noqa: F401
except ImportError:  # pragma: no cover
    np = None

OnStep = Callable[[dict[str, Any]], Awaitable[None]]


# ============================================================
# SYSTEM PROMPT / SHEETS SPECIALIST (unchanged from original)
# ============================================================

SYSTEM_CAPABILITIES = """
You are an autonomous browser operator. You can see the page,
click, type, scroll, navigate, and read content. You also have
local utility tools for fetching raw HTTP responses, reading and
writing files, and analyzing tabular data with pandas. Use the
browser for anything requiring visual interaction, JavaScript,
or authentication; use the local tools for lightweight lookups
and data processing once information has been extracted.
""".strip()

SHEETS_URL_PATTERNS = (re.compile(r"docs\.google\.com/spreadsheets", re.IGNORECASE),)


def is_google_sheets_url(text: str) -> bool:
    return any(p.search(text) for p in SHEETS_URL_PATTERNS)


SHEETS_SPECIALIST_PROMPT = """
SPREADSHEET SPECIALIST MODE
============================

You are now operating on a live Google Sheet. Before making any
destructive edit (deleting rows/columns, overwriting existing
data, clearing ranges):

1. Inspect the sheet's current structure (headers, row count,
   which columns hold what) using the browser's view of the page.
2. State a short step-by-step plan of what you are about to do.
3. Prefer additive changes (new columns/rows) over destructive
   ones unless the task explicitly asks for replacement.
4. After editing, re-read the affected range to confirm the
   change matches intent before considering the step complete.

Use the local pandas table tools (load_table/analyze_table/
edit_table/save_table/table_to_grid) to prepare data offline
when a transformation is complex, then paste the resulting grid
back into the sheet via the browser.
""".strip()


def build_task(task: str) -> str:
    return f"""
{SYSTEM_CAPABILITIES}

USER TASK
=========

{task}

EXECUTION REQUIREMENT
=====================

Complete the task end-to-end.
Do not stop after merely explaining how it could be done.
Actually operate the browser where appropriate.
Continuously verify the current state.
If something fails because of a temporary browser/network
problem, recover and continue.
If the website presents a legitimate authentication screen,
stop and allow the user to authenticate rather than attempting
to bypass it.
At the end, report the verified outcome.
""".strip()


def is_retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "429", "rate limit", "resource exhausted", "too many requests",
        "quota", "timeout", "timed out", "temporarily unavailable",
        "service unavailable", "connection reset", "connection aborted",
        "connection refused", "502", "503", "504",
    )
    return any(m in message for m in markers)


# ============================================================
# TOOLBOX (adapted from cast_fixed_v4.py's Toolbox/build_tools)
# ============================================================

class Toolbox:
    def __init__(self) -> None:
        self.http_client = None
        self._tables: dict[str, Any] = {}

    async def _client(self):
        import httpx
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=120, follow_redirects=True)
        return self.http_client

    async def close(self):
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None

    async def fetch_url(self, url: str) -> dict[str, Any]:
        client = await self._client()
        response = await client.get(url)
        return {
            "url": str(response.url),
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type"),
            "text": response.text[:100_000],
        }

    def _resolve(self, filename: str) -> Path:
        path = Path(filename)
        if not path.is_absolute():
            path = WORKSPACE_DIR / path
        return path

    async def write_file(self, filename: str, content: str) -> str:
        path = self._resolve(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)

    async def read_file(self, filename: str) -> str:
        return self._resolve(filename).read_text(encoding="utf-8")

    async def list_files(self, directory: str = ".") -> list[str]:
        path = self._resolve(directory)
        if not path.exists():
            return []
        return [str(p) for p in path.rglob("*") if p.is_file()]

    def _require_pandas(self):
        if pd is None:
            raise RuntimeError("pandas is not installed on the server.")

    async def load_table(self, filename: str, sheet_name: Optional[str] = None) -> dict[str, Any]:
        self._require_pandas()
        path = self._resolve(filename)
        if not path.exists():
            raise FileNotFoundError(str(path))
        if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
            df = pd.read_excel(path, sheet_name=sheet_name or 0)
        elif path.suffix.lower() == ".tsv":
            df = pd.read_csv(path, sep="\t")
        else:
            df = pd.read_csv(path)
        self._tables[filename] = df
        return {
            "columns": list(df.columns.astype(str)),
            "rows": len(df),
            "preview": df.head(10).to_dict(orient="records"),
        }

    async def analyze_table(self, filename: str, op: str = "describe", **kwargs) -> dict[str, Any]:
        self._require_pandas()
        df = self._tables.get(filename)
        if df is None:
            await self.load_table(filename)
            df = self._tables[filename]
        if op == "describe":
            return json.loads(df.describe(include="all").to_json())
        if op == "correlate":
            return json.loads(df.corr(numeric_only=True).to_json())
        return {"error": f"Unsupported op: {op}"}

    async def edit_table(self, filename: str, instructions: str) -> str:
        self._require_pandas()
        # Deliberately conservative: this does not eval arbitrary
        # instructions. It records the intent; complex transforms
        # should go through run_python for auditability.
        return f"Noted edit instructions for {filename}: {instructions}"

    async def save_table(self, filename: str, out_filename: str) -> str:
        self._require_pandas()
        df = self._tables.get(filename)
        if df is None:
            raise RuntimeError(f"No loaded table for {filename}")
        out_path = self._resolve(out_filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() in (".xlsx", ".xlsm"):
            df.to_excel(out_path, index=False)
        else:
            df.to_csv(out_path, index=False)
        return str(out_path)

    async def table_to_grid(self, filename: str) -> list[list[Any]]:
        self._require_pandas()
        df = self._tables.get(filename)
        if df is None:
            await self.load_table(filename)
            df = self._tables[filename]
        return [list(df.columns.astype(str))] + df.astype(object).where(df.notna(), "").values.tolist()

    async def run_python(self, code: str) -> str:
        """Sandboxed-ish local execution for quick analysis/file edits."""
        buf = io.StringIO()
        local_ns: dict[str, Any] = {"pd": pd, "np": np, "Path": Path}
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<agent_run_python>", "exec"), local_ns)  # noqa: S102
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}\n{buf.getvalue()}"
        return buf.getvalue() or "(no output)"

    async def edit_file_inplace(self, filename: str, old: str, new: str) -> str:
        path = self._resolve(filename)
        text = path.read_text(encoding="utf-8")
        if old not in text:
            raise ValueError("old text not found in file")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return str(path)

    async def system_info(self) -> dict[str, Any]:
        import importlib.metadata
        try:
            bu = importlib.metadata.version("browser-use")
        except importlib.metadata.PackageNotFoundError:
            bu = "unknown"
        return {"python": sys.version, "platform": sys.platform, "browser_use_version": bu}


def build_tools(toolbox: Toolbox) -> Any:
    tools = BrowserUseTools()

    @tools.action(description="Fetch a URL over plain HTTP (no JS rendering). Prefer the browser for interactive pages.")
    async def fetch_url(url: str) -> str:
        return json.dumps(await toolbox.fetch_url(url))

    @tools.action(description="Write text content to a file in the agent's workspace.")
    async def write_file(filename: str, content: str) -> str:
        return await toolbox.write_file(filename, content)

    @tools.action(description="Read a text file from the agent's workspace.")
    async def read_file(filename: str) -> str:
        return await toolbox.read_file(filename)

    @tools.action(description="List files in the agent's workspace.")
    async def list_files(directory: str = ".") -> str:
        return json.dumps(await toolbox.list_files(directory))

    @tools.action(description="Load a CSV/XLSX file into a cached DataFrame and return a preview.")
    async def load_table(filename: str, sheet_name: Optional[str] = None) -> str:
        return json.dumps(await toolbox.load_table(filename, sheet_name))

    @tools.action(description="Run describe/correlate analysis on a previously loaded table.")
    async def analyze_table(filename: str, op: str = "describe") -> str:
        return json.dumps(await toolbox.analyze_table(filename, op))

    @tools.action(description="Record edit instructions for a table (use run_python for actual transforms).")
    async def edit_table(filename: str, instructions: str) -> str:
        return await toolbox.edit_table(filename, instructions)

    @tools.action(description="Save a cached DataFrame to CSV/XLSX.")
    async def save_table(filename: str, out_filename: str) -> str:
        return await toolbox.save_table(filename, out_filename)

    @tools.action(description="Convert a cached table to a 2D grid suitable for pasting into a spreadsheet UI.")
    async def table_to_grid(filename: str) -> str:
        return json.dumps(await toolbox.table_to_grid(filename))

    @tools.action(description="Execute short Python for local analysis or file edits. stdout is returned.")
    async def run_python(code: str) -> str:
        return await toolbox.run_python(code)

    @tools.action(description="Replace an exact text match inside a workspace file.")
    async def edit_file_inplace(filename: str, old: str, new: str) -> str:
        return await toolbox.edit_file_inplace(filename, old, new)

    @tools.action(description="Report Python/platform/browser-use versions.")
    async def system_info() -> str:
        return json.dumps(await toolbox.system_info())

    return tools


# ============================================================
# RUNNER
# ============================================================

class AgentRunner:
    """One isolated agent run: its own LLM client, tools, and
    browser session. Create a new instance per task."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.1-flash-lite",
        max_steps: int = 60,
        max_failures: int = 5,
        step_timeout: int = 180,
        headless: bool = True,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_steps = max_steps
        self.max_failures = max_failures
        self.step_timeout = step_timeout
        self.headless = headless

        self.toolbox = Toolbox()
        self.tools = build_tools(self.toolbox)
        self.llm = ChatGoogle(model=model, api_key=api_key)
        self._agent: Optional[Agent] = None

    async def capture_screenshot(self) -> Optional[tuple[str, str]]:
        """Returns (base64_jpeg, current_url) or None if unavailable."""
        agent = self._agent
        if agent is None:
            return None
        try:
            session = getattr(agent, "browser_session", None)
            if session is None:
                return None
            b64 = await session.take_screenshot()
            if not b64:
                return None
            current_url = ""
            try:
                page = await session.get_current_page()
                current_url = getattr(page, "url", "") or ""
            except Exception:
                pass
            return b64, current_url
        except Exception:
            return None

    async def run(self, task: str, on_step: Optional[OnStep] = None) -> str:
        async def emit(kind: str, **kwargs) -> None:
            if on_step:
                await on_step({"kind": kind, **kwargs})

        prompt = build_task(task)
        started_in_sheets = is_google_sheets_url(task)
        if started_in_sheets:
            await emit("log", message="Google Sheets URL detected — activating spreadsheet specialist.")
            prompt = f"{prompt}\n\n{SHEETS_SPECIALIST_PROMPT}"

        agent_params = inspect.signature(Agent).parameters
        agent_kwargs: dict[str, Any] = {
            "task": prompt,
            "llm": self.llm,
            "headless": self.headless,
        }
        if "tools" in agent_params:
            agent_kwargs["tools"] = self.tools
        elif "controller" in agent_params:
            agent_kwargs["controller"] = self.tools

        if "max_actions_per_step" in agent_params:
            agent_kwargs["max_actions_per_step"] = 5
        if "max_failures" in agent_params:
            agent_kwargs["max_failures"] = self.max_failures
        if "llm_timeout" in agent_params:
            agent_kwargs["llm_timeout"] = 120
        if "step_timeout" in agent_params:
            agent_kwargs["step_timeout"] = self.step_timeout

        agent = Agent(**agent_kwargs)
        self._agent = agent

        sheets_mode_activated = started_in_sheets

        async def on_step_end(live_agent: Agent) -> None:
            nonlocal sheets_mode_activated
            try:
                urls = live_agent.history.urls()
                if urls:
                    await emit("log", message=f"Step complete — current URL: {urls[-1]}")
            except Exception:
                pass

            if sheets_mode_activated:
                return
            try:
                urls = live_agent.history.urls()
            except Exception:
                return
            if not urls or not is_google_sheets_url(urls[-1] or ""):
                return
            sheets_mode_activated = True
            await emit("log", message="Live navigation into Sheets detected — activating specialist mode.")
            live_agent.add_new_task(
                f"{live_agent.task}\n\n{SHEETS_SPECIALIST_PROMPT}\n\n"
                "IMPORTANT: You have just navigated into Google Sheets. "
                "Inspect its structure and follow the specialist workflow "
                "before continuing with the original task."
            )

        run_params = inspect.signature(agent.run).parameters
        run_kwargs: dict[str, Any] = {}
        if "max_steps" in run_params:
            run_kwargs["max_steps"] = self.max_steps
        if "on_step_end" in run_params:
            run_kwargs["on_step_end"] = on_step_end

        await emit("log", message="Browser session starting...")
        result = await agent.run(**run_kwargs)

        model_outputs = getattr(result, "all_model_outputs", None)
        results = getattr(result, "all_results", None)
        errors = [str(getattr(item, "error")) for item in (results or []) if getattr(item, "error", None)]

        if model_outputs == []:
            raise RuntimeError(
                "Zero model outputs produced. "
                f"Last step error: {errors[-1] if errors else 'unknown'}"
            )
        if results and all(getattr(item, "error", None) for item in results):
            raise RuntimeError(f"All steps failed. Last step error: {errors[-1] if errors else 'unknown'}")

        final_text = None
        try:
            final_text = result.final_result()
        except Exception:
            pass
        return final_text or str(result)

    async def close(self) -> None:
        await self.toolbox.close()
        agent = self._agent
        if agent is not None:
            try:
                session = getattr(agent, "browser_session", None)
                if session is not None:
                    await session.close()
            except Exception:
                pass
