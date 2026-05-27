"""Liberator Kiro gateway. Two endpoints:
  POST /v1/chat/completions   — OpenAI-shaped text-only synthesis (Opus 4.6).
  POST /v1/vision/extract     — vision endpoint: takes base64 image + prompt,
                                spawns kiro-cli with --trust-all-tools so the
                                built-in `read` tool fires, returns extracted
                                JSON / text.

Why a subprocess: `ksk_*` keys authenticate to AWS CodeWhisperer via the
kiro-cli binary (sigv4 + token refresh). The CLI does the auth dance; we
feed it the prompt and let its `read` tool open the temp image file.

Image-input gotcha (verified in probe): kiro-cli does NOT accept base64
on stdin — it must be a filesystem path. So we save the b64 to a temp
file, hand kiro-cli the absolute path inside the prompt, and clean up
after the response lands.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import secrets
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request

KIRO_API_KEY = os.environ.get("KIRO_API_KEY", "")
PROXY_API_KEY = os.environ.get("PROXY_API_KEY", "")
DEFAULT_MODEL = os.environ.get("KIRO_MODEL", "claude-opus-4.6")
KIRO_BIN = os.environ.get("KIRO_BIN", "kiro-cli")

if not KIRO_API_KEY:
    raise RuntimeError("KIRO_API_KEY env var required")
if not PROXY_API_KEY:
    PROXY_API_KEY = secrets.token_urlsafe(32)
    print(f"PROXY_API_KEY (generated): {PROXY_API_KEY}", flush=True)

app = FastAPI()

# ANSI escapes + kiro-cli's status chatter must be stripped before parsing.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_CREDITS = re.compile(r"Credits:\s*([0-9.]+)")
_JSON_BLOB = re.compile(r"\{.*\}", re.DOTALL)


def _strip(text: str) -> str:
    return _ANSI.sub("", text)


def _extract_response(stdout: str) -> str:
    """kiro-cli prefixes responses with '> '. Take everything after the LAST
    one (covers cases where the model's own output contains '>' characters)."""
    cleaned = _strip(stdout)
    # The last '> ' on its own marks the model response start.
    idx = cleaned.rfind("\n> ")
    if idx == -1:
        idx = cleaned.rfind("> ")
        if idx == -1:
            return cleaned.strip()
        return cleaned[idx + 2:].strip()
    body = cleaned[idx + 3:]
    # Trim trailing "Credits: ... • Time: ..." footer.
    body = re.split(r"\n\s*▸?\s*Credits:", body)[0]
    return body.strip()


async def _kiro_subprocess(prompt: str, model: str, timeout: float = 120) -> tuple[str, float]:
    """Run kiro-cli with ONLY the fs_read tool trusted — the read tool is the
    only one the gateway ever needs (for reading the temp image file). Every
    other tool (fs_write, execute_bash, etc.) stays denied, so even if the
    model is prompted into writing/executing something, kiro-cli refuses.
    Verified locally: fs_write returns 'rejected ... on the denied list.'
    Returns (response_text, credits_used)."""
    proc = await asyncio.create_subprocess_exec(
        KIRO_BIN, "chat",
        "--no-interactive",
        "--model", model,
        "--trust-tools=fs_read",
        prompt,
        env={**os.environ, "KIRO_API_KEY": KIRO_API_KEY},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(504, "kiro-cli timed out")
    if proc.returncode != 0:
        raise HTTPException(502, f"kiro-cli rc={proc.returncode}: {stderr_b.decode('utf-8','ignore')[:400]}")
    stdout = stdout_b.decode("utf-8", "ignore")
    response = _extract_response(stdout)
    m = _CREDITS.search(_strip(stdout))
    credits = float(m.group(1)) if m else 0.0
    return response, credits


def _check_auth(request: Request) -> None:
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer ") or auth.removeprefix("Bearer ").strip() != PROXY_API_KEY:
        raise HTTPException(401, "bad bearer token")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict[str, Any]:
    """Text-only synthesis. Used by the backend's MULTI_PAPER/EDGE_CASE jobs."""
    _check_auth(request)
    body = await request.json()
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(400, "messages[] required")
    prompt = "\n\n".join(m.get("content", "") for m in messages if m.get("content"))
    model = body.get("model") or DEFAULT_MODEL
    text, credits = await _kiro_subprocess(prompt, model)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "credits": credits},
    }


@app.post("/v1/vision/extract")
async def vision_extract(request: Request) -> dict[str, Any]:
    """Vision endpoint. JSON body:
       {
         "image_b64": "...",          # required, raw base64 (no data URI prefix)
         "prompt": "...",             # required, the VISION_PROMPT text
         "model": "claude-opus-4.6",  # optional
         "mime": "image/jpeg"         # optional; suffix for the temp file
       }
    Returns: {"text": <raw response>, "json": <parsed JSON if found, else null>,
             "credits": float, "model": str}
    """
    _check_auth(request)
    body = await request.json()
    image_b64 = body.get("image_b64") or ""
    prompt_text = body.get("prompt") or ""
    model = body.get("model") or DEFAULT_MODEL
    mime = body.get("mime") or "image/jpeg"
    if not image_b64 or not prompt_text:
        raise HTTPException(400, "image_b64 and prompt required")

    # Strip optional data URI prefix the client might've sent.
    if image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[-1]

    try:
        img_bytes = base64.b64decode(image_b64)
    except Exception:
        raise HTTPException(400, "bad base64")

    suffix = ".png" if "png" in mime else ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(img_bytes)
        tmp.flush()
        tmp.close()
        # The prompt MUST tell kiro-cli to read the file via its built-in tool.
        full_prompt = f"Read the image at {tmp.name} and follow these instructions:\n\n{prompt_text}"
        text, credits = await _kiro_subprocess(full_prompt, model, timeout=90)
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except Exception:
            pass

    # Try to surface a JSON blob if the prompt asked for one.
    parsed = None
    m = _JSON_BLOB.search(text)
    if m:
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            parsed = None
    return {"text": text, "json": parsed, "credits": credits, "model": model}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": DEFAULT_MODEL}
