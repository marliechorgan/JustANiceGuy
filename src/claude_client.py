"""Claude Code integration — dispatch directives to a headless Claude Code session.

Drop-in replacement for `openclaw_client.dispatch_openclaw`. Same contract:
shells out to a backend CLI as an async subprocess, pushes a `status_update`
QueueItem immediately, narrates tool calls as further `status_update`s (UI only),
then pushes a single `agent_result` QueueItem whose `summary` is the text the
voice loop will speak.

Why Claude Code instead of OpenClaw:
  - A headless session at the right cwd already loads Charlie's CLAUDE.md, the
    memory tree, gog/Perplexity/Notion, and the skills — so it can *explain*
    things, not just execute them.
  - `--resume <session_id>` gives native cross-turn memory (the OpenClaw
    "jarvis-voice-persistent" caching trick, but first-class).

Routing:
  target="defyner"  -> cwd ~/Defyner   (loads the Defyner repo CLAUDE.md + eng memory)
  target="personal" -> cwd ~           (loads the life-OS root CLAUDE.md + life memory)

Streaming:
  We run `--output-format stream-json --verbose` and parse newline-delimited
  events. `assistant` messages carry tool_use blocks -> narrated as status
  updates (UI only; they do NOT trigger the voice LLM). The terminal `result`
  event carries the final text, session_id, and cost.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

from voice_queue import QueueItem, VoiceLLMQueue
from cli_ui import UIState

logger = logging.getLogger("niceguy.claude")

# ─── Config (all overridable via .env) ──────────────────────────────────────
# Sonnet 4.6: faster + cheaper than Opus for the doer role, and strong enough
# for investigation/dispatch. Override with CLAUDE_MODEL.
DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
AGENT_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", "300"))

# Permission mode for the headless session. A voice agent has no TTY to approve
# prompts, so anything that would prompt simply hangs the turn. bypassPermissions
# keeps the loop alive; the directive prefix below is the real guardrail
# (read/investigate only — no sends, pushes, deletes, or money moves).
# Tighten to "acceptEdits" or "dontAsk" (+ allowedTools) via CLAUDE_PERMISSION_MODE.
PERMISSION_MODE = os.environ.get("CLAUDE_PERMISSION_MODE", "bypassPermissions")

# Where to root each target. cwd selection is what makes the correct CLAUDE.md load.
TARGET_DIRS = {
    "defyner": os.path.expanduser(os.environ.get("CLAUDE_DEFYNER_DIR", "~/Defyner")),
    "personal": os.path.expanduser(os.environ.get("CLAUDE_PERSONAL_DIR", "~")),
}

# Per-target world isolation — mirrors the shell's `claude` (personal) vs
# `claudew` (Defyner) so each headless session gets the SAME config dir, account,
# MCP servers, skills, hooks and memory as a normal terminal session.
#   personal -> default ~/.claude config + keychain account (the Max plan).
#   defyner  -> CLAUDE_CONFIG_DIR=~/.claude-defyner + the work account token
#               (~/.config/claude-switch/account-w.token) + GH_TOKEN, exactly
#               like the `claudew` shell function.
DEFYNER_CONFIG_DIR = os.path.expanduser(
    os.environ.get("CLAUDE_DEFYNER_CONFIG_DIR", "~/.claude-defyner"))
DEFYNER_TOKEN_FILE = os.path.expanduser(
    os.environ.get("CLAUDE_DEFYNER_TOKEN_FILE", "~/.config/claude-switch/account-w.token"))

# Auth-override vars cleared before selecting an account (matches claude-switch).
_AUTH_OVERRIDES = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_REFRESH_TOKEN",
    "CLAUDE_CODE_OAUTH_SCOPES", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY", "CLAUDE_CODE_USE_MANTLE",
)


def _claude_env(target: str) -> dict:
    """Build the subprocess environment for a target so the headless session
    matches the corresponding terminal world (`claude` vs `claudew`)."""
    env = os.environ.copy()
    for k in _AUTH_OVERRIDES:
        env.pop(k, None)
    if target == "defyner":
        env["CLAUDE_CONFIG_DIR"] = DEFYNER_CONFIG_DIR
        try:
            tok = Path(DEFYNER_TOKEN_FILE).read_text().strip()
        except Exception:
            tok = ""
        if tok:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = tok
            env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        else:
            logger.warning("Defyner work-account token missing (%s) — using inherited auth",
                           DEFYNER_TOKEN_FILE)
        gh = os.environ.get("GITHUB_DEFYNER_TOKEN")
        if gh:
            env["GH_TOKEN"] = gh
    else:
        # personal: default ~/.claude + keychain account — strip any inherited
        # work-account overrides so we don't accidentally bill/scope to Defyner.
        for k in ("CLAUDE_CONFIG_DIR", "CLAUDE_CODE_OAUTH_TOKEN",
                  "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "GH_TOKEN"):
            env.pop(k, None)
    return env


def _config_dir_for(target: str) -> str:
    return DEFYNER_CONFIG_DIR if target == "defyner" else os.path.expanduser("~/.claude")

# Persist one resume-able session per target so context survives across turns
# (and across process restarts within a working day).
_SESSION_FILE = Path(__file__).resolve().parent.parent / ".jarvis_claude_sessions.json"

# Guardrail + voice framing prepended to every directive.
VOICE_DIRECTIVE_PREFIX = (
    "[VOICE COMMAND VIA JARVIS — backend execution context]\n"
    "You are the backend 'doer' for a real-time voice assistant. Another AI speaks "
    "your output aloud, so write your FINAL message as natural spoken prose: no "
    "markdown, no bullet lists, no code blocks, no emojis. Be terse and concrete — "
    "lead with the answer. If you investigated, give the finding, not a play-by-play.\n"
    "SAFETY: This is a read/investigate context. Do NOT send email/Slack/WhatsApp, "
    "do NOT git commit or push, do NOT move money, do NOT delete or overwrite files. "
    "If the task would require one of those, STOP and say what you would do and why "
    "it needs confirmation, instead of doing it.\n\n"
)

# Friendly narration for tool starts (UI status only — not spoken).
_TOOL_VERBS = {
    "Read": "Reading files",
    "Edit": "Editing a file",
    "Write": "Writing a file",
    "Bash": "Running a command",
    "Grep": "Searching the code",
    "Glob": "Looking for files",
    "Task": "Spawning a sub-agent",
    "Agent": "Spawning a sub-agent",
    "WebFetch": "Fetching a page",
    "WebSearch": "Searching the web",
}


def _load_sessions() -> dict:
    try:
        return json.loads(_SESSION_FILE.read_text())
    except Exception:
        return {}


def _save_session(target: str, session_id: str) -> None:
    try:
        data = _load_sessions()
        data[target] = session_id
        _SESSION_FILE.write_text(json.dumps(data))
    except Exception as e:
        logger.warning("Could not persist Claude session id: %s", e)


def _narrate_tool(name: str) -> str:
    if name.startswith("mcp__perplexity"):
        return "Researching"
    if name.startswith("mcp__"):
        return "Using a connected tool"
    return _TOOL_VERBS.get(name, f"Using {name}")


# ─── Live event bus ─────────────────────────────────────────────────────────
# Every dispatch streams structured events here as newline-delimited JSON. The
# standalone inspector (src/inspector.py / ./watch.sh) tails this to render a
# real-time view of what each headless Claude session is doing. Append-only;
# gitignored; the inspector keeps only recent events in memory.
EVENT_BUS = Path(os.environ.get(
    "JARVIS_EVENT_BUS",
    str(Path(__file__).resolve().parent.parent / "logs" / "live_events.jsonl"),
))


_dispatch_seq = 0


def _next_dispatch_id(target: str) -> str:
    global _dispatch_seq
    _dispatch_seq += 1
    return f"{target[:3]}-{_dispatch_seq:04d}"


def _emit(kind: str, **fields) -> None:
    """Record one event: (1) into the in-process live UI state for the console
    sessions view, and (2) onto the on-disk event bus for any external viewer.
    Never raises into the dispatch."""
    rec = {"kind": kind, "ts": time.time(), **fields}
    try:
        UIState.push_event(rec)
    except Exception:
        pass
    try:
        EVENT_BUS.parent.mkdir(parents=True, exist_ok=True)
        with EVENT_BUS.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


def _tool_detail(name: str, tool_input: dict) -> str:
    """Extract the human-meaningful argument from a tool_use block so the
    inspector can show WHAT the session is doing, not just which tool."""
    if not isinstance(tool_input, dict):
        return ""
    for key in ("file_path", "path", "command", "pattern", "query", "url",
                "prompt", "description", "directive", "old_string"):
        v = tool_input.get(key)
        if v:
            s = str(v).replace("\n", " ").strip()
            return s[:160]
    # Fall back to a compact dump of the first value.
    for v in tool_input.values():
        if v:
            return str(v).replace("\n", " ").strip()[:160]
    return ""


def _clean_directive(text: str) -> str:
    """Strip JARVIS's voice-directive prefix from a transcript user message so a
    peek shows the actual instruction, not the boilerplate."""
    if VOICE_DIRECTIVE_PREFIX[:30] in text:
        text = text.replace(VOICE_DIRECTIVE_PREFIX, "")
    return text.strip().replace("\n", " ")


def read_session_tail(session_id: str, n: int = 4) -> str:
    """Return the last `n` meaningful turns (assistant text / tool calls / user
    messages) from a Claude session's on-disk transcript, as compact text.

    Lets the voice LLM 'grab the latest context' from any session without
    resuming it. Transcripts live at ~/.claude/projects/<encoded-cwd>/<id>.jsonl;
    we glob by id since it's globally unique.
    """
    import glob
    # Search both worlds' config dirs (personal ~/.claude and Defyner ~/.claude-defyner).
    matches = []
    for base in ("~/.claude", "~/.claude-defyner"):
        matches += glob.glob(os.path.expanduser(f"{base}/projects/*/{session_id}.jsonl"))
    if not matches:
        return f"(no transcript found for session {session_id[:8]})"
    path = max(matches, key=os.path.getmtime)
    rows = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = e.get("type")
                msg = e.get("message", {}) if isinstance(e.get("message"), dict) else {}
                content = msg.get("content")
                if etype == "assistant" and isinstance(content, list):
                    for b in content:
                        bt = b.get("type")
                        if bt == "text" and b.get("text", "").strip():
                            rows.append("assistant: " + b["text"].strip().replace("\n", " ")[:240])
                        elif bt == "tool_use":
                            rows.append(f"  ⚒ {b.get('name','')}  {_tool_detail(b.get('name',''), b.get('input',{}))}".rstrip())
                elif etype == "user" and isinstance(content, str) and content.strip():
                    rows.append("user: " + _clean_directive(content)[:240])
                elif etype == "user" and isinstance(content, list):
                    for b in content:
                        if b.get("type") == "text" and b.get("text", "").strip():
                            rows.append("user: " + _clean_directive(b["text"])[:240])
    except Exception as ex:
        return f"(could not read transcript: {ex})"
    if not rows:
        return "(transcript has no readable turns yet)"
    return "\n".join(rows[-n:])


async def dispatch_claude(
    directive: str,
    queue: VoiceLLMQueue,
    target: str = "personal",
    session_id: str | None = None,
) -> None:
    """Send a directive to a headless Claude Code session and push results into the queue.

    Args:
        directive: natural-language task from the voice LLM.
        queue: the VoiceLLMQueue results are pushed into.
        target: "defyner" (cwd ~/Defyner) or "personal" (cwd ~). Selects which
            CLAUDE.md / memory the session loads.
        session_id: resume THIS specific session instead of the per-target
            default — used for bidirectional control of an "entered" session.
    """
    target = target if target in TARGET_DIRS else "personal"
    cwd = TARGET_DIRS[target]
    if not os.path.isdir(cwd):
        logger.warning("Target dir %s missing for target=%s; falling back to home", cwd, target)
        cwd = os.path.expanduser("~")

    # Phase 1: immediate status update (UI + history breadcrumb; not spoken).
    queue.push(
        QueueItem(
            type="status_update",
            agent=f"claude_{target}",
            content={"target": target},
            summary=f"Dispatching to Claude ({target}): {directive[:100]}...",
        )
    )

    start = time.monotonic()
    UIState.openclaw_status = f"Claude ({target}): {directive[:40]}..."

    wrapped = VOICE_DIRECTIVE_PREFIX + directive

    # Resume an explicit session (bidirectional control of an "entered" session)
    # if given; otherwise the per-target default for cross-turn memory.
    sessions = _load_sessions()
    resume_id = session_id or sessions.get(target)

    did = _next_dispatch_id(target)
    _emit("dispatch_start", id=did, target=target, directive=directive,
          cwd=cwd, resume=resume_id)

    cmd = [
        CLAUDE_BIN, "-p", wrapped,
        "--output-format", "stream-json",
        "--verbose",                       # required for stream-json under -p
        "--model", DEFAULT_MODEL,
        "--permission-mode", PERMISSION_MODE,
    ]
    if resume_id:
        cmd += ["--resume", resume_id]

    logger.info("Claude dispatch target=%s cwd=%s resume=%s", target, cwd, resume_id or "(new)")
    logger.info("Claude directive (full): %s", directive)

    final_text: str | None = None
    new_session_id: str | None = None
    total_cost: float = 0.0
    is_error = False
    proc = None

    try:
        t_spawn = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_claude_env(target),   # match `claude`/`claudew` world per target
            # stream-json emits one JSON object per line; big tool results / the
            # final result event can exceed asyncio's default 64KB line limit and
            # raise "Separator is found, but chunk is longer than limit". Raise the
            # StreamReader buffer to 32MB so whole lines are read intact.
            limit=32 * 1024 * 1024,
        )

        async def _read_stream() -> None:
            nonlocal final_text, new_session_id, total_cost, is_error
            assert proc.stdout is not None
            async for raw_line in proc.stdout:
                line = raw_line.decode(errors="replace").strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Non-JSON line from Claude: %s", line[:200])
                    continue

                etype = evt.get("type")

                if etype == "system" and evt.get("subtype") == "init":
                    new_session_id = evt.get("session_id") or new_session_id
                    if new_session_id:
                        _emit("session", id=did, target=target, session_id=new_session_id)

                elif etype == "assistant":
                    for block in evt.get("message", {}).get("content", []):
                        btype = block.get("type")
                        if btype == "tool_use":
                            name = block.get("name", "")
                            detail = _tool_detail(name, block.get("input", {}))
                            verb = _narrate_tool(name)
                            UIState.openclaw_status = f"Claude ({target}): {verb}…"
                            # Rich event for the inspector (tool + what it's acting on).
                            _emit("tool", id=did, target=target, name=name, detail=detail)
                            # Coarse status for the voice UI (names only, not spoken).
                            queue.push(
                                QueueItem(
                                    type="status_update",
                                    agent=f"claude_{target}",
                                    content={"tool": name},
                                    summary=verb,
                                )
                            )
                        elif btype == "text":
                            txt = (block.get("text") or "").strip()
                            if txt:
                                _emit("text", id=did, target=target, text=txt[:500])

                elif etype == "result":
                    new_session_id = evt.get("session_id") or new_session_id
                    total_cost = evt.get("total_cost_usd", 0.0) or 0.0
                    if evt.get("subtype") == "success":
                        final_text = evt.get("result")
                    else:
                        is_error = True
                        final_text = evt.get("result") or f"Claude ended with {evt.get('subtype')}."

        t_spawned = time.monotonic()
        await asyncio.wait_for(_read_stream(), timeout=AGENT_TIMEOUT)
        await proc.wait()
        t_done = time.monotonic()

        logger.info(
            "Claude timing: spawn=%.0fms exec=%.0fms total=%.0fms cost=$%.4f exit=%s",
            (t_spawned - t_spawn) * 1000,
            (t_done - t_spawned) * 1000,
            (t_done - start) * 1000,
            total_cost,
            proc.returncode,
        )

        stderr_bytes = b""
        if proc.stderr is not None:
            try:
                stderr_bytes = await proc.stderr.read()
            except Exception:
                pass
        if stderr_bytes:
            logger.info("Claude stderr:\n%s", stderr_bytes.decode(errors="replace").strip()[:2000])

        UIState.openclaw_status = ""

        # Persist the session for the next turn.
        if new_session_id:
            _save_session(target, new_session_id)

        if (proc.returncode not in (0, None)) and final_text is None:
            err = stderr_bytes.decode(errors="replace").strip() or f"Exit code {proc.returncode}"
            logger.error("Claude error: %s", err)
            queue.push(
                QueueItem(
                    type="agent_result",
                    agent=f"claude_{target}",
                    content={"status": "error", "error": err, "target": target},
                    summary=f"Claude ran into a problem: {err[:200]}",
                )
            )
            UIState.session_errors["openclaw"] += 1
            _emit("error", id=did, target=target, error=err[:300])
            return

        if final_text is None:
            final_text = "Claude finished but returned no text."
            is_error = True

        logger.info("Claude result (target=%s, %d chars): %s",
                    target, len(final_text), final_text[:500])
        _emit("result", id=did, target=target, ok=not is_error,
              chars=len(final_text), cost=total_cost,
              duration_ms=int((time.monotonic() - start) * 1000),
              session_id=new_session_id, summary=final_text[:500])

        queue.push(
            QueueItem(
                type="agent_result",
                agent=f"claude_{target}",
                content={
                    "status": "error" if is_error else "success",
                    "target": target,
                    "session_id": new_session_id,
                    "cost_usd": total_cost,
                },
                summary=final_text,
            )
        )

    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        logger.error("Claude timed out after %.1fs", elapsed)
        if proc is not None:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
        # Still persist the session if we learned its id — the work may have
        # partially progressed and the next turn can resume it.
        if new_session_id:
            _save_session(target, new_session_id)
        queue.push(
            QueueItem(
                type="agent_result",
                agent=f"claude_{target}",
                content={"status": "error", "error": "timeout", "target": target},
                summary=f"That took too long — Claude timed out after {elapsed:.0f} seconds.",
            )
        )
        UIState.session_errors["openclaw"] += 1
        UIState.openclaw_status = f"Timeout ({elapsed:.0f}s)"
        _emit("error", id=did, target=target, error=f"timeout after {elapsed:.0f}s")

    except Exception as e:
        logger.error("Claude dispatch error: %s", e, exc_info=True)
        queue.push(
            QueueItem(
                type="agent_result",
                agent=f"claude_{target}",
                content={"status": "error", "error": str(e), "target": target},
                summary=f"Failed to dispatch to Claude: {e}",
            )
        )
        UIState.session_errors["openclaw"] += 1
        UIState.openclaw_status = f"Error: {type(e).__name__}"
        _emit("error", id=did, target=target, error=str(e)[:300])
