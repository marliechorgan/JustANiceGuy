"""CLI UI — Monkey-patches LiveKit's console mode for an animated 3D sphere UI.

Provides:
- UIState: shared state between agent subprocess and parent UI process
- setup_cli_ui(): patches LiveKit's FrequencyVisualizer and RichLoggingHandler
"""
import time
import math
import shutil
import threading
import logging
from collections import deque
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.live import Live

import livekit.agents.cli.cli as lk_cli
from livekit.agents.cli.readchar import key, readkey


class UIState:
    """Shared state between the agent subprocess and the CLI renderer."""
    # Audio state
    tts_playing = False
    stt_muted = False
    
    # UI display mode (True = sphere only, False = sphere + logs)
    ui_mode = True
    
    # Conversation transcript (last messages)
    last_user = ""
    last_agent = ""
    
    # Startup timestamp for expansion animation
    start_time = None
    
    # Service status — updated by agent.py in real-time
    llm_status = ""         # "Thinking...", "503 Service Unavailable", etc.
    llm_error_count = 0     # Consecutive errors in current request
    tts_status = ""         # "Speaking...", "TTS Error", etc.
    stt_status = ""         # "Listening...", "STT Error", etc.
    openclaw_status = ""    # "Working...", "Timeout", etc.
    
    # Session-wide error tallies
    session_errors = {
        "gemini": 0,
        "elevenlabs": 0,
        "deepgram": 0,
        "openclaw": 0,
    }
    
    # Last TTFB for display
    last_ttfb_ms = 0

    # Pasted-context inbox: the keypress thread appends clipboard text here
    # (via the 'v' hotkey); the agent loop drains it into the LLM chat context
    # before the next generation. Guarded by a lock for cross-thread safety.
    pending_context: list[str] = []
    context_lock = threading.Lock()
    last_context_info = ""   # e.g. "context added: 1,240 chars" (shown in UI)

    # Live Claude-session view: claude_client._emit feeds these in-process so the
    # console can show, in real time, what each headless session is doing.
    show_sessions = False                 # toggled by the 's' hotkey
    sessions: dict = {}                   # dispatch_id -> live session state
    session_events = deque(maxlen=200)    # rolling activity feed (raw event dicts)
    sessions_lock = threading.Lock()

    # Navigation + "enter into a session" (bidirectional control).
    selected_idx = 0                      # highlighted row in the sessions list
    ordered_dids: list = []               # render writes the current row order here
    entered = False                       # True = drilled into the active session
    active_session_id = None              # voice dispatches resume THIS session
    active_session_target = None          # cwd/target of the entered session

    @classmethod
    def push_event(cls, rec: dict) -> None:
        """Fold a claude_client event into live session state (same process)."""
        kind = rec.get("kind")
        did = rec.get("id")
        tgt = rec.get("target")
        now = rec.get("ts") or time.time()
        with cls.sessions_lock:
            s = cls.sessions.get(did)
            if kind == "dispatch_start":
                cls.sessions[did] = {
                    "target": tgt, "directive": rec.get("directive", ""),
                    "status": "running", "started": now, "last": now,
                    "tools": 0, "cost": 0.0,
                    "session_id": rec.get("resume"), "activity": "starting…",
                }
            elif s is not None:
                s["last"] = now
                if kind == "session":
                    s["session_id"] = rec.get("session_id") or s.get("session_id")
                elif kind == "tool":
                    s["tools"] += 1
                    d = rec.get("detail", "")
                    s["activity"] = f"{rec.get('name','')}  {d}".strip()
                elif kind == "text":
                    s["activity"] = rec.get("text", "")[:90]
                elif kind == "result":
                    s["status"] = "done" if rec.get("ok") else "error"
                    s["cost"] = rec.get("cost", 0.0)
                    s["session_id"] = rec.get("session_id") or s.get("session_id")
                    s["activity"] = "done"
                elif kind == "error":
                    s["status"] = "error"
                    s["activity"] = rec.get("error", "error")[:90]
            cls.session_events.append(rec)
            # Keep memory bounded: retain the 12 most-recent dispatches.
            if len(cls.sessions) > 12:
                for k in list(cls.sessions)[:-12]:
                    cls.sessions.pop(k, None)


_TARGET_DIRS = {"defyner": "~/Defyner", "personal": "~"}


def _target_color(t: str) -> str:
    return {"defyner": "cyan", "personal": "green"}.get(t, "white")


def _ordered_sessions() -> list:
    """Sessions sorted as the view shows them: active first, then most recent."""
    with UIState.sessions_lock:
        items = list(UIState.sessions.items())
    items.sort(key=lambda kv: (0 if kv[1]["status"] == "running" else 1, -kv[1]["last"]))
    return items


_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_KIND_ICON = {"tool": "⚒", "text": "›", "result": "[green]✓[/]",
              "error": "[red]✗[/]", "dispatch_start": "▶", "session": "·"}


def _panel_dims():
    """Responsive panel size from the live terminal (fills the window)."""
    term = shutil.get_terminal_size((100, 40))
    width = max(72, min(term.columns - 2, 170))
    inner = width - 8                 # text width inside borders + padding
    body_rows = max(8, min(term.lines - 12, 36))
    return width, inner, body_rows


def _fit(s: str, w: str) -> str:
    s = str(s).replace("\n", " ")
    return s if len(s) <= w else s[: max(0, w - 1)] + "…"


def render_sessions_panel() -> Panel:
    """In-console live view of the headless Claude sessions. 's' toggles it;
    ↑/↓ select a session; Enter drills in (and routes your voice to it); Esc
    releases. Detail view shows that session's recent turns."""
    from rich.markup import escape as esc
    now = time.time()
    width, inner, body_rows = _panel_dims()
    spin = _SPINNER[int(now * 10) % len(_SPINNER)]
    rule = "[dim]" + "─" * inner + "[/dim]"
    items = _ordered_sessions()
    UIState.ordered_dids = [did for did, _ in items]

    def ev_text(e):
        if e.get("kind") == "tool":
            return f"{e.get('name','')} {e.get('detail','')}".strip()
        return (e.get("text") or e.get("directive") or e.get("error")
                or e.get("name") or e.get("kind") or "")

    # ── Entered/detail view: drilled into the active session ────────────────
    if UIState.entered and UIState.active_session_id:
        active = UIState.active_session_id
        with UIState.sessions_lock:
            dids = {d for d, s in UIState.sessions.items() if s.get("session_id") == active}
            events = [e for e in UIState.session_events if e.get("id") in dids]
        sess = next((s for _, s in items if s.get("session_id") == active), None)
        tgt = (sess or {}).get("target", UIState.active_session_target or "")
        tc = _target_color(tgt)
        running = (sess or {}).get("status") == "running"
        live = f"[yellow]{spin} live[/]" if running else "[green]done[/]"
        lines = [f"[reverse {tc}] ENTERED · {tgt} [/]  {live}   "
                 "[dim]your voice now talks to this session[/dim]", ""]
        for e in events[-body_rows:]:
            icon = _KIND_ICON.get(e.get("kind"), "·")
            lines.append(f"{icon} {esc(_fit(ev_text(e), inner - 2))}")
        if not events:
            lines.append("[dim](no activity captured yet)[/dim]")
        lines.append(rule)
        lines.append(f"[dim]session[/] {active[:8]}…   "
                     r"[dim]\[Esc] release  ·  \[c] copy resume cmd  ·  \[s] close[/dim]")
        return Panel(Text.from_markup("\n".join(lines)),
                     title=f"[bold]► {tgt} session[/bold]", width=width, padding=(1, 2))

    # ── List view: pick a session ───────────────────────────────────────────
    lines = []
    if not items:
        lines.append("[dim]No Claude sessions yet — ask JARVIS something, then come back.[/dim]")
    else:
        UIState.selected_idx = max(0, min(UIState.selected_idx, len(items) - 1))
    status_icon = {"running": f"[yellow]{spin}[/]", "done": "[green]✓[/]", "error": "[red]✗[/]"}
    for i, (did, s) in enumerate(items[:6]):
        tc = _target_color(s["target"])
        ic = status_icon.get(s["status"], "●")
        el = int(now - s["started"])
        sel = (i == UIState.selected_idx)
        cur = "[bold white]▸[/]" if sel else " "
        lines.append(f"{cur} {ic} [bold]{did}[/] [{tc}]{s['target']}[/]  "
                     f"{el}s  ⚒{s['tools']}  ${s['cost']:.3f}   "
                     f"[dim]{esc(_fit(s.get('activity',''), inner - 40))}[/dim]")

    lines.append(rule)
    lines.append("[dim]live activity (all sessions)[/dim]")
    with UIState.sessions_lock:
        feed = list(UIState.session_events)[-(body_rows - len(items)):]
    for e in feed:
        tgt = e.get("target", "")
        tc = _target_color(tgt)
        icon = _KIND_ICON.get(e.get("kind"), "·")
        lines.append(f"[{tc}]{tgt[:3]:<3}[/] {icon} {esc(_fit(ev_text(e), inner - 8))}")

    lines.append(rule)
    lines.append(r"[dim]↑/↓ select  ·  ↵ enter session  ·  \[c] copy resume  ·  \[s] close[/dim]")
    return Panel(Text.from_markup("\n".join(lines)),
                 title=f"[bold]JARVIS · Claude sessions[/bold]  {spin}", width=width, padding=(1, 2))


def setup_cli_ui():
    """Monkey-patches the LiveKit Agents CLI to provide a pixelated avatar UI."""

    # 1. Suppress ALL log output when sphere is showing.
    #    We do this by detaching the RichLoggingHandler from the root logger
    #    and re-attaching it when logs are toggled on. This is cleaner than
    #    patching emit() because it prevents ANY console output from the
    #    handler and its sub-calls (traceback printing, extra lines, etc.)
    _original_emit = lk_cli.RichLoggingHandler.emit
    def _custom_emit(self, record):
        if UIState.ui_mode:
            return  # Drop ALL log records when sphere is visible
        _original_emit(self, record)
    lk_cli.RichLoggingHandler.emit = _custom_emit

    # Also suppress the _print_plain_traceback method which bypasses emit()
    _original_print_tb = lk_cli.RichLoggingHandler._print_plain_traceback
    def _custom_print_tb(self, record):
        if UIState.ui_mode:
            return
        _original_print_tb(self, record)
    lk_cli.RichLoggingHandler._print_plain_traceback = _custom_print_tb

    # 2. Patch the audio visualizer to draw our cool pixel avatar
    def _custom_rich(self):
        is_speaking = UIState.tts_playing
        muted = UIState.stt_muted
        t = time.time()
        levels = getattr(self, "_levels_idx", [0]*14)
        vol = sum(levels)
        
        # Live Claude-sessions view takes over the panel when toggled ('s').
        if UIState.show_sessions:
            try:
                return render_sessions_panel()
            except Exception:
                pass  # never let the view crash the render loop

        # When logs are visible, collapse to a minimal one-line status
        if not UIState.ui_mode:
            parts = ["JARVIS"]
            if muted:
                parts.append("[bold]MUTED[/bold]")
            if UIState.llm_status:
                parts.append(UIState.llm_status)
            elif UIState.tts_status:
                parts.append(UIState.tts_status)
            elif UIState.openclaw_status:
                parts.append(UIState.openclaw_status)
            else:
                parts.append("[dim]idle[/dim]")
            if UIState.last_context_info:
                parts.append(f"[cyan]{UIState.last_context_info}[/cyan]")
            errs = {k: v for k, v in UIState.session_errors.items() if v > 0}
            if errs:
                parts.append(" ".join(f"{k}:{v}" for k, v in errs.items()))
            return Panel(
                Align.center(Text.from_markup(" | ".join(parts))),
                padding=(0, 1),
            )
        
        # Optimised 3D sphere — dense, liquid, audio-reactive lava-lamp
        GX, GY = 56, 24
        ASPECT = 2.1  # terminal char aspect ratio compensation
        chars = " .,:;+*#@"
        NC = len(chars) - 1
        lines = []
        
        # Base radius expands with volume
        R = 0.85
        if is_speaking:
            R += math.sin(t * 3.0) * 0.03 + (vol / 120.0) * 0.05
        elif vol > 0:
            R += min(vol / 100.0, 0.08)

        # Tumbling wobble rotation
        spd = 1.0 if is_speaking else 0.4
        ry = t * spd
        cy, sy = math.cos(ry), math.sin(ry)
        
        rx = math.sin(t * 0.5) * 0.4  # Tumbling X axis
        cx, sx = math.cos(rx), math.sin(rx)
        
        for y in range(GY):
            row = []
            ny = (y - GY / 2.0) / (GY / 2.0) * ASPECT
            ny2 = ny * ny
            for x in range(GX):
                nx = (x - GX / 2.0) / (GX / 2.0)
                d2 = nx * nx + ny2
                
                if d2 < R * R:
                    # Calculate depth (Z) on the sphere surface
                    nz = math.sqrt(R * R - d2)
                    
                    # Apply tumbling rotation to the visual coordinates
                    px = nx * cy + nz * sy
                    pz = -nx * sy + nz * cy
                    py = ny
                    
                    px2 = px
                    py2 = py * cx + pz * sx
                    pz2 = -py * sx + pz * cx
                    
                    # Base structural volumetric noise
                    n1 = math.sin(px2 * 6.0 + t * 1.5) * math.cos(py2 * 5.0 - t * 1.0)
                    n2 = math.cos((px2 + py2) * 4.0 + t * 2.0)
                    n3 = math.sin(pz2 * 5.0 - t * 1.5)
                    
                    wave = (n1 + n2 + n3) / 3.0
                    
                    # Heavy audio reactivity
                    # Distort the liquid geometry based on frequency bands spreading across the surface
                    if is_speaking or vol > 0:
                        lat = max(0.0, 1.0 - abs(py2))
                        lon = (math.atan2(pz2, px2) / math.pi + 1.0) / 2.0
                        band_idx = int(lon * 13)
                        lvl = levels[band_idx] / 10.0 if band_idx < len(levels) else 0
                        
                        # Audio creates intense ripples and swelling
                        ripple_freq = 15.0 + lvl * 5.0
                        audio_ripple = math.sin(math.sqrt(px2**2 + py2**2) * ripple_freq - t * 12.0) * lvl * 0.8
                        wave += audio_ripple * lat
                        
                        # Overall volume swells the density
                        wave += (vol / 60.0) * 0.3
                    
                    # DENSITY THRESHOLD: Denser rendering (more pixels lit)
                    density_threshold = -0.1 if is_speaking else 0.05
                    if wave < density_threshold:
                        row.append(" ")
                        continue
                    
                    # Calculate brightness
                    bright = (wave - density_threshold) / (1.0 - density_threshold + 0.5)
                    
                    # Add rim lighting for depth
                    rim = (1.0 - nz / R) ** 2.0 * 0.4
                    bright += rim
                    
                    # Mesmerising flickers (fireflies) - scattered across the density
                    sparkle = False
                    # Higher probability of sparks when volume is high
                    spark_thresh = 0.90 - (min(vol, 100) / 100.0) * 0.1
                    if (math.sin(px2 * 25 - t * 6) * math.cos(py2 * 28 + t * 7) * math.sin(pz2 * 32 - t * 8)) > spark_thresh:
                        sparkle = True
                    
                    idx = int(bright * NC)
                    idx = max(0, min(NC, idx))
                    ch = chars[idx]
                    
                    # Dynamic liquid colors
                    if sparkle:
                        colors = ["#FFFFFF", "#FFFFAA", "#AAFFFF", "#FFAAFF", "#FFFF00", "#00FFFF"]
                        cidx = int(abs(px2 * 15 + py2 * 17 + t * 8)) % len(colors)
                        row.append(f"[bold {colors[cidx]}]{ch}[/]")
                    else:
                        # Complex gradient mapping
                        if bright > 0.8:
                            if int(px2 * 10 + t * 3) % 2 == 0:
                                row.append(f"[bold bright_magenta]{ch}[/]")
                            else:
                                row.append(f"[bold bright_cyan]{ch}[/]")
                        elif bright > 0.5:
                            if is_speaking:
                                row.append(f"[bold bright_blue]{ch}[/]" if bright > 0.65 else f"[bright_magenta]{ch}[/]")
                            else:
                                row.append(f"[magenta]{ch}[/]" if int(py2 * 10) % 2 == 0 else f"[cyan]{ch}[/]")
                        elif bright > 0.2:
                            row.append(f"[blue]{ch}[/]")
                        else:
                            row.append(f"[dim blue]{ch}[/dim blue]")
                else:
                    row.append(" ")
            
            # Sphere is exactly 56 chars wide. We'll use a 60-char panel. 2 spaces padding.
            lines.append("  " + "".join(row))
            
        # UI string is now cleanly JUST the sphere. No text, no logs.
        return Panel(
            Text.from_markup("\n".join(lines)), 
            title="[bold]JARVIS[/bold]",
            width=60,
            padding=(1, 0)
        )
    lk_cli.FrequencyVisualizer.__rich__ = _custom_rich

    # 3. Patch _audio_mode to handle 'm' and 'l' keys seamlessly
    def _custom_audio_mode(c, *, input_device, output_device):
        ctrl_t_e = threading.Event()
        visualizer = None

        def _listen_for_keys():
            while not ctrl_t_e.is_set():
                ch = readkey()
                if ch == key.CTRL_T:
                    ctrl_t_e.set()
                    break
                elif ch == "?" and visualizer is not None:
                    visualizer.show_shortcuts = not visualizer.show_shortcuts
                elif ch == key.ESC and UIState.show_sessions and UIState.entered:
                    # Release the entered session → voice returns to normal routing.
                    UIState.entered = False
                    UIState.active_session_id = None
                    UIState.active_session_target = None
                    print("\033c", end="", flush=True)
                elif ch == key.ESC and visualizer is not None:
                    visualizer.show_shortcuts = False
                elif isinstance(ch, str) and ch.lower() == "m":
                    UIState.stt_muted = not UIState.stt_muted
                    c.set_microphone_enabled(not UIState.stt_muted)
                elif isinstance(ch, str) and ch.lower() == "l":
                    UIState.ui_mode = not UIState.ui_mode
                    if UIState.ui_mode:
                        # Clear all log output from the screen
                        print("\033c", end="", flush=True)
                elif isinstance(ch, str) and ch.lower() == "v":
                    # Paste context: ingest the macOS clipboard into the LLM
                    # context (for mermaid diagrams, articles, big text chunks).
                    # The user copies anything anywhere, then presses 'v' here.
                    try:
                        import subprocess
                        text = subprocess.run(
                            ["pbpaste"], capture_output=True, text=True, timeout=5
                        ).stdout
                    except Exception:
                        text = ""
                    text = (text or "").strip()
                    if text:
                        # Cap to keep the context window sane (~80k chars).
                        if len(text) > 80_000:
                            text = text[:80_000] + "\n[...truncated]"
                        with UIState.context_lock:
                            UIState.pending_context.append(text)
                        UIState.last_context_info = f"context added: {len(text):,} chars"
                    else:
                        UIState.last_context_info = "clipboard empty — nothing pasted"
                elif isinstance(ch, str) and ch.lower() == "s":
                    # Toggle the live Claude-sessions view.
                    UIState.show_sessions = not UIState.show_sessions
                    UIState.entered = False
                    if UIState.show_sessions:
                        print("\033c", end="", flush=True)
                elif UIState.show_sessions and ch == key.UP:
                    UIState.selected_idx = max(0, UIState.selected_idx - 1)
                elif UIState.show_sessions and ch == key.DOWN:
                    UIState.selected_idx = min(
                        max(0, len(UIState.ordered_dids) - 1),
                        UIState.selected_idx + 1)
                elif UIState.show_sessions and ch in (key.ENTER, key.CR, key.LF):
                    # Enter into the highlighted session → route voice to it.
                    dids = UIState.ordered_dids
                    if 0 <= UIState.selected_idx < len(dids):
                        with UIState.sessions_lock:
                            s = UIState.sessions.get(dids[UIState.selected_idx])
                        if s and s.get("session_id"):
                            UIState.active_session_id = s["session_id"]
                            UIState.active_session_target = s["target"]
                            UIState.entered = True
                            print("\033c", end="", flush=True)
                        else:
                            UIState.last_context_info = "session has no id yet"
                elif isinstance(ch, str) and ch.lower() == "c":
                    # Copy the most-recent session's resume command to clipboard.
                    with UIState.sessions_lock:
                        items = sorted(UIState.sessions.values(),
                                       key=lambda s: -s["last"])
                    sid = next((s.get("session_id") for s in items
                                if s.get("session_id")), None)
                    tgt = next((s["target"] for s in items
                                if s.get("session_id")), "personal")
                    if sid:
                        # Use the same world the session was created in: claudew
                        # for Defyner (sets CLAUDE_CONFIG_DIR + work account + cd),
                        # plain claude for personal.
                        if tgt == "defyner":
                            cmd = f"claudew --resume {sid}"
                        else:
                            cmd = f"cd ~ && claude --resume {sid}"
                        try:
                            import subprocess
                            subprocess.run(["pbcopy"], input=cmd, text=True, timeout=5)
                            UIState.last_context_info = f"resume cmd copied ({tgt})"
                        except Exception:
                            UIState.last_context_info = "copy failed"
                    else:
                        UIState.last_context_info = "no session to resume yet"

        listener = threading.Thread(target=_listen_for_keys, daemon=True)
        listener.start()

        if UIState.start_time is None:
            UIState.start_time = time.time()
            if UIState.ui_mode:
                print("\033c", end="", flush=True)

        c.set_microphone_enabled(not UIState.stt_muted, device=input_device)
        c.set_speaker_enabled(True, device=output_device)

        visualizer = lk_cli.FrequencyVisualizer(c, label=c.input_name or "unknown")
        visualizer.update()

        with Live(visualizer, console=c.console, refresh_per_second=20, transient=True):
            while not ctrl_t_e.is_set():
                visualizer.update()
                time.sleep(0.04)

        c.set_microphone_enabled(False)
        c.set_speaker_enabled(False)
        if ctrl_t_e.is_set():
            raise lk_cli._ToggleMode()

    lk_cli._audio_mode = _custom_audio_mode
