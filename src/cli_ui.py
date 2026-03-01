"""CLI UI — Monkey-patches LiveKit's console mode for an animated 3D sphere UI.

Provides:
- UIState: shared state between agent subprocess and parent UI process
- setup_cli_ui(): patches LiveKit's FrequencyVisualizer and RichLoggingHandler
"""
import time
import math
import threading
import logging
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
            errs = {k: v for k, v in UIState.session_errors.items() if v > 0}
            if errs:
                parts.append(" ".join(f"{k}:{v}" for k, v in errs.items()))
            return Panel(
                Align.center(Text.from_markup(" | ".join(parts))),
                padding=(0, 1),
            )
        
        # Optimised 3D sphere — fixed position, organic surface animation
        GX, GY = 56, 24
        ASPECT = 2.1  # terminal char aspect ratio compensation
        chars = " .,:;+*#@"
        NC = len(chars) - 1
        lines = []
        
        # Fixed radius with subtle breathing
        R = 0.88
        breath = math.sin(t * 1.8) * 0.02  # slow gentle pulse
        if is_speaking:
            # Stronger pulse when speaking
            R += math.sin(t * 5.0) * 0.04 + 0.03 + breath
        elif not muted and vol > 0:
            R += min(vol / 120.0, 0.08) + breath
        else:
            R += breath

        # Startup animation
        if UIState.start_time is not None:
            elapsed = t - UIState.start_time
            if elapsed < 2.0:
                R *= (elapsed / 2.0) ** 2
        
        for y in range(GY):
            row = []
            ny = (y - GY / 2.0) / (GY / 2.0) * ASPECT
            ny2 = ny * ny
            for x in range(GX):
                nx = (x - GX / 2.0) / (GX / 2.0)
                d2 = nx * nx + ny2
                
                if d2 < R * R:
                    nz = math.sqrt(R * R - d2)
                    
                    # Multi-octave organic noise (no rotation — stays in place)
                    n1 = math.sin(nx * 7.0 + t * 1.2) * math.cos(ny * 5.0 - t * 0.8) * 0.25
                    n2 = math.sin((nx + ny) * 4.0 + t * 1.5) * 0.15
                    n3 = math.sin(nx * 12.0 - t * 2.0) * math.sin(ny * 10.0 + t * 1.0) * 0.1
                    noise = n1 + n2 + n3
                    
                    # Base lighting — fixed directional (top-left, into screen)
                    light = max(0.0, nx * (-0.4) + ny * (-0.6) + nz * 0.65)
                    
                    # Fresnel-style rim glow (brighter at edges)
                    rim = (1.0 - nz / R) ** 2.5 * 0.3
                    
                    if is_speaking:
                        # Ripple emanating from centre
                        dist = math.sqrt(d2)
                        ripple = math.sin(dist * 12.0 - t * 8.0) * 0.2 * max(0, 1.0 - dist)
                        # Shimmer across surface
                        shimmer = math.sin(nx * 15.0 + t * 6.0) * math.cos(ny * 10.0 - t * 4.0) * 0.15
                        light += ripple + shimmer + noise * 0.4 + rim + 0.12
                    elif not muted and vol > 0:
                        # Audio bands drive surface glow
                        band = int((x / GX) * 14)
                        band = max(0, min(13, band))
                        light += levels[band] / 12.0 + noise * (vol / 50.0) + rim
                    else:
                        # Idle: gentle surface motion + rim highlight
                        light += noise * 0.12 + rim + math.sin(t * 1.0) * 0.03
                    
                    idx = int(max(0.0, min(1.0, light)) * NC + 0.5)
                    ch = chars[idx]
                    if idx > NC - 2:
                        row.append(f"[bold]{ch}[/bold]")
                    elif idx > NC // 2:
                        row.append(ch)
                    else:
                        row.append(f"[dim]{ch}[/dim]")
                else:
                    row.append(" ")
            lines.append("".join(row))
            
        lines.append("")
        
        # --- Status line ---
        status_parts = []
        if UIState.llm_status:
            if UIState.llm_error_count > 0:
                status_parts.append(f"[bold reverse] {UIState.llm_status} [/bold reverse]")
            else:
                status_parts.append(f"[dim]{UIState.llm_status}[/dim]")
        if UIState.tts_status:
            status_parts.append(f"[dim]{UIState.tts_status}[/dim]")
        if UIState.openclaw_status:
            status_parts.append(f"[dim]{UIState.openclaw_status}[/dim]")
        if status_parts:
            lines.append("  ".join(status_parts))
        
        # --- TTFB indicator ---
        if UIState.last_ttfb_ms > 0 and not UIState.llm_status:
            lines.append(f"[dim]Last TTFB: {UIState.last_ttfb_ms:.0f}ms[/dim]")
        
        # --- Service health bar ---
        errs = UIState.session_errors
        err_parts = []
        for svc, count in errs.items():
            if count > 0:
                err_parts.append(f"{svc}:{count}")
        if err_parts:
            lines.append(f"[bold reverse] Errors: {' | '.join(err_parts)} [/bold reverse]")
        
        lines.append("")
        if UIState.last_user:
            lines.append(f"[dim]You:[/dim] {UIState.last_user}")
        if UIState.last_agent:
            lines.append(f"[bold]JARVIS:[/bold] {UIState.last_agent}")
            
        lines.append("")
        lines.append("[dim]Controls: \\[m] Mute Mic | \\[l] Toggle Logs[/dim]")
        if muted:
            lines.append("[bold]MIC MUTED[/bold]")
            
        return Panel(
            Align.center(Text.from_markup("\n".join(lines))), 
            title="[bold]JARVIS[/bold]",
            padding=(1, 2)
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
