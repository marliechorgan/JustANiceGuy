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
        
        # Optimised 3D sphere — sparse, liquid, evolving, mesmerising
        GX, GY = 56, 24
        ASPECT = 2.1  # terminal char aspect ratio compensation
        chars = " .,:;+*#@"
        NC = len(chars) - 1
        lines = []
        
        # Liquid base radius
        R = 0.85
        if is_speaking:
            R += math.sin(t * 2.0) * 0.04
        elif vol > 0:
            R += min(vol / 150.0, 0.05)

        # Original tumbling "wobble" rotation
        spd = 0.8 if is_speaking else 0.3
        ry = t * spd
        cy, sy = math.cos(ry), math.sin(ry)
        
        rx = math.sin(t * 0.4) * 0.5  # Tumbling X axis
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
                    # This gives the "wobble" of the overall shape
                    px = nx * cy + nz * sy
                    pz = -nx * sy + nz * cy
                    py = ny
                    
                    px2 = px
                    py2 = py * cx + pz * sx
                    pz2 = -py * sx + pz * cx
                    
                    # Smooth, slow liquid noise acting as "density"
                    n1 = math.sin(px2 * 5.0 + t * 1.2) * math.cos(py2 * 4.0 - t * 0.8)
                    n2 = math.cos((px2 + py2) * 3.0 + t * 1.5)
                    n3 = math.sin(pz2 * 6.0 - t * 1.0)
                    
                    # The wave height defines the "liquid" surface volume
                    wave = (n1 + n2 + n3) / 3.0
                    
                    if is_speaking or vol > 0:
                        # Speech ripples the liquid geometry vividly
                        wave += math.sin(math.sqrt(px2**2 + py2**2) * 10.0 - t * 8.0) * 0.3 * (vol / 50.0)
                    
                    # DENSITY THRESHOLD: Not all pixels are lit!
                    # This makes it sparse, creating floating liquid blobs
                    density_threshold = 0.15 if is_speaking else 0.25
                    if wave < density_threshold:
                        row.append(" ")
                        continue
                    
                    # Base character index calculation based on wave height above threshold
                    bright = (wave - density_threshold) / (1.0 - density_threshold)
                    idx = int(bright * NC)
                    idx = max(0, min(NC, idx))
                    ch = chars[idx]
                    
                    # Rare mesmerising flickers (fireflies)
                    # High frequency math, only triggered if value > 0.95
                    sparkle = False
                    if (math.sin(px2 * 20 - t * 4) * math.cos(py2 * 25 + t * 5) * math.sin(pz2 * 30 - t * 6)) > 0.95:
                        sparkle = True
                    
                    # Evolving liquid color palette based on position, amplitude, and time
                    if sparkle:
                        # Rare, intense flickers
                        colors = ["#FFFFFF", "#FFFFAA", "#AAFFFF", "#FFAAFF"]
                        cidx = int(abs(px2 * 11 + py2 * 13 + t * 5)) % len(colors)
                        row.append(f"[bold {colors[cidx]}]{ch}[/]")
                    else:
                        # Liquid gradient mapping
                        if bright > 0.7:
                            row.append(f"[bold bright_magenta]{ch}[/]" if int(px2 * 10) % 2 == 0 else f"[bold bright_cyan]{ch}[/]")
                        elif bright > 0.4:
                            row.append(f"[magenta]{ch}[/]" if int(py2 * 10) % 2 == 0 else f"[cyan]{ch}[/]")
                        elif bright > 0.1:
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
