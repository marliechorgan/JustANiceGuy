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
        
        # Optimised 3D sphere — mesmerizing rotation, sparkling organic surface
        GX, GY = 56, 24
        ASPECT = 2.1  # terminal char aspect ratio compensation
        chars = " .,:;+*#@"
        NC = len(chars) - 1
        lines = []
        
        # Reactive radius with subtle breathing
        R = 0.88
        breath = math.sin(t * 1.5) * 0.02
        if is_speaking:
            R += math.sin(t * 5.0) * 0.03 + breath
        elif not muted and vol > 0:
            R += min(vol / 120.0, 0.06) + breath
        else:
            R += breath

        # Startup animation
        if UIState.start_time is not None:
            elapsed = t - UIState.start_time
            if elapsed < 2.0:
                R *= (elapsed / 2.0) ** 2

        # 3D Rotation Matrix Calculation
        # Spin around Y axis (faster when speaking)
        spd = 2.0 if is_speaking else 0.5
        ry = t * spd
        cy, sy = math.cos(ry), math.sin(ry)
        
        # Gentle tilt on X axis to see the "poles" rotating
        rx = 0.4
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
                    
                    # Apply INVERSE rotation to map current 2D screen coordinate
                    # back to the 3D surface of the rotating sphere.
                    # Undo X tilt:
                    ty = ny * cx + nz * sx
                    tz = -ny * sx + nz * cx
                    tx = nx
                    
                    # Undo Y spin:
                    ox = tx * cy + tz * sy
                    oz = -tx * sy + tz * cy
                    oy = ty
                    
                    # Evaluate organic noise using the locked 3D surface coordinates (ox, oy, oz)
                    # This makes the "texture" spin perfectly with the sphere
                    noise = (
                        math.sin(ox * 7.0 + t * 2.0) * math.cos(oy * 6.0) * 0.2 +
                        math.sin(oy * 5.0 - t * 1.0) * math.cos(oz * 4.0) * 0.2 +
                        math.sin((ox + oz) * 5.0) * 0.1
                    )
                    
                    # High-frequency sparkling noise
                    sparkle = 0
                    if is_speaking or (not muted and vol > 0):
                        # Extreme high frequency intersecting waves
                        s1 = math.sin(ox * 25.0 + t * 18.0)
                        s2 = math.cos(oy * 20.0 - t * 15.0)
                        s3 = math.sin(oz * 30.0 + t * 22.0)
                        val = s1 * s2 * s3
                        
                        # Lowered threshold to create far more specks
                        if val > 0.15:
                            lat = max(0.0, 1.0 - abs(oy) * 1.5)
                            sparkle = val * lat * 2.5
                    
                    # Base directional lighting (top-left, fixed to screen)
                    light = max(0.0, nx * (-0.4) + ny * (-0.6) + nz * 0.6)
                    
                    # Rim glow at the edges of the sphere
                    rim = (1.0 - nz / R) ** 2.0 * 0.3
                    
                    # Calculate final brightness
                    bright = light + noise + rim
                    
                    if is_speaking or (not muted and vol > 0):
                        # Audio reactivity - map freq bands around the sphere's equator (lon/lat)
                        lat = max(0.0, 1.0 - abs(oy) * 1.5)  # Intensity drops off near poles
                        lon = (math.atan2(oz, ox) / math.pi + 1.0) / 2.0  # Range 0.0 to 1.0
                        band_idx = int(lon * 13)
                        band_val = levels[band_idx] / 10.0 if band_idx < len(levels) else 0
                        
                        if is_speaking:
                            # Speaking mode: Audio bands + Front-facing pulsing waves + sparks
                            front_dist = math.sqrt(nx*nx + ny*ny)
                            pulse = math.sin(front_dist * 10.0 - t * 8.0) * 0.3 * max(0, 1.0 - front_dist * 1.5)
                            bright += band_val * lat * 0.5 + pulse + 0.15 + sparkle * 0.5
                        else:
                            # Listening mode: Pure audio bands reacting around the surface + sparks
                            bright += band_val * lat * 0.8 + sparkle * 0.5
                    else:
                        # Idle: Just slow breathing shift
                        bright += math.sin(t) * 0.05
                    
                    # Map brightness to character index
                    idx = int(max(0.0, min(1.0, bright)) * NC + 0.5)
                    idx = max(0, min(NC, idx))
                    ch = chars[idx]
                    
                    # Mesmerizing color formatting
                    if is_speaking or (not muted and vol > 0):
                        if sparkle > 0.2:
                            # Mesmerizing, flickering neon palette
                            neon = ["#00FFFF", "#FF00FF", "#FFFF00", "#00FF66", "#FF3399", "#9933FF", "#FFFFFF", "#00CCFF"]
                            # Rapidly shifting color index based on 3D coordinate and time
                            cidx = int(abs(math.sin(ox*12 + oz*8 + t*4)) * len(neon)) % len(neon)
                            
                            if sparkle > 0.5:
                                row.append(f"[bold {neon[cidx]}]{ch}[/]")
                            else:
                                row.append(f"[{neon[cidx]}]{ch}[/]")
                        elif is_speaking:
                            if idx > NC - 2:
                                row.append(f"[bold cyan]{ch}[/bold cyan]")
                            elif idx > NC // 2:
                                row.append(f"[cyan]{ch}[/cyan]")
                            else:
                                row.append(f"[dim cyan]{ch}[/dim cyan]")
                        else: # user speaking (listening)
                            if idx > NC - 2:
                                row.append(f"[bold]{ch}[/bold]")
                            elif idx > NC // 2:
                                row.append(ch)
                            else:
                                row.append(f"[dim]{ch}[/dim]")
                    else: # idle
                        if idx > NC - 2:
                            row.append(f"[bold]{ch}[/bold]")
                        elif idx > NC // 2:
                            row.append(ch)
                        else:
                            row.append(f"[dim]{ch}[/dim]")
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
