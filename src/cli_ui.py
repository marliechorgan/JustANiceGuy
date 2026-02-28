import time
import math
import threading
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.live import Live

import livekit.agents.cli.cli as lk_cli
from livekit.agents.cli.readchar import key, readkey

class UIState:
    tts_playing = False
    stt_muted = False
    ui_mode = True # No logs by default
    last_user = ""
    last_agent = ""
    start_time = None
    llm_status = ""  # e.g. "thinking...", "503 Server Error", "429 Rate Limited"
    llm_error_count = 0

def setup_cli_ui():
    """Monkey-patches the LiveKit Agents CLI to provide a pixelated avatar UI."""

    # 1. Hide logs when ui_mode is on
    _original_emit = lk_cli.RichLoggingHandler.emit
    def _custom_emit(self, record):
        if UIState.ui_mode:
            return
        _original_emit(self, record)
    lk_cli.RichLoggingHandler.emit = _custom_emit

    # 2. Patch the audio visualizer to draw our cool pixel avatar
    def _custom_rich(self):
        is_speaking = UIState.tts_playing
        muted = UIState.stt_muted
        t = time.time()
        levels = getattr(self, "_levels_idx", [0]*14)
        vol = sum(levels)
        
        # Optimised 3D sphere with organic surface animation
        GX, GY = 40, 18
        chars = " .,:;+*#@"
        NC = len(chars) - 1
        lines = []
        
        # Reactive radius
        R = 0.82
        if is_speaking:
            R += math.sin(t * 6.0) * 0.06 + 0.04
        elif not muted:
            R += min(vol / 100.0, 0.15)

        # Startup animation
        if UIState.start_time is not None:
            elapsed = t - UIState.start_time
            if elapsed < 2.0:
                R *= (elapsed / 2.0) ** 2

        # Rotation
        spd = 2.0 if is_speaking else 0.6
        ry = t * spd
        cy, sy = math.cos(ry), math.sin(ry)
        
        for y in range(GY):
            row = []
            ny = (y - GY / 2.0) / (GY / 2.0) * 2.0  # aspect corrected
            ny2 = ny * ny
            for x in range(GX):
                nx = (x - GX / 2.0) / (GX / 2.0)
                d2 = nx * nx + ny2
                
                if d2 < R * R:
                    nz = math.sqrt(R * R - d2)
                    # Rotate around Y axis
                    rx = nx * cy + nz * sy
                    rz = -nx * sy + nz * cy
                    
                    # Organic noise: overlapping sine waves on the rotated surface
                    noise = (
                        math.sin(rx * 8.0 + t * 3.0) * 0.3 +
                        math.sin(ny * 6.0 + t * 2.0) * 0.2 +
                        math.sin((rx + ny) * 5.0 - t * 4.0) * 0.2
                    )
                    
                    # Base lighting from upper-left
                    light = max(0.0, nx * (-0.5) + ny * (-0.7) + nz * 0.5)
                    
                    if is_speaking:
                        # Orbiting bright spot + heavy pulsing noise
                        ox = math.cos(t * 4.0) * 0.6
                        oy = math.sin(t * 4.0) * 0.6
                        spot = max(0.0, 1.0 - ((nx - ox)**2 + (ny - oy)**2) * 3.0)
                        light += spot * 0.7 + noise * 0.5 + 0.15
                    elif not muted and vol > 0:
                        # Audio-reactive surface ripple
                        band = int((x / GX) * 14)
                        band = max(0, min(13, band))
                        light += levels[band] / 10.0 + noise * (vol / 40.0)
                    else:
                        # Gentle idle breathing
                        light += noise * 0.15 + math.sin(t * 1.5) * 0.05
                    
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
        # Status line
        if UIState.llm_status:
            if UIState.llm_error_count > 0:
                lines.append(f"[bold reverse] {UIState.llm_status} [/bold reverse]")
            else:
                lines.append(f"[dim]{UIState.llm_status}[/dim]")
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
                        # Clear logs completely off the screen!
                        print("\033c", end="", flush=True)

        listener = threading.Thread(target=_listen_for_keys, daemon=True)
        listener.start()

        if UIState.start_time is None:
            UIState.start_time = time.time()
            if UIState.ui_mode:
                print("\033c", end="", flush=True)

        # Check initial mute state
        c.set_microphone_enabled(not UIState.stt_muted, device=input_device)
        c.set_speaker_enabled(True, device=output_device)

        visualizer = lk_cli.FrequencyVisualizer(c, label=c.input_name or "unknown")
        visualizer.update()

        with Live(visualizer, console=c.console, refresh_per_second=15, transient=True):
            while not ctrl_t_e.is_set():
                visualizer.update()
                time.sleep(0.05)

        c.set_microphone_enabled(False)
        c.set_speaker_enabled(False)
        if ctrl_t_e.is_set():
            raise lk_cli._ToggleMode()

    lk_cli._audio_mode = _custom_audio_mode
