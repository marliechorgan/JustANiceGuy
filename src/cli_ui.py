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
    ui_mode = False
    last_user = ""
    last_agent = ""

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
        t = time.time() * 2.0
        levels = getattr(self, "_levels_idx", [0]*14)
        vol = sum(levels)
        
        # Super minimal, 3D animated sphere (the Quantum Core)
        GRID_X = 28
        GRID_Y = 14
        
        chars = [" ", "░", "▒", "▓", "█"]
        lines = []
        
        # Base radius: reacts to audio
        r = 0.8
        if not is_speaking and not muted:
            r += (vol / 120.0) # Pulses slightly when user talks
        elif is_speaking:
            r += math.sin(t * 4) * 0.05 # Pulses automatically when JARVIS talks
            
        for y in range(GRID_Y):
            row = ""
            for x in range(GRID_X):
                # Normalize coords, account for terminal character aspect ratio (~1:2)
                nx = (x - (GRID_X // 2)) / (GRID_X / 2.0)
                ny = (y - (GRID_Y // 2)) / (GRID_Y / 2.0)
                ny *= 1.8 # aspect ratio correction
                
                d = math.sqrt(nx*nx + ny*ny)
                
                if d < r:
                    # Calculate Z for 3D sphere
                    z = math.sqrt(abs(r*r - d*d)) / r
                    
                    # Normal vector (n = 1 on surface)
                    nnx, nny, nnz = nx/r, ny/r, z
                    
                    if is_speaking:
                        # Orbiting light when speaking
                        lx = math.cos(t * 3) * 0.8
                        ly = math.sin(t * 3) * 0.8
                        lz = 0.5
                    elif muted:
                        # Dim static light when muted
                        lx, ly, lz = 0.0, 0.0, 0.3
                    else:
                        # Static light, reacting to frequency band at this X position
                        band_idx = int((x / GRID_X) * 14)
                        band_idx = max(0, min(13, band_idx))
                        band_vol = levels[band_idx]
                        
                        lx, ly, lz = 0.3, -0.6, 0.8
                        # Intensity boosted locally by frequency bands to deform light
                        lz += band_vol / 10.0
                    
                    # Normalize light vector
                    ld = math.sqrt(lx*lx + ly*ly + lz*lz)
                    if ld > 0:
                        lx, ly, lz = lx/ld, ly/ld, lz/ld
                    
                    # Diffuse dot product
                    dot = nnx*lx + nny*ly + nnz*lz
                    intensity = max(0.0, dot)
                    
                    # Map to characters
                    char_idx = int(intensity * 4.99)
                    
                    if char_idx >= 3:
                         row += f"[bold white]{chars[char_idx]}[/bold white]"
                    elif char_idx > 0:
                         row += f"[white]{chars[char_idx]}[/white]" # Using white instead of grey for contrast
                    else:
                         row += f"[bold black]{chars[char_idx]}[/bold black]" # Darkest shade
                else:
                    row += " "
            lines.append(row)
            
        lines.append("")
        if UIState.last_user:
            lines.append(f"[dim]You:[/dim] {UIState.last_user}")
        if UIState.last_agent:
            lines.append(f"[bold white]JARVIS:[/bold white] {UIState.last_agent}")
            
        lines.append("")
        lines.append("[dim]Controls: \\[m] Mute Mic | \\[l] Toggle Logs[/dim]")
        if muted:
            lines.append("[bold white]MIC MUTED[/bold white]")
            
        return Panel(
            Align.center(Text.from_markup("\n".join(lines))), 
            title="[bold white]JARVIS Quantum Core[/bold white]",
            border_style="white",
            padding=(1, 4)
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

        listener = threading.Thread(target=_listen_for_keys, daemon=True)
        listener.start()

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
