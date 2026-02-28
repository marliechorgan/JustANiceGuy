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
        t = time.time() * 2.5
        levels = getattr(self, "_levels_idx", [0]*14)
        vol = sum(levels)
        
        # Super minimal, mesmerizing 3D globe using dense ASCII (monochrome)
        GRID_X = 54
        GRID_Y = 22
        
        # High contrast pixel density characters, perfectly visible on light and dark terminals
        chars = " .',-~+:;=!*#$@"
        
        lines = []
        
        # Reactive radius
        R = 0.85
        if is_speaking:
            R += math.sin(t * 3.0) * 0.04
        elif not muted:
            R += (vol / 150.0)

        # Reactive rotation speeds
        rot_y = t * 1.5 if is_speaking else t * 0.4
        rot_z = t * 0.8 if is_speaking else t * 0.15
        
        cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
        cos_z, sin_z = math.cos(rot_z), math.sin(rot_z)
        
        for y in range(GRID_Y):
            row = ""
            for x in range(GRID_X):
                nx = (x - (GRID_X / 2.0)) / (GRID_X / 2.0)
                ny = (y - (GRID_Y / 2.0)) / (GRID_Y / 2.0)
                ny *= 1.8 # aspect ratio correction for typical terminal fonts
                
                d2 = nx*nx + ny*ny
                
                if d2 < R*R:
                    nz = math.sqrt(R*R - d2)
                    px, py, pz = nx/R, ny/R, nz/R
                    
                    # Y rotation
                    rx1 = px * cos_y + pz * sin_y
                    rz1 = -px * sin_y + pz * cos_y
                    # Z rotation
                    rx2 = rx1 * cos_z - py * sin_z
                    ry2 = rx1 * sin_z + py * cos_z
                    rz2 = rz1
                    
                    # Lat/Lon for parametric wireframe
                    lat = math.asin(max(-1.0, min(1.0, ry2)))
                    lon = math.atan2(rz2, rx2)
                    
                    grid_w = 0.15 # wireframe thickness
                    grid1 = abs((lon * 6.0 / math.pi) % 1.0 - 0.5) < grid_w
                    grid2 = abs((lat * 6.0 / math.pi) % 1.0 - 0.5) < grid_w
                    on_wire = grid1 or grid2
                    
                    # Dynamic lighting
                    lx, ly, lz = -0.6, -0.6, 0.5
                    light = max(0.0, px*lx + py*ly + pz*lz)
                    
                    if is_speaking:
                        # Pulsing internal glow
                        light += 0.4 + math.sin(t * 8.0) * 0.1
                    elif not muted:
                        # Audio reactive lighting peaks
                        band_idx = int((x / GRID_X) * 14)
                        band_idx = max(0, min(13, band_idx))
                        light += (levels[band_idx] / 12.0)
                    
                    # Density assignment
                    if on_wire:
                        char_val = (light * 1.5) + 0.2
                    else:
                        char_val = light * 0.5
                        
                    idx = int(char_val * (len(chars) - 1))
                    idx = max(0, min(len(chars) - 1, idx))
                    
                    char = chars[idx]
                    
                    # Pure layout with generic text emphasis, no explicit colors
                    if idx > len(chars) - 4:
                        row += f"[bold]{char}[/bold]"
                    elif idx > len(chars) // 3:
                        row += f"{char}"
                    else:
                        row += f"[dim]{char}[/dim]"
                else:
                    row += " "
            lines.append(row)
            
        lines.append("")
        if UIState.last_user:
            lines.append(f"[dim]You:[/dim] {UIState.last_user}")
        if UIState.last_agent:
            lines.append(f"[bold]JARVIS:[/bold] {UIState.last_agent}")
            
        lines.append("")
        controls = "Controls: \\[m] Mute Mic | \\[l] Toggle Logs"
        lines.append(f"[dim]{controls}[/dim]")
        if muted:
            lines.append(f"[bold]MIC MUTED[/bold]")
            
        return Panel(
            Align.center(Text.from_markup("\n".join(lines))), 
            title="[bold]JARVIS Quantum Core[/bold]",
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
