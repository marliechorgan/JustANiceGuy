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
        
        lines = []
        if is_speaking:
            # Abstract blue morphing hologram for TTS
            color = "cyan"
            for y in range(8):
                row = ""
                for x in range(14):
                    v = math.sin(t*3 + x*0.8 + y*0.5) + math.cos(t*2 - x*0.4)
                    if v > 0.4:
                        row += "█"
                    elif v > -0.2:
                        row += "▒"
                    else:
                        row += "░"
                lines.append(f"[{color}]{row}[/{color}]")
        else:
            # Abstract green/grey input responsive visualizer mimicking a face/core
            main_color = "red" if muted else ("bright_green" if vol > 5 else "grey37")
            for y in range(8):
                row = ""
                for x in range(14):
                    # Symmetrical visualizer using the 14 levels mapped from center out
                    idx = x if x < 7 else 13 - x
                    val = levels[idx]
                    if 7 - y <= val:
                        row += f"[{main_color}]█[/{main_color}]"
                    else:
                        row += f"[grey37]░[/grey37]"
                lines.append(row)
                
        lines.append("")
        if UIState.last_user:
            lines.append(f"[dim]You:[/dim] {UIState.last_user}")
        if UIState.last_agent:
            lines.append(f"[bold cyan]JARVIS:[/bold cyan] {UIState.last_agent}")
            
        lines.append("")
        lines.append("[dim]Controls: \\[m] Mute Mic | \\[l] Toggle Logs[/dim]")
        if muted:
            lines.append("[bold red]MIC MUTED[/bold red]")
            
        return Panel(
            Align.center(Text.from_markup("\n".join(lines))), 
            title="[bold]JARVIS Quantum Core[/bold]",
            border_style="cyan" if is_speaking else ("red" if muted else "green"),
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
