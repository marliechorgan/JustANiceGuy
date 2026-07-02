#!/usr/bin/env python3
"""Generate minimal Bebas Neue wordmark icons and install them into the app
bundles. Scheme: Personal = white bg / black text; Defyner (work) = inverted
(black bg / white text); Scout = personal family (white / black).

Pipeline: SVG -> rsvg-convert (1024 PNG) -> sips (iconset sizes) -> iconutil
(.icns) -> copy into <app>/Contents/Resources/<iconfile>.icns, preserving each
bundle's existing CFBundleIconFile name, then re-sign ad-hoc if needed.
Run:  python3 make-icons.py
"""
import subprocess
import tempfile
from pathlib import Path

APPS = Path.home() / "Applications"

# (app name, background, foreground, big word, subtitle)
ICONS = [
    ("JARVIS Personal", "#FFFFFF", "#000000", "JARVIS", "PERSONAL"),
    ("JARVIS Defyner",  "#000000", "#FFFFFF", "JARVIS", "DEFYNER"),
    ("Scout",           "#FFFFFF", "#000000", "OS",     ""),
]

FONT = "Bebas Neue"


def svg(bg, fg, big, sub) -> str:
    if sub:
        # Two-line lockup, vertically centred as a block (~y=512).
        text = (
            f'<text x="512" y="560" font-family="{FONT}" font-size="340" '
            f'letter-spacing="6" text-anchor="middle" fill="{fg}">{big}</text>'
            f'<text x="512" y="702" font-family="{FONT}" font-size="116" '
            f'letter-spacing="38" text-anchor="middle" fill="{fg}" '
            f'fill-opacity="0.92">{sub}</text>'
        )
    else:
        # Single word, optically centred.
        text = (
            f'<text x="512" y="648" font-family="{FONT}" font-size="400" '
            f'letter-spacing="8" text-anchor="middle" fill="{fg}">{big}</text>'
        )
    # Hairline border only when the tile is white, so it reads as an edge on a
    # light Dock; black tiles need none.
    border = ('<rect x="64" y="64" width="896" height="896" rx="200" fill="none" '
              'stroke="#000000" stroke-opacity="0.10" stroke-width="3"/>'
              if bg.upper() == "#FFFFFF" else "")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" '
            f'viewBox="0 0 1024 1024">'
            f'<rect x="64" y="64" width="896" height="896" rx="200" fill="{bg}"/>'
            f'{border}{text}</svg>')


def build_icns(tag, bg, fg, big, sub, workdir: Path) -> Path:
    sp = workdir / f"{tag}.svg"
    png = workdir / f"{tag}_1024.png"
    sp.write_text(svg(bg, fg, big, sub))
    subprocess.run(["rsvg-convert", "-w", "1024", "-h", "1024", str(sp), "-o", str(png)], check=True)
    iconset = workdir / f"{tag}.iconset"
    iconset.mkdir(exist_ok=True)
    specs = [(16, "icon_16x16.png"), (32, "icon_16x16@2x.png"),
             (32, "icon_32x32.png"), (64, "icon_32x32@2x.png"),
             (128, "icon_128x128.png"), (256, "icon_128x128@2x.png"),
             (256, "icon_256x256.png"), (512, "icon_256x256@2x.png"),
             (512, "icon_512x512.png"), (1024, "icon_512x512@2x.png")]
    for sz, fn in specs:
        subprocess.run(["sips", "-z", str(sz), str(sz), str(png), "--out", str(iconset / fn)],
                       check=True, capture_output=True)
    icns = workdir / f"{tag}.icns"
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)], check=True)
    return icns


def _plist_get(plist: Path, key: str) -> str:
    r = subprocess.run(["/usr/libexec/PlistBuddy", "-c", f"Print :{key}", str(plist)],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def install(name: str, icns: Path) -> None:
    app = APPS / f"{name}.app"
    plist = app / "Contents" / "Info.plist"
    res = app / "Contents" / "Resources"
    res.mkdir(parents=True, exist_ok=True)
    # Preserve the bundle's existing icon filename (Scout uses "AppIcon").
    iconfile = _plist_get(plist, "CFBundleIconFile") or "icon"
    iconfile = iconfile.removesuffix(".icns")
    subprocess.run(["cp", str(icns), str(res / f"{iconfile}.icns")], check=True)
    if not _plist_get(plist, "CFBundleIconFile"):
        subprocess.run(["/usr/libexec/PlistBuddy", "-c",
                        f"Add :CFBundleIconFile string {iconfile}", str(plist)],
                       capture_output=True)
    # Re-sign ad-hoc if the bundle was signed (Scout) so changing Resources
    # doesn't invalidate it; harmless no-op concern for the unsigned launchers.
    signed = subprocess.run(["codesign", "-dv", str(app)], capture_output=True).returncode == 0
    if signed:
        subprocess.run(["codesign", "--force", "--sign", "-", str(app)], capture_output=True)
    subprocess.run(["touch", str(app)])
    lsreg = ("/System/Library/Frameworks/CoreServices.framework/Frameworks/"
             "LaunchServices.framework/Support/lsregister")
    subprocess.run([lsreg, "-f", str(app)], capture_output=True)


def main():
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        for (name, bg, fg, big, sub) in ICONS:
            if not (APPS / f"{name}.app").exists():
                print(f"skip {name}: bundle missing")
                continue
            tag = name.replace(" ", "_")
            icns = build_icns(tag, bg, fg, big, sub, wd)
            install(name, icns)
            print(f"installed icon -> {name}.app")
    # Refresh the icon caches.
    subprocess.run(["killall", "Dock"], capture_output=True)
    subprocess.run(["killall", "Finder"], capture_output=True)
    print("Done. (Dock + Finder restarted to pick up the new icons.)")


if __name__ == "__main__":
    main()
