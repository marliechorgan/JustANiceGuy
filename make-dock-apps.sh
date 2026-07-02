#!/usr/bin/env bash
# Build two double-clickable .app launchers (one per world) in ~/Applications,
# so "JARVIS Personal" and "JARVIS Defyner" can each be pinned to the Dock as
# separate icons. Each opens a Terminal.app window running ./start.sh <mode>.
# Re-run any time to regenerate. Then: drag each .app from ~/Applications to the Dock.
#
# Note: we use Terminal.app (not Ghostty) — Ghostty on macOS drops the launch
# command on a real Dock double-click (its single-instance handoff opens a plain
# shell instead of running -e/command). Terminal's AppleScript `do script` is the
# reliable path.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
APPS_DIR="$HOME/Applications"
mkdir -p "$APPS_DIR"

make_app() {
    local name="$1" mode="$2"
    local app="$APPS_DIR/${name}.app"
    rm -rf "$app"
    mkdir -p "$app/Contents/MacOS"

    cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>${name}</string>
    <key>CFBundleDisplayName</key><string>${name}</string>
    <key>CFBundleIdentifier</key><string>ai.justaniceguy.jarvis.${mode}</string>
    <key>CFBundleExecutable</key><string>run</string>
    <key>CFBundleIconFile</key><string>icon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
</dict>
</plist>
PLIST

    # The launcher: open a Terminal.app window running start.sh in the chosen mode.
    cat > "$app/Contents/MacOS/run" <<RUN
#!/bin/bash
osascript -e 'tell application "Terminal" to do script "cd ${PROJECT_DIR} && ./start.sh ${mode}"' \\
          -e 'tell application "Terminal" to activate'
RUN
    chmod +x "$app/Contents/MacOS/run"
    # Refresh Launch Services so the new bundle is recognised immediately.
    /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$app" 2>/dev/null || true
    echo "built: $app  (mode: ${mode})"
}

make_app "JARVIS Personal" "personal"
make_app "JARVIS Defyner" "defyner"

# Generate + install the energy-core icons (emerald = Personal, violet = Defyner).
if command -v rsvg-convert >/dev/null 2>&1; then
    python3 "$PROJECT_DIR/make-icons.py" || echo "(icon generation skipped)"
else
    echo "(rsvg-convert not found — skipping icons; brew install librsvg to enable)"
fi

echo
echo "Done. Open ~/Applications and drag 'JARVIS Personal' and 'JARVIS Defyner' to your Dock."
open "$APPS_DIR"
