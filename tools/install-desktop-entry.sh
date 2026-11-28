#!/bin/sh
# Adds "droid-harness Studio" to your desktop's app menu (Linux, freedesktop menus).
# Run from the checkout:  sh tools/install-desktop-entry.sh
# Remove it again:        rm ~/.local/share/applications/droid-harness-studio.desktop
set -eu
repo=$(cd "$(dirname "$0")/.." && pwd)
bin="$repo/.venv/bin/droid-harness"
[ -x "$bin" ] || { echo "no $bin: create the venv first (README, Install)" >&2; exit 1; }
dir="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$dir"
cat > "$dir/droid-harness-studio.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=droid-harness Studio
Comment=Watch, play and refine the Clash Royale bot
Exec=sh -c 'cd "$repo" && exec "$bin" studio >> "$repo/.studio/studio.log" 2>&1'
Path=$repo
Terminal=false
Icon=applications-games
Categories=Development;Game;
DESKTOP
mkdir -p "$repo/.studio"
echo "Installed $dir/droid-harness-studio.desktop (opens the Studio for $repo)"
