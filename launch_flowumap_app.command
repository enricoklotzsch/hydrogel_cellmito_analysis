#!/usr/bin/env zsh
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
APP="$ROOT/dist/FlowUMAP.app"

if [[ ! -d "$APP" ]]; then
  echo "App bundle not found: $APP"
  echo "Build it first with: pyinstaller --onedir --windowed --name FlowUMAP flow_umap_analysis/gui/umap_explorer.py"
  exit 1
fi

# Remove macOS quarantine if it blocks opening the app.
xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true

echo "Opening FlowUMAP app…"
open "$APP"
