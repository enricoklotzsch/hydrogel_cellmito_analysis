#!/usr/bin/env zsh
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv311"
APP="$ROOT/flow_umap_analysis/gui/umap_explorer.py"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Virtual environment not found at $VENV"
  echo "Create it first with: python3 -m venv .venv311"
  exit 1
fi

if [[ ! -f "$APP" ]]; then
  echo "App file not found: $APP"
  exit 1
fi

PORT="${PORT:-}"
if [[ -z "$PORT" ]]; then
  PORT="$($VENV/bin/python - <<'PY'
import socket
for port in (8504, 8501, 8502, 8503, 8505, 8506, 8510, 9000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            print(port)
            break
        except OSError:
            continue
else:
    import random
    print(40000 + random.randint(1000, 9999))
PY
)"
fi

echo "Launching Flow UMAP Explorer on http://localhost:$PORT"
open "http://localhost:$PORT" || true

exec "$VENV/bin/python" -m streamlit run "$APP" \
  --server.headless true \
  --server.address 0.0.0.0 \
  --server.port "$PORT"
