#!/usr/bin/env zsh
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv311"
APP="$ROOT/flow_umap_analysis/gui/umap_explorer.py"
PORT="${PORT:-8501}"
URL="http://localhost:$PORT"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing virtual environment at $VENV"
  exit 1
fi

if [[ ! -f "$APP" ]]; then
  echo "Missing app file: $APP"
  exit 1
fi

# Kill previous app if it is already using this port.
for pid in $(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true); do
  kill "$pid" 2>/dev/null || true
done

LOG="$ROOT/flowumap_server.log"
"$VENV/bin/python" -m streamlit run "$APP" --server.headless true --server.address 0.0.0.0 --server.port "$PORT" > "$LOG" 2>&1 &
PID=$!

echo "Starting FlowUMAP on $URL"
for i in {1..120}; do
  if curl -fsS "$URL" >/dev/null 2>&1; then
    echo "Server ready at $URL"
    open "$URL" || true
    wait "$PID"
    exit 0
  fi
  sleep 1
done

echo "Server did not become ready within timeout. Log:"
cat "$LOG" || true
exit 1
