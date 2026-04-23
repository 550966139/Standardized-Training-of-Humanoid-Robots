#!/usr/bin/env bash
# Simple background supervisor: start / stop / status / logs
set -euo pipefail
cd "$(dirname "$0")/.."

PID_FILE="data/hrtrain.pid"
LOG_FILE="data/hrtrain.log"
mkdir -p data

start() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "already running (pid $(cat "$PID_FILE"))"
    return 0
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export HRTRAIN_PORT="${HRTRAIN_PORT:-6006}"
  export HRTRAIN_HOST="${HRTRAIN_HOST:-0.0.0.0}"
  nohup hrtrain > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 2
  if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "started (pid $(cat "$PID_FILE")) on :$HRTRAIN_PORT"
  else
    echo "failed to start — see $LOG_FILE"
    exit 1
  fi
}

stop() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    kill "$(cat "$PID_FILE")"
    echo "stopped"
  else
    echo "not running"
  fi
  rm -f "$PID_FILE"
}

status() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "running (pid $(cat "$PID_FILE"))"
  else
    echo "stopped"
    [ -f "$PID_FILE" ] && rm -f "$PID_FILE"
  fi
}

case "${1:-start}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; start ;;
  status)  status ;;
  logs)    tail -f "$LOG_FILE" ;;
  *)       echo "usage: $0 {start|stop|restart|status|logs}"; exit 2 ;;
esac
