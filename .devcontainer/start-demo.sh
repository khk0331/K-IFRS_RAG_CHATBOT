#!/usr/bin/env bash
set -eu

if ! curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
  nohup uvicorn kifrs_rag.api:app --host 0.0.0.0 --port 8000 > /tmp/kifrs-rag-demo.log 2>&1 &
fi
