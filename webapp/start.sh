#!/bin/bash
cd "$(dirname "$0")"
PORT=3000
echo "Starting Splat Analyzer at http://localhost:$PORT"
python3 -m http.server $PORT &
SERVER_PID=$!
sleep 0.5
open "http://localhost:$PORT"
echo "Server PID $SERVER_PID — press Ctrl+C to stop"
wait $SERVER_PID
