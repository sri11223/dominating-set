#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
chmod -R go-rwx /tests
chmod 700 /logs/verifier
mkdir -p /root
chmod 755 /root
python3 /tests/harness_export.py

solver_log=$(mktemp)
if ! python3 /harness_run.py > "$solver_log" 2>&1; then
    echo "solver harness exited nonzero"
fi
cat "$solver_log"
pkill -KILL -u nobody 2>/dev/null || true
final_cleanup_ok=1
if pgrep -u nobody >/dev/null 2>&1; then
    echo "cleanup failed" >&2
    final_cleanup_ok=0
fi

grading_ok=0
if /opt/private-grader/bin/python -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_certificate.py -rA -p no:cacheprovider; then
    grading_ok=1
fi

if [ "$final_cleanup_ok" -eq 1 ] && [ "$grading_ok" -eq 1 ]; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
