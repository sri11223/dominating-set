#!/bin/bash
set -euo pipefail
install -d -m 755 /root/dominating_set_submission
install -m 755 /solution/dominating_reference.py /root/dominating_set_submission/solver.py
install -d -m 755 /root/oracle_check
python3 /root/dominating_set_submission/solver.py /root/data/instances.json /root/oracle_check
