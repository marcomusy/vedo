#!/bin/bash
# source run_all.sh
#
for f in test_*.py
do
    echo "Processing $f script.."
    python3 "$f"
done
