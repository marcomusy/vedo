#!/bin/bash
# source run_all.sh
#
set -e

for f in test_*.py
do
    echo "Processing $f script.."
    python3 "$f"
done

echo '---------'
echo "All good."
echo '---------'
