#!/bin/bash
# source run_all.sh
#
echo Press Esc at anytime to skip example, F1 to interrupt

for f in *.py
    do
        echo "----------------------------------------"
        echo "Processing $f script.."
        python "$f"
    done
