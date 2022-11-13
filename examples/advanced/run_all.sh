#!/bin/bash
# source run_all.sh
#
echo #############################################
echo    Press Esc at anytime to skip example
echo #############################################
echo
echo

for f in *.py
    do
        if [[ "$f" == *"geological_model"* ]]; then continue; fi

        echo "Processing $f script.."
        python3 "$f"
    done
