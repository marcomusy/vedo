#!/bin/bash
# source run_all.sh
#
echo Press Esc at anytime to skip example, F1 to interrupt

for f in *.py; do
    case $f in qt*.py) continue;; esac
    case $f in wx*.py) continue;; esac
    case $f in trame*.py)  continue;; esac
    case $f in *video*.py) continue;; esac
    case $f in *napari*.py) continue;; esac
    echo "Processing: examples/other/$f"
    python "$f"
done