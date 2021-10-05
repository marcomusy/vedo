#!/bin/bash
# source run_all.sh
#
echo #############################################
echo    Press Esc at anytime to skip example
echo #############################################
echo
echo

echo Running clone2d.py
python3 clone2d.py

echo Running flag_labels.py
python3 flag_labels.py

echo Running icon.py
python3 icon.py

echo Running inset.py
python3 inset.py

echo Running vpolyscope.py
python3 vpolyscope.py

echo Running meshio_read.py
python3 meshio_read.py

echo Running pygmsh_cut.py
python3 pygmsh_cut.py

echo Running nevergrad_opt.py
python3 nevergrad_opt.py

echo Running non_blocking.py
python3 non_blocking.py

echo Running qt_window.py # needs qt5
python3 qt_window1.py

echo Running qt_window_split.py # needs qt5
python qt_window2.py

echo Running qt_tabs.py # needs qt5
python3 qt_tabs.py

echo Running self_org_maps2d.py
python3 self_org_maps2d.py

echo Running value-iteration.py
python3 value-iteration.py

echo Running remesh_meshfix.py
python3 remesh_meshfix.py

echo Running spherical_harmonics1.py
python3 spherical_harmonics1.py

echo Running export_numpy.py
python3 export_numpy.py
