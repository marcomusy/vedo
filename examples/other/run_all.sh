#!/bin/bash
# source run_all.sh
#
echo #############################################
echo    Press Esc at anytime to skip example
echo #############################################
echo
echo

echo Running clone2d.py
python clone2d.py

echo Running flag_labels.py
python flag_labels.py

echo Running icon.py
python icon.py

echo Running inset.py
python inset.py

echo Running vpolyscope.py
python vpolyscope.py

echo Running meshio_read.py
python meshio_read.py

echo Running nevergrad_opt.py
python nevergrad_opt.py

echo Running non_blocking.py
python non_blocking.py

echo Running qt_window.py # needs qt5
python qt_window1.py

echo Running qt_window_split.py # needs qt5
python qt_window2.py

echo Running qt_tabs.py # needs qt5
python qt_tabs.py

echo Running self_org_maps2d.py
python self_org_maps2d.py

echo Running value-iteration.py
python value-iteration.py

echo Running remesh_meshfix.py
python remesh_meshfix.py

echo Running spherical_harmonics1.py
python spherical_harmonics1.py 

echo Running export_numpy.py
python export_numpy.py
