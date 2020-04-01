#!/bin/bash
# source run_all.sh
#
echo #############################################
echo    Press Esc at anytime to skip example
echo #############################################
echo
echo

echo Running colorpalette.py
python colorpalette.py

echo Running printc.py
python printc.py

echo Running flag_labels.py
python flag_labels.py

echo Running icon.py
python icon.py

echo Running inset.py
python inset.py

echo Running qt_window.py # needs qt5
python qt_window.py

echo Running qt_window_split.py # needs qt5
python qt_window_split.py

echo Running qt_tabs.py # needs qt5
python qt_tabs.py

echo Running self_org_maps2d.py
python self_org_maps2d.py

echo Running value-iteration.py
python value-iteration.py

echo Running spherical_harmonics1.py
python spherical_harmonics1.py 

echo Running tf_learn_volume.py
python tf_learn_volume.py

echo Running export_x3d.py
python export_x3d.py

echo Running create_logo.py
python create_logo.py

echo Running export_numpy.py
python export_numpy.py

echo Running save_as_numpy.py
python save_as_numpy.py
