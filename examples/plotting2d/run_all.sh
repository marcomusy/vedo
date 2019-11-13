#!/bin/bash
# source run_all.sh
#
printf "\033c"

echo #############################################
echo    Press Esc at anytime to skip example
echo #############################################
echo
echo

echo Running annotations.py
python annotations.py    

echo Running customAxes.py
python customAxes.py

echo Running fonts.py
python fonts.py    

echo Running fxy.py
python fxy.py

echo Running histogram.py
python histogram.py

echo Running histoHexagonal.py
python histoHexagonal.py

echo Running donutPlot.py
python donutPlot.py

echo Running latex.py
python latex.py

echo Running markers.py
python markers.py

echo Running markpoint.py
python markpoint.py    

echo Running numpy2picture.py
python numpy2picture.py

echo Running plotxy.py
python plotxy.py

echo Running polarPlot.py
python polarPlot.py

echo Running polarHisto.py
python polarHisto.py

