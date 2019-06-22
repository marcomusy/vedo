#!/bin/bash
# source run_all.sh
#
printf "\033c"

echo #############################################
echo    Press F1 at anytime to skip example
echo #############################################
echo
echo
echo


################################### simulations
echo Running aspring.py
python aspring.py

echo Running cell_main.py
python cell_main.py

echo Running brownian2D.py
python brownian2D.py

echo Running gas.py
python gas.py

echo Running gyroscope1.py
python gyroscope1.py

echo Running gyroscope2.py
python gyroscope2.py

echo Running multiple_pendulum.py
python multiple_pendulum.py

echo Running hanoi3d.py
python hanoi3d.py

echo Running airplanes.py
python airplanes.py

echo Running pendulum.py
python pendulum.py

echo Running wave_equation.py
python wave_equation.py

echo Running turing.py
python turing.py

echo Running particle_simulator.py
python particle_simulator.py

echo Running doubleslit.py
python doubleslit.py

echo Running tunnelling2.py
python tunnelling2.py

echo Running volterra.py
python volterra.py

echo '-----------------------------'
echo '-----------------------------'
echo 'cd ../other/dolfin; ./run_all.sh'
echo '-----------------------------'
echo '-----------------------------'
