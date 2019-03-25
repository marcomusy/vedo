#!/bin/bash
# source run_all.sh
#
printf "\033c"
echo Running examples in directory dolfin/

echo Running ex01_show-mesh.py
python ex01_show-mesh.py

echo Running ex02_tetralize-mesh.py
python ex02_tetralize-mesh.py

echo Running ex03_poisson.py
python ex03_poisson.py

echo Running ex04_mixed-poisson.py
python ex04_mixed-poisson.py

echo Running ex05_non-matching-meshes.py
python ex05_non-matching-meshes.py

echo Running ex06_elasticity1.py
python ex06_elasticity1.py

echo Running ex06_elasticity2.py
python ex06_elasticity2.py

echo Running ex07_stokes-iterative.py
python ex07_stokes-iterative.py



##########################
echo Running ascalarbar.py
python ascalarbar.py

echo Running collisions.py
python collisions.py

echo Running markmesh.py
python markmesh.py

echo Running elastodynamics.py
python elastodynamics.py




######################################
echo 
echo
echo Running examples in directory dolfin/noplot/
echo
echo

echo Running noplot/ex01_show-mesh.py
python noplot/ex01_show-mesh.py

echo Running noplot/ex02_tetralize-mesh.py
python noplot/ex02_tetralize-mesh.py

echo Running noplot/ex03_poisson.py
python noplot/ex03_poisson.py

echo Running noplot/ex04_mixed-poisson.py
python noplot/ex04_mixed-poisson.py

echo Running noplot/ex05_non-matching-meshes.py
python noplot/ex05_non-matching-meshes.py

echo Running noplot/ex06_elasticity1.py
python noplot/ex06_elasticity1.py

echo Running noplot/ex06_elasticity2.py
python noplot/ex06_elasticity2.py

echo Running noplot/ex07_stokes-iterative.py
python noplot/ex07_stokes-iterative.py
