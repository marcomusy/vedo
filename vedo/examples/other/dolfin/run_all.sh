#!/bin/bash
# source run_all.sh
#
##########################
echo Running ascalarbar.py
python ascalarbar.py

echo Running collisions.py
python collisions.py

echo Running calc_surface_area.py
python calc_surface_area.py

echo Running markmesh.py
python markmesh.py

echo Running scalemesh.py
python scalemesh.py

echo Running pi_estimate.py
python pi_estimate.py

echo Running submesh_boundary.py
python submesh_boundary.py

echo Running demo_submesh.py
python demo_submesh.py

echo Running elastodynamics.py
python elastodynamics.py

echo Running elasticbeam.py
python elasticbeam.py

echo Running magnetostatics.py
python magnetostatics.py

echo Running curl2d.py
python curl2d.py

echo Running pointLoad.py
python pointLoad.py

echo Running meshEditor.py
python meshEditor.py

echo Running nodal_u.py
python nodal_u.py

echo Running simple1Dplot.py
python simple1Dplot.py


######################################
echo Running ex01_show-mesh.py
python ex01_show-mesh.py

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


######################################
echo Running ft02_poisson_membrane.py
python ft02_poisson_membrane.py

echo Running ft04_heat_gaussian.py
python ft04_heat_gaussian.py

echo Running ft07_navier_stokes_channel.py
python ft07_navier_stokes_channel.py

echo Running ft08_navier_stokes_cylinder.py
python ft08_navier_stokes_cylinder.py

echo Running navier-stokes_lshape.py
python navier-stokes_lshape.py

echo Running ft09_reaction_system.py
python ft09_reaction_system.py

echo Running stokes.py
python stokes.py

echo Running stokes2.py
python stokes2.py

echo Running demo_cahn-hilliard.py
python demo_cahn-hilliard.py

echo Running turing_pattern.py
python turing_pattern.py

echo Running heatconv.py
python heatconv.py

echo Running wavy_1d.py
python wavy_1d.py

echo Running awefem.py
python awefem.py

echo Running demo_eigenvalue.py
python demo_eigenvalue.py

echo Running demo_auto-adaptive-poisson.py
python demo_auto-adaptive-poisson.py





