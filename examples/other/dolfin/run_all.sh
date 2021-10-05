#!/bin/bash
# source run_all.sh
#
##########################
echo Running ascalarbar.py
python3 ascalarbar.py

echo Running collisions.py
python3 collisions.py

echo Running calc_surface_area.py
python3 calc_surface_area.py

echo Running markmesh.py
python3 markmesh.py

echo Running scalemesh.py
python3 scalemesh.py

echo Running pi_estimate.py
python3 pi_estimate.py

echo Running submesh_boundary.py
python3 submesh_boundary.py

echo Running demo_submesh.py
python3 demo_submesh.py

echo Running elastodynamics.py
python3 elastodynamics.py

echo Running elasticbeam.py
python3 elasticbeam.py

echo Running magnetostatics.py
python3 magnetostatics.py

echo Running curl2d.py
python3 curl2d.py

echo Running pointLoad.py
python3 pointLoad.py

echo Running meshEditor.py
python3 meshEditor.py

echo Running nodal_u.py
python3 nodal_u.py

echo Running simple1Dplot.py
python3 simple1Dplot.py


######################################
echo Running ex01_show-mesh.py
python3 ex01_show-mesh.py

echo Running ex03_poisson.py
python3 ex03_poisson.py

echo Running ex04_mixed-poisson.py
python3 ex04_mixed-poisson.py

echo Running ex05_non-matching-meshes.py
python3 ex05_non-matching-meshes.py

echo Running ex06_elasticity1.py
python3 ex06_elasticity1.py

echo Running ex06_elasticity2.py
python3 ex06_elasticity2.py

echo Running ex07_stokes-iterative.py
python3 ex07_stokes-iterative.py


######################################
echo Running ft02_poisson_membrane.py
python3 ft02_poisson_membrane.py

echo Running ft04_heat_gaussian.py
python3 ft04_heat_gaussian.py

echo Running ft07_navier_stokes_channel.py
python3 ft07_navier_stokes_channel.py

echo Running ft08_navier_stokes_cylinder.py
python3 ft08_navier_stokes_cylinder.py

echo Running navier-stokes_lshape.py
python3 navier-stokes_lshape.py

echo Running ft09_reaction_system.py
python3 ft09_reaction_system.py

echo Running stokes.py
python3 stokes.py

echo Running stokes2.py
python3 stokes2.py

echo Running demo_cahn-hilliard.py
python3 demo_cahn-hilliard.py

echo Running turing_pattern.py
python3 turing_pattern.py

echo Running heatconv.py
python3 heatconv.py

echo Running wavy_1d.py
python3 wavy_1d.py

echo Running awefem.py
python3 awefem.py

echo Running demo_eigenvalue.py
python3 demo_eigenvalue.py

echo Running demo_auto-adaptive-poisson.py
python3 demo_auto-adaptive-poisson.py





