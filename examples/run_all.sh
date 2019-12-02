#!/bin/bash
#
printf "\033c"

echo ###########################################################
echo    Press Esc at anytime to skip example, F1 to interrupt
echo ###########################################################
echo
echo

#python tutorial.py

cd basic;       ./run_all.sh; cd ..

cd advanced;    ./run_all.sh; cd ..

cd simulations; ./run_all.sh; cd ..

cd volumetric;  ./run_all.sh; cd ..

cd plotting2d;  ./run_all.sh; cd ..

cd other;       ./run_all.sh; cd ..

# other/dolfin
if python -c 'import pkgutil; exit(not pkgutil.find_loader("dolfin"))'; then
    cd other/dolfin; ./run_all.sh; cd ../..
else
    echo 'dolfin not found, skip.'
fi

# other/trimesh
if python -c 'import pkgutil; exit(not pkgutil.find_loader("trimesh"))'; then
    cd other/trimesh; ./run_all.sh; cd ../..
else
    echo 'trimesh not found, skip.'
fi

#################################  command line tests
printf "\033c"
echo '---------------------------- command line tests'
echo vtkplotter  data/2*.vtk
vtkplotter       data/2*.vtk

echo '----------------------------'
echo vtkplotter  data/embryo.tif
vtkplotter       data/embryo.tif

echo '----------------------------'
echo vtkplotter --lego --cmap afmhot_r data/embryo.tif
vtkplotter      --lego --cmap afmhot_r data/embryo.tif

echo '----------------------------'
echo vtkplotter -g -c blue data/embryo.slc
vtkplotter      -g -c blue data/embryo.slc

echo '----------------------------'
echo vtkplotter --slicer data/embryo.slc
vtkplotter      --slicer data/embryo.slc

echo '----------------------------'
echo vtkplotter -s  "data/2??.vtk"
vtkplotter      -s   data/2??.vtk

echo '----------------------------'
echo vtkplotter -s  "data/images/airplanes_frames/*jpg"
vtkplotter      -s   data/images/airplanes_frames/*jpg

echo '---------------------------- should open a GUI'
echo vtkplotter
vtkplotter

##################################### not run/ignored:
# python basic/closewindow.py
# python basic/lights.py
# python basic/multiblocks.py
# python basic/kspline.py
# python advanced/pointsCutMesh2.py
# python simulations/tunnelling1.py
# python plotting2d/text_just.py
# python other/animation1.py
# python other/animation2.py
# python other/qt_tabs_ui.py
# python other/remesh_meshfix.py
# python other/pygmsh_extrude.py
# python other/voronoi3d.py
# python other/makeVideo.py
# python other/spherical_harmonics2.py
# python other/remesh_ACVD.py
# python other/tf_learn_embryo.py
# python other/self_org_maps3d.py
