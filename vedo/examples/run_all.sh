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

cd tetmesh;     ./run_all.sh; cd ..

cd pyplot;      ./run_all.sh; cd ..

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
echo '---------------------------- command line tests'
echo vedo  data/2*.vtk
vedo       data/2*.vtk

echo '----------------------------'
echo vedo  data/2*.vtk
vedo  -ni  data/2*.vtk

echo '----------------------------'
echo vedo  data/embryo.tif
vedo       data/embryo.tif

echo '----------------------------'
echo vedo --lego --cmap afmhot_r data/embryo.tif
vedo      --lego --cmap afmhot_r data/embryo.tif

echo '----------------------------'
echo vedo -g -c blue data/embryo.slc
vedo      -g -c blue data/embryo.slc

echo '----------------------------'
echo vedo --slicer data/embryo.tif
vedo      --slicer data/embryo.tif

echo '----------------------------'
echo vedo -s  "data/2??.vtk"
vedo      -s   data/2??.vtk

echo '---------------------------- should open a GUI'
echo vedo
vedo

##################################### not run/ignored:
# python basic/closewindow.py
# python basic/lights.py
# python basic/multiblocks.py
# python other/animation1.py
# python other/animation2.py
# python other/pygmsh_extrude.py
# python other/voronoi3d.py
# python other/makeVideo.py
# python other/spherical_harmonics2.py
