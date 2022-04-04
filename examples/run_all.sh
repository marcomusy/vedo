#!/bin/bash
#

cd basic;       ./run_all.sh; cd ..

cd advanced;    ./run_all.sh; cd ..

cd simulations; ./run_all.sh; cd ..

cd volumetric;  ./run_all.sh; cd ..

cd pyplot;      ./run_all.sh; cd ..

cd other;       ./run_all.sh; cd ..

# other/dolfin
if python3 -c 'import pkgutil; exit(not pkgutil.find_loader("dolfin"))'; then
    cd other/dolfin; ./run_all.sh; cd ../..
else
    echo 'dolfin not found, skip.'
fi

# other/trimesh
if python3 -c 'import pkgutil; exit(not pkgutil.find_loader("trimesh"))'; then
    cd other/trimesh; ./run_all.sh; cd ../..
else
    echo 'trimesh not found, skip.'
fi

#################################  command line tests
echo '---------------------------- command line tests'
echo vedo  ../data/2*.vtk
vedo       ../data/2*.vtk

echo '----------------------------'
echo vedo  ../data/2*.vtk
vedo  -ni -k glossy data/2*.vtk

echo '----------------------------'
echo vedo -s  "../data/2??.vtk"
vedo      -s   ../data/2??.vtk

echo '----------------------------'
echo vedo  ../data/embryo.tif
vedo       ../data/embryo.tif

echo '----------------------------'
echo vedo --lego --cmap afmhot_r ../data/embryo.tif
vedo      --lego --cmap afmhot_r ../data/embryo.tif

echo '----------------------------'
echo vedo -g -c blue ../data/embryo.slc
vedo      -g -c blue ../data/embryo.slc

echo '----------------------------'
echo vedo --slicer2d ../data/embryo.tif
vedo      --slicer2d ../data/embryo.tif

echo '----------------------------'
echo vedo --slicer3d ../data/embryo.tif
vedo      --slicer3d ../data/embryo.tif

echo '----------------------------'
echo vedo --eog ../data/Wnt5a.jpg
vedo      --eog ../data/Wnt5a.jpg

echo '---------------------------- should open a GUI'
echo vedo
vedo

