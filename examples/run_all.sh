#!/bin/bash
#

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
echo vedo  /home/musy/Dropbox/Public/vtk_work/vedo_data/2*.vtk
vedo       /home/musy/Dropbox/Public/vtk_work/vedo_data/2*.vtk

echo '----------------------------'
echo vedo  /home/musy/Dropbox/Public/vtk_work/vedo_data/2*.vtk
vedo  -ni -k glossy /home/musy/Dropbox/Public/vtk_work/vedo_data/2*.vtk

echo '----------------------------'
echo vedo  /home/musy/Dropbox/Public/vtk_work/vedo_data/embryo.tif
vedo       /home/musy/Dropbox/Public/vtk_work/vedo_data/embryo.tif

echo '----------------------------'
echo vedo --lego --cmap afmhot_r /home/musy/Dropbox/Public/vtk_work/vedo_data/embryo.tif
vedo      --lego --cmap afmhot_r /home/musy/Dropbox/Public/vtk_work/vedo_data/embryo.tif

echo '----------------------------'
echo vedo -g -c blue /home/musy/Dropbox/Public/vtk_work/vedo_data/embryo.slc
vedo      -g -c blue /home/musy/Dropbox/Public/vtk_work/vedo_data/embryo.slc

echo '----------------------------'
echo vedo --slicer /home/musy/Dropbox/Public/vtk_work/vedo_data/embryo.tif
vedo      --slicer /home/musy/Dropbox/Public/vtk_work/vedo_data/embryo.tif

echo '----------------------------'
echo vedo -s  "/home/musy/Dropbox/Public/vtk_work/vedo_data/2??.vtk"
vedo      -s   /home/musy/Dropbox/Public/vtk_work/vedo_data/2??.vtk

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
