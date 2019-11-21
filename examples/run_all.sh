#!/bin/bash
# source run_all.sh
printf "\033c"

echo ###########################################################
echo    Press Esc at anytime to skip example, F1 to interrupt
echo ###########################################################
echo
echo

#################################### tutorial script
#python tutorial.py

#################################### basic
cd basic;       ./run_all.sh; cd ..

#################################### advanced
cd advanced;    ./run_all.sh; cd ..

#################################### simulations
cd simulations; ./run_all.sh; cd ..

################################### volumetric
cd volumetric;  ./run_all.sh; cd ..

#################################### plotting2d
cd plotting2d;  ./run_all.sh; cd ..

#################################### other
cd other;       ./run_all.sh; cd ..

#################################### other/dolfin
if python -c 'import pkgutil; exit(not pkgutil.find_loader("dolfin"))'; then
    cd other/dolfin; ./run_all.sh; cd ../..
else
    echo 'dolfin not found, skip.'
fi

#################################### other/trimesh
if python -c 'import pkgutil; exit(not pkgutil.find_loader("trimesh"))'; then
    cd other/trimesh; ./run_all.sh; cd ../..
else
    echo 'trimesh not found, skip.'
fi


####################################  command line tests
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
echo vtkplotter -s  "data/timecourse1d/*vtk"
vtkplotter      -s   data/timecourse1d/*vtk

echo '----------------------------'
echo vtkplotter -s  "data/2??.vtk"
vtkplotter      -s   data/2??.vtk

echo '----------------------------'
echo vtkplotter -s  "data/images/airplanes_frames/*jpg"
vtkplotter      -s   data/images/airplanes_frames/*jpg

echo '----------------------------'
echo vtkplotter --lego  "data/SainteHelens.dem"
vtkplotter      --lego   data/SainteHelens.dem

echo '---------------------------- should open a GUI'
echo vtkplotter
vtkplotter


##################################### not run/ignored:
# examples/basic/lights.py
# examples/plotting2d/text_just.py
# examples/other/makeVideo.py
# examples/other/spherical_harmonics2.py
# examples/other/remesh_ACVD.py
# examples/other/tf_learn_embryo.py
# examples/other/self_org_maps3d.py
