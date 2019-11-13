#!/bin/bash
# source run_all.sh
#
printf "\033c"

echo #############################################
echo    Press Esc at anytime to skip example
echo #############################################
echo
echo

echo Running readVolumeAsIsoSurface.py
python readVolumeAsIsoSurface.py 

echo Running readVolume.py
python readVolume.py 

echo Running probePoints.py
python probePoints.py

echo Running probeLine.py
python probeLine.py

echo Running probePlane.py
python probePlane.py

echo Running volumeOperations.py
python volumeOperations.py

echo Running volumeFromMesh.py
python volumeFromMesh.py

echo Running read_vti.py
python read_vti.py

echo Running interpolateVolume.py
python interpolateVolume.py

echo Running isosurfaces1.py
python isosurfaces1.py

echo Running isosurfaces2.py
python isosurfaces2.py

echo Running legosurface.py
python legosurface.py

echo Running mesh2volume.py
python mesh2volume.py

echo Running streamlines1.py
python streamlines1.py

echo Running streamlines2.py
python streamlines2.py

echo Running streamribbons.py
python streamribbons.py

echo Running lowpassfilter.py
python lowpassfilter.py

echo Running numpy2volume.py
python numpy2volume.py

echo Running numpy2volume2.py
python numpy2volume2.py

echo Running tensors.py
python tensors.py

echo Running tensor_grid.py
python tensor_grid.py

echo Running pointDensity.py
python pointDensity.py

echo Running erode_dilate.py
python erode_dilate.py

echo Running euclDist.py
python euclDist.py

echo Running vol2points.py
python vol2points.py

echo Running tet_mesh_ugrid.py
python tet_mesh_ugrid.py

