#!/bin/bash
# source run_all.sh
#
printf "\033c"

echo #############################################
echo    Press Esc at anytime to skip example
echo #############################################
echo

echo Running fatlimb.py
python fatlimb.py

echo Running fitplanes.py
python fitplanes.py

echo Running fitspheres1.py
python fitspheres1.py

echo Running quadratic_morphing.py
python quadratic_morphing.py

echo Running moving_least_squares1D.py
python moving_least_squares1D.py

echo Running moving_least_squares2D.py
python moving_least_squares2D.py

echo Running moving_least_squares3D.py
python moving_least_squares3D.py

echo Running recosurface.py
python recosurface.py

echo Running skeletonize.py
python skeletonize.py

echo Running centerline1.py
python centerline1.py

echo Running centerline2.py
python centerline2.py

echo Running mesh_smoothers.py
python mesh_smoothers.py

echo Running interpolateScalar.py
python interpolateScalar.py

echo Running interpolateField.py
python interpolateField.py

echo Running thinplate_morphing1.py
python thinplate_morphing1.py

echo Running thinplate_morphing2.py
python thinplate_morphing2.py

echo Running thinplate_morphing_2d.py
python thinplate_morphing_2d.py

echo Running thinplate_grid.py
python thinplate_grid.py

echo Running meshquality.py
python meshquality.py

echo Running intersect2d.py
python intersect2d.py

echo Running cutWithMesh.py
python cutWithMesh.py

echo Running cutAndCap.py
python cutAndCap.py

echo Running pointsCutMesh1.py
python pointsCutMesh1.py

echo Running geodesic.py
python geodesic.py

echo Running splitmesh.py
python splitmesh.py

echo Running projectsphere.py
python projectsphere.py

echo Running convexHull.py
python convexHull.py

echo Running densifycloud.py
python densifycloud.py
