#!/bin/bash
# source run_all.sh
#
printf "\033c"

echo #############################################
echo    Press Esc at anytime to skip example
echo #############################################
echo
echo
echo

#################################### basic
echo Running tutorial.py
python tutorial.py

echo Running basic/a_first_example.py
python  basic/a_first_example.py

echo Running basic/acollection.py
python  basic/acollection.py

#################################### 
echo Running basic/align1.py
python basic/align1.py

echo Running basic/align2.py
python basic/align2.py

echo Running basic/align3.py
python basic/align3.py

echo Running basic/carcrash.py
python basic/carcrash.py

echo Running basic/colormaps.py
python basic/colormaps.py

echo Running basic/buildpolydata.py
python basic/buildpolydata.py

echo Running basic/delaunay2d.py
python basic/delaunay2d.py

echo Running basic/distance2mesh.py
python basic/distance2mesh.py

echo Running basic/clustering.py
python basic/clustering.py

echo Running basic/connVtx.py
python basic/connVtx.py

echo Running basic/connCells.py
python basic/connCells.py

echo Running basic/fitline.py
python basic/fitline.py

echo Running basic/fxy.py
python basic/fxy.py

echo Running basic/keypress.py
python basic/keypress.py

echo Running basic/lorenz.py
python basic/lorenz.py

echo Running basic/multiwindows.py
python basic/multiwindows.py

echo Running basic/rotateImage.py
python basic/rotateImage.py

echo Running basic/shrink.py
python basic/shrink.py

echo Running basic/manyspheres.py
python basic/manyspheres.py

echo Running basic/mesh_coloring.py
python basic/mesh_coloring.py

echo Running basic/mesh_custom.py
python basic/mesh_custom.py

echo Running basic/mesh_bands.py
python basic/mesh_bands.py

echo Running basic/mesh_alphas.py
python basic/mesh_alphas.py

echo Running basic/mesh_sharemap.py
python basic/mesh_sharemap.py

echo Running basic/mesh_threshold.py
python basic/mesh_threshold.py

echo Running basic/mesh_modify.py
python basic/mesh_modify.py

echo Running basic/pca.py
python basic/pca.py

echo Running basic/trail.py
python basic/trail.py

echo Running basic/colorcubes.py
python basic/colorcubes.py

echo Running basic/largestregion.py
python basic/largestregion.py

echo Running basic/mirror.py
python basic/mirror.py

echo Running basic/sliders.py
python basic/sliders.py

echo Running basic/sliders3d.py
python basic/sliders3d.py   

echo Running basic/buttons.py
python basic/buttons.py

echo Running basic/cutter.py
python basic/cutter.py

echo Running basic/texturecubes.py
python basic/texturecubes.py

echo Running basic/bgImage.py
python basic/bgImage.py

echo Running basic/mouseclick.py
python basic/mouseclick.py

echo Running basic/ribbon.py
python basic/ribbon.py

echo Running basic/flatarrow.py
python basic/flatarrow.py

echo Running basic/histo2D.py
python basic/histo2D.py

echo Running basic/fillholes.py
python basic/fillholes.py

echo Running basic/interactionstyle.py
python basic/interactionstyle.py

echo Running basic/tube.py
python basic/tube.py

echo Running basic/boolean.py
python basic/boolean.py       # fails for vtk version<7

echo Running basic/annotations.py
python basic/annotations.py    

echo Running basic/markpoint.py
python basic/markpoint.py    

echo Running basic/glyphs.py
python basic/glyphs.py   


#################################### advanced
echo Running advanced/fatlimb.py
python advanced/fatlimb.py

echo Running advanced/fitplanes.py
python advanced/fitplanes.py

echo Running advanced/fitspheres1.py
python advanced/fitspheres1.py

echo Running advanced/quadratic_morphing.py
python advanced/quadratic_morphing.py

echo Running advanced/moving_least_squares1D.py
python advanced/moving_least_squares1D.py

echo Running advanced/moving_least_squares2D.py
python advanced/moving_least_squares2D.py

echo Running advanced/moving_least_squares3D.py
python advanced/moving_least_squares3D.py

echo Running advanced/recosurface.py
python advanced/recosurface.py

echo Running advanced/skeletonize.py
python advanced/skeletonize.py

echo Running advanced/mesh_smoothers.py
python advanced/mesh_smoothers.py

echo Running advanced/interpolateScalar.py
python advanced/interpolateScalar.py

echo Running advanced/interpolateField.py
python advanced/interpolateField.py

echo Running advanced/thinplate.py
python advanced/thinplate.py

echo Running advanced/thinplate_grid.py
python advanced/thinplate_grid.py

echo Running advanced/thinplate_morphing.py
python advanced/thinplate_morphing.py

echo Running advanced/meshquality.py
python advanced/meshquality.py

echo Running advanced/cutWithMesh.py
python advanced/cutWithMesh.py

echo Running advanced/cutAndCap.py
python advanced/cutAndCap.py

echo Running advanced/geodesic.py
python advanced/geodesic.py

echo Running advanced/splitmesh.py
python advanced/splitmesh.py

echo Running advanced/projectsphere.py
python advanced/projectsphere.py

echo Running advanced/convexHull.py
python advanced/convexHull.py


################################### simulations
echo Running simulations/aspring.py
python simulations/aspring.py

echo Running simulations/brownian2D.py
python simulations/brownian2D.py

echo Running simulations/gas.py
python simulations/gas.py

echo Running simulations/gyroscope1.py
python simulations/gyroscope1.py

echo Running simulations/gyroscope2.py
python simulations/gyroscope2.py

echo Running simulations/multiple_pendulum.py
python simulations/multiple_pendulum.py

echo Running simulations/pendulum.py
python simulations/pendulum.py

echo Running simulations/wave_equation.py
python simulations/wave_equation.py

echo Running simulations/turing.py
python simulations/turing.py

echo Running simulations/particle_simulator.py
python simulations/particle_simulator.py

echo Running simulations/doubleslit.py
python simulations/doubleslit.py

echo Running simulations/tunnelling2.py
python simulations/tunnelling2.py


################################### volumetric
echo Running volumetric/readVolumeAsIsoSurface.py
python volumetric/readVolumeAsIsoSurface.py 

echo Running volumetric/readVolume.py
python volumetric/readVolume.py 

echo Running volumetric/readStructuredPoints.py
python volumetric/readStructuredPoints.py

echo Running volumetric/probeLine.py
python volumetric/probeLine.py

echo Running volumetric/probePlane.py
python volumetric/probePlane.py

echo Running volumetric/imageOperations.py
python volumetric/imageOperations.py

echo Running volumetric/signedDistance.py
python volumetric/signedDistance.py

echo Running volumetric/read_vti.py
python volumetric/read_vti.py

echo Running volumetric/interpolateVolume.py
python volumetric/interpolateVolume.py

echo Running volumetric/isosurfaces1.py
python volumetric/isosurfaces1.py

echo Running volumetric/isosurfaces2.py
python volumetric/isosurfaces2.py

echo Running volumetric/mesh2volume.py
python volumetric/mesh2volume.py


#################################### Other
echo Running other/colorpalette.py
python other/colorpalette.py

echo Running other/printc.py
python other/printc.py

echo Running other/icon.py
python other/icon.py

echo Running other/qt_embed.py # needs qt5
python other/qt_embed.py

echo Running other/spherical_harmonics1.py
python other/spherical_harmonics1.py 


##################################### not ran/ignored:
#basic/text_just.py
#basic/lights.py
#basic/ids.py
#basic/surfIntersect.py
#other/makeVideo.py
#other/spherical_harmonics2.py 

#################################### 
echo
echo
echo '---------------------------- command lines'
echo vtkplotter  data/2*.vtk
vtkplotter       data/2*.vtk

echo '----------------------------'
echo vtkplotter  data/embryo.tif
vtkplotter       data/embryo.tif

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


