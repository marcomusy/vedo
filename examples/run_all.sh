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

echo Running basic/buildmesh.py
python  basic/buildmesh.py

#################################### 
echo Running basic/align1.py
python basic/align1.py

echo Running basic/align2.py
python basic/align2.py

echo Running basic/align3.py
python basic/align3.py

echo Running basic/bgImage.py
python basic/bgImage.py

echo Running basic/boolean.py
python basic/boolean.py      

echo Running basic/carcrash.py
python basic/carcrash.py

echo Running basic/colormaps.py
python basic/colormaps.py

echo Running basic/colorMeshCells.py
python basic/colorMeshCells.py

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

echo Running basic/deleteMeshPoints.py
python basic/deleteMeshPoints.py

echo Running basic/extrude.py
python basic/extrude.py

echo Running basic/cellsWithinBounds.py
python basic/cellsWithinBounds.py

echo Running basic/fitline.py
python basic/fitline.py

echo Running basic/fxy.py
python basic/fxy.py

echo Running basic/keypress.py
python basic/keypress.py

echo Running basic/kspline.py
python basic/kspline.py

echo Running basic/linInterpolate.py
python basic/linInterpolate.py

echo Running basic/lorenz.py
python basic/lorenz.py

echo Running basic/multiwindows.py
python basic/multiwindows.py

echo Running basic/rotateImage.py
python basic/rotateImage.py

echo Running basic/shrink.py
python basic/shrink.py

echo Running basic/manypoints.py
python basic/manypoints.py   

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

echo Running basic/mesh_map2cell.py
python basic/mesh_map2cell.py

echo Running basic/isolines.py
python basic/isolines.py

echo Running basic/pca.py
python basic/pca.py

echo Running basic/silhouette.py
python basic/silhouette.py

echo Running basic/silhouette2.py
python basic/silhouette2.py

echo Running basic/trail.py
python basic/trail.py

echo Running basic/colorcubes.py
python basic/colorcubes.py

echo Running basic/largestregion.py
python basic/largestregion.py

echo Running basic/mirror.py
python basic/mirror.py

echo Running basic/scalarbars.py
python basic/scalarbars.py

echo Running basic/sliders.py
python basic/sliders.py

echo Running basic/slider_browser.py
python advanced/slider_browser.py

echo Running basic/sliders3d.py
python basic/sliders3d.py   

echo Running basic/buttons.py
python basic/buttons.py

echo Running basic/cutter.py
python basic/cutter.py

echo Running basic/texturecubes.py
python basic/texturecubes.py

echo Running basic/mouseclick.py
python basic/mouseclick.py

echo Running basic/ribbon.py
python basic/ribbon.py

echo Running basic/flatarrow.py
python basic/flatarrow.py

echo Running basic/fillholes.py
python basic/fillholes.py

echo Running basic/interactionstyle.py
python basic/interactionstyle.py

echo Running basic/tube.py
python basic/tube.py

echo Running basic/fonts.py
python basic/fonts.py    

echo Running basic/glyphs.py
python basic/glyphs.py   

echo Running basic/glyphs_arrows.py
python basic/glyphs_arrows.py   

echo Running basic/shadow.py
python basic/shadow.py   

echo Running basic/specular.py
python basic/specular.py   

echo Running basic/lightings.py
python basic/lightings.py   

echo Running basic/surfIntersect.py
python basic/surfIntersect.py   



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

echo Running advanced/thinplate_morphing_2d.py
python advanced/thinplate_morphing_2d.py

echo Running advanced/meshquality.py
python advanced/meshquality.py

echo Running advanced/cutWithMesh.py
python advanced/cutWithMesh.py

echo Running advanced/cutAndCap.py
python advanced/cutAndCap.py

echo Running advanced/pointsCutMesh1.py
python advanced/pointsCutMesh1.py

echo Running advanced/geodesic.py
python advanced/geodesic.py

echo Running advanced/splitmesh.py
python advanced/splitmesh.py

echo Running advanced/projectsphere.py
python advanced/projectsphere.py

echo Running advanced/convexHull.py
python advanced/convexHull.py

echo Running advanced/densifycloud.py
python advanced/densifycloud.py

################################### volumetric
echo Running volumetric/readVolumeAsIsoSurface.py
python volumetric/readVolumeAsIsoSurface.py 

echo Running volumetric/readVolume.py
python volumetric/readVolume.py 

echo Running volumetric/probePoints.py
python volumetric/probePoints.py

echo Running volumetric/probeLine.py
python volumetric/probeLine.py

echo Running volumetric/probePlane.py
python volumetric/probePlane.py

echo Running volumetric/volumeOperations.py
python volumetric/volumeOperations.py

echo Running volumetric/volumeFromMesh.py
python volumetric/volumeFromMesh.py

echo Running volumetric/read_vti.py
python volumetric/read_vti.py

echo Running volumetric/interpolateVolume.py
python volumetric/interpolateVolume.py

echo Running volumetric/isosurfaces1.py
python volumetric/isosurfaces1.py

echo Running volumetric/isosurfaces2.py
python volumetric/isosurfaces2.py

echo Running volumetric/legosurface.py
python volumetric/legosurface.py

echo Running volumetric/mesh2volume.py
python volumetric/mesh2volume.py

echo Running volumetric/streamlines1.py
python volumetric/streamlines1.py

echo Running volumetric/streamlines2.py
python volumetric/streamlines2.py

echo Running volumetric/streamribbons.py
python volumetric/streamribbons.py

echo Running volumetric/lowpassfilter.py
python volumetric/lowpassfilter.py

echo Running volumetric/numpy2volume.py
python volumetric/numpy2volume.py

echo Running volumetric/tensors.py
python volumetric/tensors.py

echo Running volumetric/pointDensity.py
python volumetric/pointDensity.py

echo Running volumetric/erode_dilate.py
python volumetric/erode_dilate.py

echo Running volumetric/euclDist.py
python volumetric/euclDist.py

echo Running volumetric/vol2points.py
python volumetric/vol2points.py

echo Running volumetric/tet_mesh_ugrid.py
python volumetric/tet_mesh_ugrid.py

cd volumetric
echo Running office.py
python office.py
cd ..


#################################### plotting2d
echo Running plotting2d/annotations.py
python plotting2d/annotations.py    

echo Running plotting2d/customAxes.py
python plotting2d/customAxes.py

echo Running plotting2d/histogram.py
python plotting2d/histogram.py

echo Running plotting2d/histoHexagonal.py
python plotting2d/histoHexagonal.py

echo Running plotting2d/donutPlot.py
python plotting2d/donutPlot.py

echo Running plotting2d/latex.py
python plotting2d/latex.py

echo Running plotting2d/markers.py
python plotting2d/markers.py

echo Running plotting2d/markpoint.py
python plotting2d/markpoint.py    

echo Running plotting2d/numpy2picture.py
python plotting2d/numpy2picture.py

echo Running plotting2d/plotxy.py
python plotting2d/plotxy.py

echo Running plotting2d/polarPlot.py
python plotting2d/polarPlot.py

echo Running plotting2d/polarHisto.py
python plotting2d/polarHisto.py


#################################### Other
echo Running other/colorpalette.py
python other/colorpalette.py

echo Running other/printc.py
python other/printc.py

echo Running other/icon.py
python other/icon.py

echo Running other/inset.py
python other/inset.py

echo Running other/qt_window.py # needs qt5
python other/qt_window.py

echo Running other/qt_window_split.py # needs qt5
python other/qt_window_split.py

echo Running other/qt_tabs.py # needs qt5
python other/qt_tabs.py

echo Running other/self_org_maps2d.py
python other/self_org_maps2d.py

echo Running other/value-iteration.py
python other/value-iteration.py

echo Running other/spherical_harmonics1.py
python other/spherical_harmonics1.py 

echo Running other/tf_learn_volume.py
python other/tf_learn_volume.py

echo Running other/voronoi3d.py
python other/voronoi3d.py

echo Running other/export_x3d.py
python other/export_x3d.py

echo Running other/create_logo.py
python other/create_logo.py

echo Running other/export_numpy.py
python other/export_numpy.py

echo Running other/save_as_numpy.py
python other/save_as_numpy.py


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
echo vtkplotter --lego --cmap afmhot_r  data/embryo.tif
vtkplotter      --lego --cmap afmhot_r  data/embryo.tif

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

echo '----------------------------'
echo '----------------------------'
echo 'cd simulations;  ./run_all.sh'
echo '----------------------------'
echo '----------------------------'


##################################### not run/ignored:
#plotting2d/text_just.py
#basic/lights.py
#other/makeVideo.py
#other/spherical_harmonics2.py 
#other/remesh_ACVD.py 
#other/tf_learn_embryo.py
#other/self_org_maps3d.py
