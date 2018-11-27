#!/bin/bash
# source run_all.sh
#
echo Running tutorial.py
python tutorial.py

#################################### basic
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

echo Running basic/delaunay2d.py
python basic/delaunay2d.py

echo Running basic/clustering.py
python basic/clustering.py

echo Running basic/diffusion.py
python basic/diffusion.py

echo Running basic/fitline.py
python basic/fitline.py

echo Running basic/fxy.py
python basic/fxy.py

echo Running basic/icon.py
python basic/icon.py

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

echo Running basic/spring.py
python basic/spring.py

#echo Running basic/manyspheres.py
#python basic/manyspheres.py

echo Running basic/mesh_coloring.py
python basic/mesh_coloring.py

echo Running basic/mesh_alphas.py
python basic/mesh_alphas.py

echo Running basic/mesh_modify.py
python basic/mesh_modify.py

echo Running basic/pca.py
python basic/pca.py

echo Running basic/trail.py
python basic/trail.py

echo
echo Running basic/colorpalette.py
python basic/colorpalette.py

echo
echo Running basic/colorprint.py
python basic/colorprint.py

echo
echo Running basic/colorcubes.py
python basic/colorcubes.py

echo Running basic/largestregion.py
python basic/largestregion.py

echo Running basic/earth.py
python basic/earth.py

echo Running basic/mirror.py
python basic/mirror.py

echo Running basic/sliders.py
python basic/sliders.py

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

echo Running basic/histo2D.py
python basic/histo2D.py



#################################### advanced
echo Running advanced/brownian2D.py
python advanced/brownian2D.py

echo Running advanced/fatlimb.py
python advanced/fatlimb.py

echo Running advanced/fitplanes.py
python advanced/fitplanes.py

echo Running advanced/fitspheres1.py
python advanced/fitspheres1.py

echo Running advanced/gas.py
python advanced/gas.py

echo Running advanced/gyroscope1.py
python advanced/gyroscope1.py

echo Running advanced/gyroscope2.py
python advanced/gyroscope2.py

echo Running advanced/multiple_pendulum.py
python advanced/multiple_pendulum.py

echo Running advanced/quadratic_morphing.py
python advanced/quadratic_morphing.py

echo Running advanced/wave_equation.py
python advanced/wave_equation.py

echo Running advanced/turing.py
python advanced/turing.py

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

echo Running advanced/particle_simulator.py
python advanced/particle_simulator.py

echo Running advanced/doubleslit.py
python advanced/doubleslit.py

echo Running advanced/tunnelling2.py
python advanced/tunnelling2.py

echo Running advanced/blackbody.py
python advanced/blackbody.py

echo Running advanced/interactor.py
python advanced/interactor.py

echo Running advanced/mesh_smoothers.py
python advanced/mesh_smoothers.py


################################### volumetric
echo Running volumetric/readVolumeAsIsoSurface.py
python volumetric/readVolumeAsIsoSurface.py 

echo Running volumetric/readVolume.py
python volumetric/readVolume.py 

echo Running basic/readStructuredPoints.py
python volumetric/readStructuredPoints.py

echo Running basic/probeLine.py
python volumetric/probeLine.py

echo Running basic/probePlane.py
python volumetric/probePlane.py

echo Running basic/imageOperations.py
python volumetric/imageOperations.py

echo Running basic/signedDistance.py
python volumetric/signedDistance.py

echo Running volumetric/read_vti.py
python volumetric/read_vti.py


####################################these may fail
echo Running basic/surfIntersect.py
python basic/surfIntersect.py # fails for vtk version<7

echo Running basic/boolean.py
python basic/boolean.py   # fails for vtk version<7

echo Running advanced/spherical_harmonics1.py
python advanced/spherical_harmonics1.py # fails if sphtool not installed

echo Running advanced/spherical_harmonics2.py
python advanced/spherical_harmonics2.py # fails if sphtool not installed





