# source run_all.sh
#
python tutorial.py

python basic/align1.py
python basic/align2.py
python basic/carcrash.py
python basic/colormaps.py
python basic/delaunay2d.py
python basic/clustering.py

python basic/diffusion.py
python basic/fitline.py
python basic/fxy.py
python basic/gyroscope1.py
python basic/gyroscope2.py
python basic/keypress.py
python basic/lights.py
python basic/lorenz.py
python basic/multiwindows.py
python basic/rotateImage.py
python basic/shrink.py
python basic/spring.py
python basic/manyspheres.py
python basic/mesh_coloring.py
python basic/trail.py
python basic/colorpalette.py
python basic/colortext.py
python basic/largestregion.py
python basic/earth.py
python basic/mirror.py
python basic/sliders.py


python advanced/brownian2D.py
python advanced/cell_main.py
python advanced/fatlimb.py
python advanced/fitplanes.py
python advanced/fitspheres1.py
python advanced/fitspheres2.py
python advanced/gas.py
python advanced/multiple_pendulum.py
python advanced/quadratic_morphing.py
python advanced/wave_equation.py
python advanced/turing.py
python advanced/moving_least_squares1D.py
python advanced/moving_least_squares2D.py
python advanced/recosurface.py
python advanced/skeletonize.py
python advanced/particle_simulator.py

#these may fail
python basic/readVolume.py # fails for vtk version<7
python basic/surfIntersect.py # fails for vtk version<7
python basic/boolean.py   # fails for vtk version<7
python advanced/spherical_harmonics1.py # fails if sphtool not installed
python advanced/spherical_harmonics2.py # fails if sphtool not installed
