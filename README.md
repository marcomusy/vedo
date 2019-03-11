![vlogo](https://user-images.githubusercontent.com/32848391/52522718-50d83880-2c89-11e9-80ff-df1b5618a84a.png)

[![Downloads](https://pepy.tech/badge/vtkplotter)](https://pepy.tech/project/vtkplotter)
[![lics](https://img.shields.io/badge/license-MIT-blue.svg)](https://en.wikipedia.org/wiki/MIT_License)
[![pythvers](https://img.shields.io/badge/python-2.7%7C3.6-brightgreen.svg)](https://pypi.org/project/vtkplotter)
[![gdocs](https://img.shields.io/badge/docs%20by-gendocs-blue.svg)](https://gendocs.readthedocs.io/en/latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2561402.svg)](https://doi.org/10.5281/zenodo.2561402)

A python module for scientific visualization, analysis and animation of 3D objects 
and *point clouds* based on [VTK](https://www.vtk.org/) 
and [numpy](http://www.numpy.org/).<br>

## Download and Install:
Use [pip](https://projects.raspberrypi.org/en/projects/using-pip-on-windows) to install:
```bash
pip install vtkplotter
```

## Documentation
Automatically generated documentation can be found [**here**](https://vtkplotter.embl.es).

## Features

Intuitive and straightforward API which can be combined with VTK seamlessly 
in a program, whilst mantaining access to the full range of VTK native classes.

It includes a [large set of working examples](https://github.com/marcomusy/vtkplotter/tree/master/examples)
for the all following functionalities:

  - Import meshes from VTK format, STL, Wavefront OBJ, 3DS, XML, Neutral, GMSH, OFF, PCD (PointCloud), volumetric TIFF stacks, SLC, MHD, 2D images PNG, JPEG.
  - Export meshes as ASCII or binary to VTK, STL, OBJ, PLY formats.
  - Mesh analysis through the built-in methods of VTK package. Additional analysis tools like *Moving Least Squares*, mesh morphing.
  - Tools to visualize and edit meshes (cutting a mesh with another mesh, slicing, normalizing, moving vertex positions, etc..). Interactive cutter widget.
  - Split mesh based on surface connectivity. Extract the largest connected area.
  - Calculate mass properties, like area, volume, center of mass, average size etc.
  - Calculate vertex and face normals, curvatures, feature edges. Fill mesh holes.
  - Subdivide faces of a mesh, increasing the number of vertex points. Mesh simplification.
  - Coloring and thresholding of meshes based on associated scalar or vectorial data.
  - Point-surface operations: find nearest points, determine if a point lies inside or outside a mesh.
  - Create primitive objects like: spheres, arrows, cubes, torus, ellipsoids... 
  - Generate *glyphs* (associating a mesh to each vertex of a source mesh).
  - Create animations easily by just defining the position of the displayed objects in the 3D scene. Add trailing lines to moving objects automatically.
  - Straightforward support for multiple *sync-ed* or independent renderers in  the same window.
  - Registration (alignment) of meshes with different techniques.
  - Mesh smoothing with *Laplacian* and *WindowedSinc* algorithms.
  - Delaunay triangulation in 2D and 3D.
  - Generate meshes by joining nearby lines in space.
  - Find the closest path from one point to another, travelling along the edges of a mesh.
  - Find the intersection of a mesh with a line (or with another mesh).
  - Analysis of *Point Clouds*:
	 - *Moving Least Squares* smoothing of 2D, 3D and 4D clouds
    - Fit lines, planes and spheres in space
    - Perform PCA (Principal Component Analysis) on point coordinates
    - Identify outliers in a distribution of points
    - Decimate a cloud to a uniform distribution.
  - Basic histogramming and function plotting in 1D and 2D.
  - Interpolate scalar and vectorial fields with *Radial Basis Functions* and *Thin Plate Splines*.
  - Analysis of volumetric datasets:
    - Isosurfacing of volumes
    - Direct maximum projection rendering
    - Generate volumetric signed-distance data from an input surface mesh
    - Probe a volume with lines and planes.
  - Add sliders and buttons to interact with the scene and the individual objects.
  - Examples using [SHTools](https://shtools.oca.eu/shtools) package for *spherical harmonics* expansion of a mesh shape.
  - Integration with the *Qt5* framework.

## Hello World example
In your python script, load a simple `3DS` file and display it:
```python
from vtkplotter import show

show('data/shapes/flamingo.3ds') 
```
![flam](https://user-images.githubusercontent.com/32848391/50738813-58af4380-11d8-11e9-84ce-53579c1dba65.png)


## Command-line usage
```bash
vtkplotter meshfile.vtk 
# valid formats: [vtk,vtu,vts,vtp,ply,obj,stl,3ds,xml,neutral,gmsh,pcd,xyz,txt,byu,tif,off,slc,vti,mhd,png,jpg]
```
to visualize multiple files or files time-sequences try `-n` or `-s` options. Use `-h` for help.<br> 
Voxel-data (_vti, slc, tiff_) files can also be visualized with options `-g` and `--slicer`,
e.g.:
```bash
vtkplotter -g -c blue examples/data/embryo.slc  # (3D scan of a mouse embryo)
vtkplotter --slicer   examples/data/embryo.slc    
```
![e2](https://user-images.githubusercontent.com/32848391/50738810-58af4380-11d8-11e9-8fc7-6c6959207224.jpg)


## Examples Gallery
A get-started tutorial script is available for download:
```bash
git clone https://github.com/marcomusy/vtkplotter.git
cd vtkplotter/examples
python tutorial.py  
```
**More than 100 examples can be found in directories** _(scroll down to see the screenshots):_ <br>
[**examples/basic**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic)<br>
[**examples/advanced**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced)<br>
[**examples/volumetric**](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric)<br>
[**examples/simulations**](https://github.com/marcomusy/vtkplotter/blob/master/examples/simulations)<br>
[**examples/other**](https://github.com/marcomusy/vtkplotter/blob/master/examples/other).<br>

|                                                                                                                   |      |
|:-----------------------------------------------------------------------------------------------------------------:|:-----|
| ![rabbit](https://user-images.githubusercontent.com/32848391/50738808-5816ad00-11d8-11e9-9854-c952be6fb941.jpg)   | Apply a *Moving Least Squares* algorithm to obtain a smooth surface from a to a large cloud of scattered points in space ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/moving_least_squares2D.py)) <br />  `python advanced/moving_least_squares2D.py` |
|                                                                                                                   |      |
| ![gyro](https://user-images.githubusercontent.com/32848391/39766016-85c1c1d6-52e3-11e8-8575-d167b7ce5217.gif)     | Simulation of a gyroscope hanging from a spring ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/simulations/gyroscope1.py)) <br /> `python simulations/gyroscope1.py`|
|                                                                                                                   |      |
| ![ruth](https://user-images.githubusercontent.com/32848391/43984362-5c545a0e-9d00-11e8-8ce5-572b96bb91d1.gif)     | Simulation of [Rutherford scattering](https://en.wikipedia.org/wiki/Rutherford_scattering) of charged particles on a fixed target ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/simulations/particle_simulator.py))<br /> `python simulations/particle_simulator.py`   |
|                                                                                                                   |      |
| ![qsine2](https://user-images.githubusercontent.com/32848391/47751431-06aae880-dc92-11e8-9fcf-6659123edbfa.gif)   | Quantum-tunnelling effect integrating the Schroedinger equation with 4th order Runge-Kutta method. The animation shows the evolution of a particle in a box hitting a sinusoidal potential barrier. ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/simulations/tunnelling2.py)) <br /> `python simulations/tunnelling2.py`   |
|                                                                                                                   |      |
| ![turing](https://user-images.githubusercontent.com/32848391/40665257-1412a30e-635d-11e8-9536-4c73bf6bdd92.gif)   | Visualizing a Turing system of reaction-diffusion between two molecules<sup>1</sup> ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/simulations/turing.py)) <br /> `python simulations/turing.py`  |
<br />


## References
Scientific publications using `vtkplotter` so far:

1. X. Diego _et al._: 
*"Key features of Turing systems are determined purely by network topology"*, 
[Physical Review X, 20 June 2018](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.021071). 
2. M. Musy, K. Flaherty _et al._:
*"A Quantitative Method for Staging Mouse Limb Embryos based on Limb Morphometry"*,
Development, 5 April 2018, [doi: 10.1242/dev.154856](http://dev.biologists.org/content/145/7/dev154856).
3. G. Dalmasso *et al.*, *"Evolution in space and time of 3D volumetric images"*, in preparation.

**Have you found this software useful for your research? Please cite it as:**<br>
M. Musy  _et al._
"`vtkplotter`*, a python module for scientific visualization and analysis of 3D objects 
and point clouds based on VTK*", 
Zenodo, 10 February 2019, [doi: 10.5281/zenodo.2561402](http://doi.org/10.5281/zenodo.2561402).
