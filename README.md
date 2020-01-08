![vlogo](https://user-images.githubusercontent.com/32848391/52522718-50d83880-2c89-11e9-80ff-df1b5618a84a.png)

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/e8c5f1f9afb6433a9cdf4edb5499bd46)](https://app.codacy.com/app/marcomusy/vtkplotter-examples?utm_source=github.com&utm_medium=referral&utm_content=marcomusy/vtkplotter-examples&utm_campaign=Badge_Grade_Dashboard)
[![Downloads](https://pepy.tech/badge/vtkplotter)](https://pepy.tech/project/vtkplotter)
[![lics](https://img.shields.io/badge/license-MIT-blue.svg)](https://en.wikipedia.org/wiki/MIT_License)
[![pythvers](https://img.shields.io/badge/python-2.7%7C3-brightgreen.svg)](https://pypi.org/project/vtkplotter)
[![gdocs](https://img.shields.io/badge/docs%20by-gendocs-blue.svg)](https://gendocs.readthedocs.io/en/latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2561402.svg)](https://doi.org/10.5281/zenodo.2561402)
[![CircleCI](https://circleci.com/gh/marcomusy/vtkplotter.svg?style=svg)](https://circleci.com/gh/marcomusy/vtkplotter)

A lightweight pure python module for scientific analysis and visualization of 3D objects
and *point clouds* based on [VTK](https://www.vtk.org/) and [numpy](http://www.numpy.org/).<br>


## ✨ Philosophy
Inspired by the [vpython](https://vpython.org/) *manifesto* "3D programming for ordinary mortals",
*vtkplotter* makes it easy to work wth three-dimensional objects, create displays and animations
in just a few lines of code, even for those with less programming experience.

Because life is short.

## 🎯 Table of Contents
* [Installation](https://github.com/marcomusy/vtkplotter#-installation)
* [Documentation](https://github.com/marcomusy/vtkplotter#-documentation)
	* [Need help?](https://github.com/marcomusy/vtkplotter#-need-help)
* [Features](https://github.com/marcomusy/vtkplotter#-features)
  * [Command Line Interface](https://github.com/marcomusy/vtkplotter#command-line-interface)
  * [Graphic User Interface](https://github.com/marcomusy/vtkplotter#graphic-user-interface)
* [Examples](https://github.com/marcomusy/vtkplotter#-examples)
* [References](https://github.com/marcomusy/vtkplotter#-references)



## 💾 Installation
Use [pip](https://projects.raspberrypi.org/en/projects/using-pip-on-windows) to install:
```bash
pip install -U vtkplotter
```
*Windows-10 users* can place this file
[vtkplotter.bat](https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter.bat)
on the desktop to *drag&drop* files to visualize.
(Need to edit the path of the local Anaconda installation).


## 📙 Documentation
Automatically generated documentation can be found [**here**](https://vtkplotter.embl.es).


#### 📌 Need help?
Have any question, or wish to suggest or ask for a missing feature?
Do not hesitate to open a [**issue**](https://github.com/marcomusy/vtkplotter-examples/issues)
or send an [email](mailto:marco.musy@embl.es).


## 🎨 Features
Intuitive and straightforward API which can be combined with VTK seamlessly
in a program, whilst mantaining access to the full range of VTK native classes.

It includes a [**large set of working examples**](https://github.com/marcomusy/vtkplotter-examples/tree/master/vtkplotter_examples)
for all the following functionalities:

  - Import meshes from VTK format, STL, Wavefront OBJ, 3DS, Dolfin-XML, Neutral, GMSH, OFF, PCD (PointCloud), volumetric TIFF stacks, DICOM, SLC, MHD, 2D images PNG, JPEG.
  - Export meshes as ASCII or binary to VTK, STL, OBJ, PLY formats.
  - Mesh analysis through the built-in methods of VTK. Additional analysis tools like *Moving Least Squares*, mesh morphing and more..
  - Tools to visualize and edit meshes (cutting a mesh with another mesh, slicing, normalizing, moving vertex positions, etc..).
  - Split mesh based on surface connectivity. Extract the largest connected area.
  - Calculate areas, volumes, center of mass, average sizes etc.
  - Calculate vertex and face normals, curvatures, feature edges. Fill mesh holes.
  - Subdivide faces of a mesh, increasing the number of vertex points. Mesh simplification.
  - Coloring and thresholding of meshes based on associated scalar or vectorial data.
  - Point-surface operations: find nearest points, determine if a point lies inside or outside of a mesh.
  - Create primitive shapes: spheres, arrows, cubes, torus, ellipsoids...
  - Generate *glyphs* (associate a mesh to every vertex of a source mesh).
  - Create animations easily by just setting the position of the displayed objects in the 3D scene. Add trailing lines and shadows to moving objects is supported.
  - Straightforward support for multiple *sync-ed* or independent renderers in  the same window.
  - Registration (alignment) of meshes with different techniques.
  - Mesh smoothing.
  - Delaunay triangulation in 2D and 3D.
  - Generate meshes by joining nearby lines in space.
  - Find the closest path from one point to another, travelling along the edges of a mesh.
  - Find the intersection of a mesh with lines, planes or other meshes.
  - Analysis of *Point Clouds*:
	 - *Moving Least Squares* smoothing of 2D, 3D and 4D clouds
    - Fit lines, planes, spheres and ellipses in space
    - Identify outliers in a distribution of points
    - Decimate a cloud to a uniform distribution.
  - Histogramming and function plotting in 1D and 2D.
  - Interpolate scalar and vectorial fields with *Radial Basis Functions* and *Thin Plate Splines*.
  - Analysis of volumetric datasets:
    - Isosurfacing of volumes
    - Composite and maximum projection volumetric rendering
    - Generate volumetric signed-distance data from an input surface mesh
    - Probe a volume with lines and planes
    - Generate stream-lines and stream-tubes from vectorial fields
  - Add sliders and buttons to interact with the scene and the individual objects.
  - Fully customizable axis style.
  - Visualization of tensors.
  - Draw `latex`-formatted formulas in the rending window.
  - Integration with the *Qt5* framework.
  - Examples using [SHTools](https://shtools.oca.eu/shtools) package for *spherical harmonics* expansion of a mesh shape.
  - Support for [FEniCS/Dolfin](https://fenicsproject.org/) platform for visualization of finite-element calculations.
  - Interoperability with the [trimesh](https://trimsh.org/) library.
  - Export a 3D scene and embed it into a [web page](https://vtkplotter.embl.es/examples/fenics_elasticity.html).
  - Embed the 3D rendering in a *jupyter* notebook with [K3D](https://github.com/K3D-tools/K3D-jupyter) (can export an interactive 3D-snapshot page [here](https://vtkplotter.embl.es/examples/K3D_snapshot.html)).


### ➜ Command Line Interface
Visualize a mesh from a terminal window with:
```bash
vtkplotter mesh.obj
# valid formats: [vtk,vtu,vts,vtp,vtm,ply,obj,stl,3ds,dolfin-xml,neutral,gmsh,
#                 pcd,xyz,txt,byu,tif,off,slc,vti,mhd,dcm,dem,nrrd,nii,bmp,png,jpg]
```
Voxel-data (_mhd, vti, slc, tiff, dicom etc.._) files can be visualized with options `-g`. E.g.:<br>
`vtkplotter -g examples/data/embryo.slc`<br>

![isohead](https://user-images.githubusercontent.com/32848391/58336107-5a09a180-7e43-11e9-8c4e-b50e4e95ae71.gif)

To visualize multiple files or files time-sequences try `-n` or `-s` options. Use `-h` for the complete list of options.

| Use a slider to control isosurfacing of a volume:|  Load and browse a sequence of meshes:| Slice a 3D volume with a plane:| Visualize colorized voxels:|
|:--------|:-----|:----|:----|
|`vtkplotter head.vti` |`vtkplotter -s *.vtk` |`vtkplotter `<br>`--slicer embr.slc` |   `vtkplotter --lego embryo.slc`|
|![isohead](https://user-images.githubusercontent.com/32848391/56972083-a7f3f800-6b6a-11e9-9cb3-1047b69dcad2.gif)|   ![viz_raycast](https://user-images.githubusercontent.com/32848391/58336919-f7b1a080-7e44-11e9-9106-f574371093a8.gif)  | ![viz_slicer](https://user-images.githubusercontent.com/32848391/56972084-a7f3f800-6b6a-11e9-98c4-dc4ffec70a5e.png)  |![lego](https://user-images.githubusercontent.com/32848391/56969949-71b47980-6b66-11e9-8251-4bbdb275cb22.jpg) |

### ➜ Graphic User Interface
A Graphic User Interface is available (mainly useful to *Windows 10* users):

![gui](https://user-images.githubusercontent.com/32848391/63259840-c861d280-c27f-11e9-9c2a-99d0fae85313.png)

## 🐾 Examples
Run any of the available scripts from the [vtkplotter-examples](https://github.com/marcomusy/vtkplotter-examples) module with:
```bash
pip install -U git+https://github.com/marcomusy/vtkplotter-examples
vtkplotter --list
vtkplotter --run tube.py
```
**More than 280 working examples can be found in directories** _(scroll down to see thumbnails):_ <br>
[**examples/basic**](https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/basic)<br>
[**examples/advanced**](https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/advanced)<br>
[**examples/volumetric**](https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/volumetric)<br>
[**examples/simulations**](https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/simulations)<br>
[**examples/plotting2d**](https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/plotting2d)<br>
[**examples/other**](https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/other)<br>
[**examples/other/dolfin**](https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/other/dolfin)<br>
[**examples/other/trimesh**](https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/other/trimesh)<br>
[**examples/notebooks**](https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/notebooks).<br>

|         |      |
|:--------|:-----|
|Apply a *Moving Least Squares* algorithm to obtain a smooth surface from a to a large cloud of scattered points in space ([script](https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/advanced/moving_least_squares2D.py))<br>![rabbit](https://user-images.githubusercontent.com/32848391/50738808-5816ad00-11d8-11e9-9854-c952be6fb941.jpg)  | ![airplanes](https://user-images.githubusercontent.com/32848391/57341963-b8910900-713c-11e9-898a-84b6d3712bce.gif)<br> Create a simple 3D animation in exactly 10 lines of code ([script](https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/simulations/airplanes.py)).<br>Trails and shadows can be added to moving objects easily.|
|         |      |
| Simulation of a gyroscope hanging from a spring ([script](https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/simulations/gyroscope1.py)).<br> ![gyro](https://user-images.githubusercontent.com/32848391/39766016-85c1c1d6-52e3-11e8-8575-d167b7ce5217.gif)     | ![qsine2](https://user-images.githubusercontent.com/32848391/47751431-06aae880-dc92-11e8-9fcf-6659123edbfa.gif)<br>  Quantum-tunnelling effect integrating the Schroedinger equation with 4th order Runge-Kutta method. The animation shows the evolution of a particle in a box hitting a sinusoidal potential barrier. ([script](https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/simulations/tunnelling2.py)) |
|         |      |
|Visualizing a Turing system of reaction-diffusion between two molecules<sup>1</sup> ([script](https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/simulations/turing.py)) <br> ![turing](https://user-images.githubusercontent.com/32848391/40665257-1412a30e-635d-11e8-9536-4c73bf6bdd92.gif)  | ![dolf](https://user-images.githubusercontent.com/32848391/58368591-8b3fab80-7eef-11e9-882f-8b8eaef43567.gif) <br> Support for the [FEniCS/Dolfin](https://fenicsproject.org/) platform for visualization of PDE and finite element solutions ([see here](https://github.com/marcomusy/vtkplotter-examples/blob/master/vtkplotter_examples/other/dolfin)) |

<br>


## 📜 References

- M. Musy, G. Dalmasso, J. Sharpe and N. Sime, "`vtkplotter`*: plotting in FEniCS with python*", ([link](https://github.com/marcomusy/vtkplotter-examples/blob/master/docs/fenics_poster.pdf)).
Poster at the [FEniCS'2019](https://fenicsproject.org/fenics19/) Conference,
Carnegie Institution for Science Department of Terrestrial Magnetism, Washington DC, June 2019.

- G. Dalmasso, *"Evolution in space and time of 3D volumetric images"*. Talk at the Conference for [Image-based Modeling and Simulation of Morphogenesis](https://www.pks.mpg.de/imsm19/).
Max Planck Institute for the Physics of Complex Systems, Dresden, Germany, March 2019.


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
and point clouds based on VTK (Visualization Toolkit)*",
Zenodo, 10 February 2019, [doi: 10.5281/zenodo.2561402](http://doi.org/10.5281/zenodo.2561402).

[![embl_logo](https://user-images.githubusercontent.com/32848391/58046204-e9157180-7b44-11e9-81c9-e916cdf9ba84.gif)](https://www.embl.es)
