![vlogo](https://user-images.githubusercontent.com/32848391/52522718-50d83880-2c89-11e9-80ff-df1b5618a84a.png)

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/e8c5f1f9afb6433a9cdf4edb5499bd46)](https://app.codacy.com/app/marcomusy/vtkplotter?utm_source=github.com&utm_medium=referral&utm_content=marcomusy/vtkplotter&utm_campaign=Badge_Grade_Dashboard)
[![Downloads](https://pepy.tech/badge/vtkplotter)](https://pepy.tech/project/vtkplotter)
[![lics](https://img.shields.io/badge/license-MIT-blue.svg)](https://en.wikipedia.org/wiki/MIT_License)
[![pythvers](https://img.shields.io/badge/python-2.7%7C3-brightgreen.svg)](https://pypi.org/project/vtkplotter)
[![gdocs](https://img.shields.io/badge/docs%20by-gendocs-blue.svg)](https://gendocs.readthedocs.io/en/latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2561402.svg)](https://doi.org/10.5281/zenodo.2561402)
[![CircleCI](https://circleci.com/gh/marcomusy/vtkplotter.svg?style=svg)](https://circleci.com/gh/marcomusy/vtkplotter)

A lightweight pure python module for scientific analysis and visualization of 3D objects
and *point clouds* based on [VTK](https://www.vtk.org/) and [numpy](http://www.numpy.org/).<br>

[![isohead](https://user-images.githubusercontent.com/32848391/81347909-59eacc00-90bd-11ea-9392-f1a3ecc0c850.jpg)](https://vtkplotter.embl.es/examples/geo_scene.html)

*(click on the picture to open a 3D viewer)*

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
  * [Gallery](https://github.com/marcomusy/vtkplotter#-random-gallery)
* [References](https://github.com/marcomusy/vtkplotter#-references)



## 💾 Installation
Use [pip](https://projects.raspberrypi.org/en/projects/using-pip-on-windows) to install:
```bash
pip install -U vtkplotter
```
or use [conda](https://anaconda.org/conda-forge/vtkplotter) to install from the conda-forge channel:
```
conda install -c conda-forge vtkplotter
```

*Windows-10 users* can place this file
[vtkplotter.bat](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter.bat)
on the desktop to *drag&drop* files to visualize.
(Need to edit the path of the local python installation).


## 📙 Documentation
Automatically generated documentation can be found [**here**](https://vtkplotter.embl.es).


##### 📌 Need help?
Have any question, or wish to suggest or ask for a missing feature?
Do not hesitate to open a [**issue**](https://github.com/marcomusy/vtkplotter/issues)
or send an [email](mailto:marco.musy@embl.es).


## 🎨 Features
Intuitive and straightforward API which can be combined with VTK seamlessly
in a program, whilst mantaining access to the full range of VTK native classes.

It includes a **[large set of working examples](https://github.com/marcomusy/vtkplotter/tree/master/vtkplotter/examples)**
for a wide range of functionalities *(click triangle to expand...)*:
<details>
<summary>working with polygonal meshes and point clouds</summary>

	- Import meshes from VTK format, STL, Wavefront OBJ, 3DS, Dolfin-XML, Neutral, GMSH, OFF, PCD (PointCloud),

	- Export meshes as ASCII or binary to VTK, STL, OBJ, PLY ... formats.

	- Analysis tools like Moving Least Squares, mesh morphing and more..

	- Tools to visualize and edit meshes (cutting a mesh with another mesh, slicing, normalizing, moving vertex positions, etc..).

	- Split mesh based on surface connectivity. Extract the largest connected area.

	- Calculate areas, volumes, center of mass, average sizes etc.

	- Calculate vertex and face normals, curvatures, feature edges. Fill mesh holes.

	- Subdivide faces of a mesh, increasing the number of vertex points. Mesh simplification.

	- Coloring and thresholding of meshes based on associated scalar or vectorial data.

	- Point-surface operations: find nearest points, determine if a point lies inside or outside of a mesh.

	- Create primitive shapes: spheres, arrows, cubes, torus, ellipsoids...

	- Generate glyphs (associate a mesh to every vertex of a source mesh).

	- Create animations easily by just setting the position of the displayed objects in the 3D scene. Add trailing lines and shadows to moving objects is supported.

	- Straightforward support for multiple sync-ed or independent renderers in  the same window.

	- Registration (alignment) of meshes with different techniques.

	- Mesh smoothing.

	- Delaunay triangulation in 2D and 3D.

	- Generate meshes by joining nearby lines in space.

	- Find the closest path from one point to another, travelling along the edges of a mesh.

	- Find the intersection of a mesh with lines, planes or other meshes.

	- Interpolate scalar and vectorial fields with Radial Basis Functions and Thin Plate Splines.

	- Add sliders and buttons to interact with the scene and the individual objects.

	- Visualization of tensors.

	- Analysis of Point Clouds:

	 - Moving Least Squares smoothing of 2D, 3D and 4D clouds

	 - Fit lines, planes, spheres and ellipses in space

	 - Identify outliers in a distribution of points

	 - Decimate a cloud to a uniform distribution.
</details>
<details>
<summary>working with volumetric data</summary>

	- Import data from VTK format volumetric TIFF stacks, DICOM, SLC, MHD and more

	- Import 2D images as PNG, JPEG, BMP

	- Isosurfacing of volumes

	- Composite and maximum projection volumetric rendering

	- Generate volumetric signed-distance data from an input surface mesh

	- Probe a volume with lines and planes

	- Generate stream-lines and stream-tubes from vectorial fields

	- Slice and crop volumes

	- Support for other volumetric structures (structured and grid data)
</details>
<details>
<summary>plotting and histogramming</summary>

	- Fully customizable axis styles

	- 'donut' plots and pie charts

	- Scatter plots in 2D and 3D

	- Surface function plotting

	- 1D customizable histograms

	- 2D hexagonal histograms

	- Polar plots, spherical plots and histogramming

	- Draw latex-formatted formulas in the rendering window.

	- Quiver plots
	
	- Stream line plots

	- Point markers analogous to matplotlib
</details>


Moreover:
- Integration with the *Qt5* framework.
- Examples using [SHTools](https://shtools.oca.eu/shtools) package for *spherical harmonics* expansion of a mesh shape.
- Support for [FEniCS/Dolfin](https://fenicsproject.org/) platform for visualization of finite-element calculations.
- Interoperability with the [trimesh](https://trimsh.org/) library.
- Export a 3D scene and embed it into a [web page](https://vtkplotter.embl.es/examples/fenics_elasticity.html).
- Embed the 3D rendering in a *jupyter* notebook with [K3D](https://github.com/K3D-tools/K3D-jupyter) (can export an interactive 3D-snapshot page [here](https://vtkplotter.embl.es/examples/geo_scene.html)).


## ⌨ Command Line Interface
Visualize a polygonal mesh from a terminal window with:
```bash
vtkplotter mesh.obj
# valid formats: [vtk,vtu,vts,vtp,vtm,ply,obj,stl,3ds,dolfin-xml,neutral,gmsh,
#                 pcd,xyz,txt,byu,tif,off,slc,vti,mhd,dcm,dem,nrrd,nii,bmp,png,jpg]
```
Voxel-data (_mhd, vti, slc, tiff, dicom etc.._) files can be visualized with options:<br>

|Use sliders to slice a volume in the 3 planes:<br>`vtkplotter --slicer embryo.slc`|  Ray-casting rendering:<br>`vtkplotter -g embryo.slc`|
|:--------|:-----|
| ![slicer](https://user-images.githubusercontent.com/32848391/80292484-50757180-8757-11ea-841f-2c0c5fe2c3b4.jpg)|![isohead](https://user-images.githubusercontent.com/32848391/58336107-5a09a180-7e43-11e9-8c4e-b50e4e95ae71.gif)|


<br>

To visualize multiple files or files time-sequences try `-n` or `-s` options. Use `-h` for the complete list of options.

| Interactive volume isosurfacing:|  Load and browse a sequence of meshes:| Slice a 3D volume with a plane:| Visualize colorized voxels:|
|:--------|:-----|:----|:----|
|`vtkplotter head.vti` |`vtkplotter -s *.vtk` |`vtkplotter `<br>`--slicer2d em.slc` |   `vtkplotter --lego embryo.slc`|
|![isohead](https://user-images.githubusercontent.com/32848391/56972083-a7f3f800-6b6a-11e9-9cb3-1047b69dcad2.gif)|   ![viz_raycast](https://user-images.githubusercontent.com/32848391/58336919-f7b1a080-7e44-11e9-9106-f574371093a8.gif)  | ![viz_slicer](https://user-images.githubusercontent.com/32848391/56972084-a7f3f800-6b6a-11e9-98c4-dc4ffec70a5e.png)  |![lego](https://user-images.githubusercontent.com/32848391/56969949-71b47980-6b66-11e9-8251-4bbdb275cb22.jpg) |

## 🖥 Graphic User Interface
<details>
<summary>A GUI is available (mainly useful to *Windows 10* users)</summary>

![gui](https://user-images.githubusercontent.com/32848391/63259840-c861d280-c27f-11e9-9c2a-99d0fae85313.png)
</details>

## 🐾 Examples
**More than 300 working examples can be found in directories** _(scroll down to see thumbnails):_ <br>
[**examples/basic**](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/basic)<br>
[**examples/advanced**](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced)<br>
[**examples/volumetric**](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/volumetric)<br>
[**examples/simulations**](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/simulations)<br>
[**examples/pyplot**](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/pyplot)<br>
[**examples/other**](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/other)<br>
[**examples/other/dolfin**](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/other/dolfin)<br>
[**examples/other/trimesh**](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/other/trimesh)<br>
[**examples/notebooks**](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/notebooks).<br>

|         |      |
|:--------|:-----|
|Apply a *Moving Least Squares* algorithm to obtain a smooth surface from a to a large cloud of scattered points in space ([script](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced/moving_least_squares2D.py))<br>![rabbit](https://user-images.githubusercontent.com/32848391/50738808-5816ad00-11d8-11e9-9854-c952be6fb941.jpg)  | ![airplanes](https://user-images.githubusercontent.com/32848391/57341963-b8910900-713c-11e9-898a-84b6d3712bce.gif)<br> Create a simple 3D animation in exactly 10 lines of code ([script](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/simulations/airplanes.py)).<br>Trails and shadows can be added to moving objects easily.|
|         |      |
| Simulation of a gyroscope hanging from a spring ([script](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/simulations/gyroscope1.py)).<br> ![gyro](https://user-images.githubusercontent.com/32848391/39766016-85c1c1d6-52e3-11e8-8575-d167b7ce5217.gif)     | ![qsine2](https://user-images.githubusercontent.com/32848391/47751431-06aae880-dc92-11e8-9fcf-6659123edbfa.gif)<br>  Quantum-tunnelling effect integrating the Schroedinger equation with 4th order Runge-Kutta method. The animation shows the evolution of a particle in a box hitting a sinusoidal potential barrier. ([script](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/simulations/tunnelling2.py)) |
|         |      |
|Visualizing a Turing system of reaction-diffusion between two molecules<sup>1</sup> ([script](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/simulations/turing.py)) <br> ![turing](https://user-images.githubusercontent.com/32848391/40665257-1412a30e-635d-11e8-9536-4c73bf6bdd92.gif)  | ![dolf](https://user-images.githubusercontent.com/32848391/58368591-8b3fab80-7eef-11e9-882f-8b8eaef43567.gif) <br> Support for the [FEniCS/Dolfin](https://fenicsproject.org/) platform for visualization of PDE and finite element solutions ([see here](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/other/dolfin)) |

<br>

### Random Gallery
Run any of the built-in examples from command line, e.g.:

`vtkplotter -ir  grayscott`

![covid](https://user-images.githubusercontent.com/32848391/80291855-87e11f80-8751-11ea-9428-12e193a2a66e.gif)


More examples *(press spacebar to continue or `F1` to interrupt execution)*:

`vtkplotter --list`

|     |     |     |     |
|:---:|:---:|:---:|:---:|
| ![covid](https://user-images.githubusercontent.com/32848391/77330206-4824b380-6d1f-11ea-9bc3-e3aef970dcc2.gif) <br>`covid19`| ![lineage_graph](https://user-images.githubusercontent.com/32848391/80291851-8152a800-8751-11ea-893e-4a0bb85397b1.png) <br>`lineage_graph` |![siluette](https://user-images.githubusercontent.com/32848391/57179369-8e5df380-6e7d-11e9-99b4-3b1a120dd375.png) <br>`silhouette1`  | ![levelterrain](https://user-images.githubusercontent.com/32848391/72433087-f00a8780-3798-11ea-9778-991f0abeca70.png)<br>`isolines`|
| ![gyro](https://user-images.githubusercontent.com/32848391/50738942-687b5780-11d9-11e9-97f0-72bbd63f7d6e.gif) <br>`gyroscope2` | ![thinplate_grid](https://user-images.githubusercontent.com/32848391/51433540-d188b380-1c4c-11e9-81e7-a1cf4642c54b.png ) <br>`thinplate_grid`  | ![trail](https://user-images.githubusercontent.com/32848391/58370826-4aee2680-7f0b-11e9-91e6-3120770cfede.gif) <br>`trail`   | ![quadratic_morphing](https://user-images.githubusercontent.com/32848391/50738890-db380300-11d8-11e9-9cef-4c1276cca334.jpg)  <br>`quadratic_morphing`  |
| ![shrink](https://user-images.githubusercontent.com/32848391/46819143-41042280-cd83-11e8-9492-4f53679887fa.png) <br>`shrink` | ![mesh_custom](https://user-images.githubusercontent.com/32848391/51390972-20d9c180-1b31-11e9-955d-025f1ef24cb7.png) <br>`mesh_custom`   | ![spring](https://user-images.githubusercontent.com/32848391/36788885-e97e80ae-1c8f-11e8-8b8f-ffc43dad1eb1.gif) <br>`spring`   | ![lorenz](https://user-images.githubusercontent.com/32848391/46818115-be7a6380-cd80-11e8-8ffb-60af2631bf71.png) <br>`lorentz`   |
| ![sliders](https://user-images.githubusercontent.com/32848391/50738848-be033480-11d8-11e9-9b1a-c13105423a79.jpg) <br>`sliders` | ![fitspheres1](https://user-images.githubusercontent.com/32848391/50738943-687b5780-11d9-11e9-87a6-054e0fe76241.jpg) <br>`fitspheres1`   | ![fxy](https://user-images.githubusercontent.com/32848391/36611824-fd524fac-18d4-11e8-8c76-d3d1b1bb3954.png) <br>`plot4_fxy`   | ![histogram](https://user-images.githubusercontent.com/32848391/68141260-77cc4e00-ff2d-11e9-9280-0efc5b87314d.png) <br>`histo_1D`   |
| ![plot_err_band](https://user-images.githubusercontent.com/32848391/73483464-c019d180-439f-11ea-9a8c-59fa49e9ecf4.png) <br>`plot2_errband` | ![histogram2D](https://user-images.githubusercontent.com/32848391/72452359-b5671600-37bd-11ea-8b1d-c44d884496ed.png) <br>`histo_2D`| ![histoHexagonal.py](https://user-images.githubusercontent.com/32848391/72434748-b471bc80-379c-11ea-95d7-d70333770582.png) <br>`histo_hexagonal`    | ![sphericPlot](https://user-images.githubusercontent.com/32848391/72433091-f0a31e00-3798-11ea-86bd-6c522e23ec61.png) <br>`plot5_spheric`    |
| ![boolean](https://user-images.githubusercontent.com/32848391/50738871-c0fe2500-11d8-11e9-8812-442b69be6db9.png) <br>`boolean` | ![brownian2D](https://user-images.githubusercontent.com/32848391/50738948-73ce8300-11d9-11e9-8ef6-fc4f64c4a9ce.gif) <br>`brownian2D`   | ![gas](https://user-images.githubusercontent.com/32848391/50738954-7e891800-11d9-11e9-95aa-67c92ca6476b.gif) <br>`gas`   | ![self_org_maps2d](https://user-images.githubusercontent.com/32848391/54557310-1ade5080-49bb-11e9-9b97-1b53a7689a9b.gif)  <br>`self_org_maps2d`    |
| ![geodesic](https://user-images.githubusercontent.com/32848391/51855637-015f4780-232e-11e9-92ca-053a558e7f70.png)<br>`geodesic` | ![convexHull](https://user-images.githubusercontent.com/32848391/51932732-068cc700-2400-11e9-9b68-30294a4fa4e3.png)  <br>`convexHull`  | ![flatarrow](https://user-images.githubusercontent.com/32848391/54612632-97c00780-4a59-11e9-8532-940c25a5dfd8.png) <br>`flatarrow`   | ![latex](https://user-images.githubusercontent.com/32848391/55568648-6190b200-5700-11e9-9547-0798c588a7a5.png)  <br>`latex`  |
| ![legosurface](https://user-images.githubusercontent.com/32848391/56820682-da40e500-684c-11e9-8ea3-91cbcba24b3a.png)<br>`legosurface`| ![streamlines2](https://user-images.githubusercontent.com/32848391/56964001-9145a500-6b5a-11e9-935b-1b2425bd7dd2.png) <br>`streamlines2`   | ![office](https://user-images.githubusercontent.com/32848391/56964003-9145a500-6b5a-11e9-9d9e-9736d90e1900.png) <br>`office.py`   | ![value-iteration](https://user-images.githubusercontent.com/32848391/56964055-afaba080-6b5a-11e9-99cf-3fac99df9878.jpg)  <br>`value-iteration`  |
| ![shadow](https://user-images.githubusercontent.com/32848391/57312574-1d714280-70ee-11e9-8741-04fc5386d692.png) <br>`shadow`| ![multiple_pendulum](https://user-images.githubusercontent.com/32848391/50738892-db380300-11d8-11e9-807c-fb320c7b7917.gif) <br>`multiple_pend`   | ![interpolateVolume](https://user-images.githubusercontent.com/32848391/59095175-1ec5a300-8918-11e9-8bc0-fd35c8981e2b.jpg) <br>`interpolateVolume`   | ![histo_polar](https://user-images.githubusercontent.com/32848391/64912717-5754f400-d733-11e9-8a1f-612165955f23.png)  <br>`histo_polar`  |
| ![streamplot](https://user-images.githubusercontent.com/32848391/73614123-93162a80-45fc-11ea-969b-9a3293b26f35.png) <br>`plot7_stream`| ![violin](https://user-images.githubusercontent.com/32848391/73481240-b55d3d80-439b-11ea-89a4-6c35ecc84b0d.png) <br>`histo_violin`   | ![plot3_pip](https://user-images.githubusercontent.com/32848391/73393632-4ff64780-42dc-11ea-8798-45a81c067f45.png) <br>`plot3_pip`   | ![histo_spheric](https://user-images.githubusercontent.com/32848391/73392901-fccfc500-42da-11ea-828a-9bad6982a823.png)  <br>`histo_spheric`  |
| ![gray_scott](https://user-images.githubusercontent.com/32848391/59788744-aaeaa980-92cc-11e9-825d-58da26ca21ff.gif) | ![donutPlot](https://user-images.githubusercontent.com/32848391/64998178-6f6b7580-d8e3-11e9-9bd8-8dfb9ccd90e4.png)  <br>`donut`  | ![extrude](https://user-images.githubusercontent.com/32848391/65963682-971e1a00-e45b-11e9-9f29-05522ae4a800.png) <br>`extrude`   | ![plotxy](https://user-images.githubusercontent.com/32848391/69158509-d6c1c380-0ae6-11ea-9dbf-ff5cd396a9a6.png) <br>`plot1_errbars`   |
| ![particle_simulator](https://user-images.githubusercontent.com/32848391/50738891-db380300-11d8-11e9-84c2-0f55be7228f1.gif) <br>`particle_simulator`| ![heatconv](https://user-images.githubusercontent.com/32848391/57455107-b200af80-726a-11e9-897d-9c7bcb9854ac.gif) <br>`heatconv` |![elastodynamics](https://user-images.githubusercontent.com/32848391/54932788-bd4a8680-4f1b-11e9-9326-33645171a45e.gif) <br>`elastodynamics`  | ![navier-stokes_lshape](https://user-images.githubusercontent.com/32848391/56671156-6bc91f00-66b4-11e9-8c58-e6b71e2ad1d0.gif)<br>`stokes_lshape`|

## 📜 References

- M. Musy, G. Dalmasso, J. Sharpe and N. Sime, "`vtkplotter`*: plotting in FEniCS with python*", ([link](https://github.com/marcomusy/vtkplotter/blob/master/docs/fenics_poster.pdf)).
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
4. F. Claudi, A. L. Tyson, T. Branco, *"Brainrender. A python based software for visualisation of neuroanatomical and morphological data."*
bioRxiv 2020.02.23.961748; doi: https://doi.org/10.1101/2020.02.23.961748 

**Have you found this software useful for your research? Please cite it as:**<br>
M. Musy  _et al._
"`vtkplotter`*, a python module for scientific visualization and analysis of 3D objects
and point clouds based on VTK (Visualization Toolkit)*",
Zenodo, 10 February 2019, [doi: 10.5281/zenodo.2561402](http://doi.org/10.5281/zenodo.2561402).

[![embl_logo](https://user-images.githubusercontent.com/32848391/58046204-e9157180-7b44-11e9-81c9-e916cdf9ba84.gif)](https://www.embl.es)
