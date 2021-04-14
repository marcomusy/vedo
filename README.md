
![vlogo](https://user-images.githubusercontent.com/32848391/110344277-9bc20700-802d-11eb-8c0d-2e97226a9a32.png)

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/4acbc84816a540bfb9dc67bbff520d38)](https://www.codacy.com/manual/marcomusy/vedo?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=marcomusy/vedo&amp;utm_campaign=Badge_Grade)
[![Downloads](https://pepy.tech/badge/vedo)](https://pepy.tech/project/vedo)
[![lics](https://img.shields.io/badge/license-MIT-blue.svg)](https://en.wikipedia.org/wiki/MIT_License)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/vedo/badges/version.svg)](https://anaconda.org/conda-forge/vedo)
[![Ubuntu 20.10](https://repology.org/badge/version-for-repo/ubuntu_20_10/vedo.svg)](https://repology.org/project/vedo/versions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4609336.svg)](https://doi.org/10.5281/zenodo.4609336)
[![CircleCI](https://circleci.com/gh/marcomusy/vedo.svg?style=svg)](https://circleci.com/gh/marcomusy/vedo)


`vedo` is a lightweight and powerful python module
for scientific analysis and **v**isualization of **3d** **o**bjects.<br>


## ✨  Philosophy
Inspired by the [vpython](https://vpython.org/) *manifesto* "3D programming for ordinary mortals",
`vedo` makes it easy to work with three-dimensional meshes and volumes, creating displays and animations
in just a few lines of code, even for less experienced programmers.

`vedo` is based on [numpy](http://www.numpy.org/) and [VTK](https://www.vtk.org/),
with no other dependencies.


## 💾  Installation
```bash
pip install vedo
```

<details>
<summary>additional installation details <i>(click to expand)</i> </summary>

- As some of the features work better and faster on the older VTK version 8 you can install it via:<br>
`pip install vtk==8.1.2`

- To install the latest _dev_ version of `vedo`: <br>
`pip install -U git+https://github.com/marcomusy/vedo.git`

- To install from the conda-forge channel: <br>
`conda install -c conda-forge vedo`

</details>

---------------------------------------------------------------------

📌 **Done?** Run any of the built-in examples. In a terminal, type e.g.:

**`vedo -r covid19`**

Visualize a file from web URL (or your dropbox!), type:

**`vedo https://vedo.embl.es/examples/data/panther.stl.gz`**


## 📙  Documentation
[The webpage of the library with documentation is available here](https://vedo.embl.es).

📌 **Need help?** Have any question, or wish to ask for a missing feature?
Do not hesitate to open a [**issue**](https://github.com/marcomusy/vedo/issues)
(or send an [email](mailto:marco.musy@embl.es)).


## 🎨  Features
The `vedo` library includes a [large set of working examples](https://github.com/marcomusy/vedo/tree/master/examples)
for a wide range of functionalities:

<details>
<summary>working with polygonal meshes and point clouds <i>(click to expand)</i> </summary>
<i>

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
- Find the closest path from one point to another, traveling along the edges of a mesh.
- Find the intersection of a mesh with lines, planes or other meshes.
- Interpolate scalar and vectorial fields with Radial Basis Functions and Thin Plate Splines.
- Add sliders and buttons to interact with the scene and the individual objects.
- Visualization of tensors.
- Analysis of Point Clouds:
- Moving Least Squares smoothing of 2D, 3D and 4D clouds
- Fit lines, planes, spheres and ellipsoids in space
- Identify outliers in a distribution of points
- Decimate a cloud to a uniform distribution.

</i>
</details>

<details>
<summary>working with volumetric data and tetrahedral meshes</summary>
<i>

- Import data from VTK format volumetric TIFF stacks, DICOM, SLC, MHD and more
- Import 2D images as PNG, JPEG, BMP
- Isosurfacing of volumes
- Composite and maximum projection volumetric rendering
- Generate volumetric signed-distance data from an input surface mesh
- Probe volumes with lines and planes
- Generate stream-lines and stream-tubes from vectorial fields
- Slice and crop volumes
- Support for other volumetric structures (structured and grid data)

</i>
</details>

<details>
<summary>plotting and histogramming in 2D and 3D</summary>
<i>

- Polygonal 3D text rendering with Latex-like syntax and unicode characters, with 14 different fonts.
- Fully customizable axis styles
- donut plots and pie charts
- Scatter plots in 2D and 3D
- Surface function plotting
- 1D customizable histograms
- 2D hexagonal histograms
- Polar plots, spherical plots and histogramming
- Draw latex-formatted formulas in the rendering window.
- Quiver, violin, whisker and stream-line plots
- Graphical markers analogous to matplotlib

</i>
</details>

<details>
<summary>integration with other libraries</summary>
<i>

- Integration with the [Qt5](https://www.qt.io/) framework.
- Support for [FEniCS/Dolfin](https://fenicsproject.org/) platform for visualization of PDE/FEM solutions.
- Interoperability with the [trimesh](https://trimsh.org/), [pyvista](https://github.com/pyvista/pyvista) and [pymeshlab](https://github.com/cnr-isti-vclab/PyMeshLab) libraries.
- Export 3D scenes and embed them into a [web page](https://vedo.embl.es/examples/fenics_elasticity.html).
- Embed 3D scenes in *jupyter* notebooks with [K3D](https://github.com/K3D-tools/K3D-jupyter) (can export an interactive 3D-snapshot page [here](https://vedo.embl.es/examples/geo_scene.html)).

</i>
</details>


## ⌨  Command Line Interface
Visualize a polygonal mesh from a terminal window simply with e.g.:
```bash
vedo my_mesh.obj
```
Volumetric files (_mhd, vti, slc, tiff, DICOM etc.._) can be visualized in different modes:


|Volume 3D slicing<br>`vedo --slicer embryo.slc`| Ray-casting<br>`vedo -g`| 2D slicing<br>`vedo --slicer2d`| Colorize voxels<br>`vedo --lego`|
|:--------|:-----|:--------|:-----|
| ![slicer](https://user-images.githubusercontent.com/32848391/80292484-50757180-8757-11ea-841f-2c0c5fe2c3b4.jpg)|![isohead](https://user-images.githubusercontent.com/32848391/58336107-5a09a180-7e43-11e9-8c4e-b50e4e95ae71.gif)|![viz_slicer](https://user-images.githubusercontent.com/32848391/90966778-fc955200-e4d6-11ea-8e29-215f7aea3860.png)  |![lego](https://user-images.githubusercontent.com/32848391/56969949-71b47980-6b66-11e9-8251-4bbdb275cb22.jpg) |

Type `vedo -h` for the complete list of options.<br>

## 🐾  Examples
`vedo` currently includes 300+ working [examples](https://github.com/marcomusy/vedo/tree/master/examples) and [notebooks](https://github.com/marcomusy/vedo/tree/master/examples/notebooks). <br>

|         |         |         |
|:--------|:--------|:--------|
| [![cubecut](https://user-images.githubusercontent.com/32848391/99916179-e763d580-2d08-11eb-9044-b647115167ed.jpg)](https://github.com/marcomusy/vedo/tree/master/examples/advanced/cutWithMesh2.py)   | [![greyscott](https://user-images.githubusercontent.com/32848391/80291855-87e11f80-8751-11ea-9428-12e193a2a66e.gif)](https://github.com/marcomusy/vedo/tree/master/examples/simulations/grayscott.py)| [![quatumsine](https://user-images.githubusercontent.com/32848391/47751431-06aae880-dc92-11e8-9fcf-6659123edbfa.gif)](https://github.com/marcomusy/vedo/tree/master/examples/simulations/tunnelling2.py) |
| *Easily work with volumes, tetrahedral and polygonal meshes.*        | *Turing system of reaction-diffusion between two molecules.*                                                                   |  *Quantum-tunnelling of a particle in a box hitting a sinusoidal potential.*  |
| [![trimesh](https://user-images.githubusercontent.com/32848391/91164151-e8b44080-e6ce-11ea-8213-cf5b12aa4d16.png)](https://github.com/marcomusy/vedo/blob/master/examples/other/trimesh)              | [![dolf](https://user-images.githubusercontent.com/32848391/58368591-8b3fab80-7eef-11e9-882f-8b8eaef43567.gif)](https://vedo.embl.es/content/vedo/dolfin.html)| [![whisker](https://user-images.githubusercontent.com/32848391/99916183-e8950280-2d08-11eb-8070-8bb1146c7c62.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/whiskers.py) |
| *Interoperability with external libraries like [trimesh](https://trimsh.org/), [pymeshlab](https://github.com/cnr-isti-vclab/PyMeshLab), and [pyvista](https://github.com/pyvista/pyvista).*  |  *Support for the [FEniCS/Dolfin](https://fenicsproject.org/) library for PDE and finite element solutions.*       | *Advanced 2D/3D histogramming and plotting capablities.* |


### Galleries
Check out the example galleries organized by subject here:

<a href="https://vedo.embl.es/#gallery" target="_blank">

![](https://user-images.githubusercontent.com/32848391/104370203-d1aba900-551e-11eb-876c-41e0961fcdb5.jpg)

</a>
<br>


## 📜  References

**Scientific publications leveraging `vedo` (formerly known as `vtkplotter`):**

- X. Diego *et al.*:
*"Key features of Turing systems are determined purely by network topology"*,
[Physical Review X, 20 June 2018](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.021071).
- M. Musy, K. Flaherty *et al.*:
*"A Quantitative Method for Staging Mouse Limb Embryos based on Limb Morphometry"*,
Development, 5 April 2018, [doi: 10.1242/dev.154856](http://dev.biologists.org/content/145/7/dev154856).
- G. Dalmasso *et al.*, *"Evolution in space and time of 3D volumetric images"*, in preparation.
- F. Claudi, A. L. Tyson, T. Branco, *"Brainrender. A python based software for visualisation of neuroanatomical and morphological data."*,
 [DOI](https://doi.org/10.1101/2020.02.23.961748).
- J. S. Bennett, D. Sijacki, <i>"Resolving shocks and filaments in galaxy formation
                simulations: effects on gas properties and star formation in the circumgalactic medium"</i>,
                Monthly Notices of the Royal Astronomical Society, Vol. 499, Issue 1, Nov. 2020, <a href="https://doi.org/10.1093/mnras/staa2835">DOI</a>.
- J.D.P. Deshapriya et al., *"Spectral analysis of craters on (101955) Bennu"*. (DOI: 10.1016/j.icarus.2020.114252)

**Have you found this software useful for your research? Star ✨ the project and cite it as:**

M. Musy  <em>et al.</em>,
"<code>vedo</code>, a python module for scientific analysis and visualization of 3D objects and point clouds",
Zenodo, 2021, <a href="https://doi.org/10.5281/zenodo.4609336">doi: 10.5281/zenodo.4609336</a>.

[![embl_logo](https://user-images.githubusercontent.com/32848391/58046204-e9157180-7b44-11e9-81c9-e916cdf9ba84.gif)](https://www.embl.es)


