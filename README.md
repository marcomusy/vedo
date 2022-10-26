
![vlogo](https://user-images.githubusercontent.com/32848391/110344277-9bc20700-802d-11eb-8c0d-2e97226a9a32.png)


[![lics](https://img.shields.io/badge/license-MIT-blue.svg)](https://en.wikipedia.org/wiki/MIT_License)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/vedo/badges/version.svg)](https://anaconda.org/conda-forge/vedo)
[![Ubuntu 22.10 package](https://repology.org/badge/version-for-repo/ubuntu_22_10/vedo.svg)](https://repology.org/project/vedo/versions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5842090.svg)](https://doi.org/10.5281/zenodo.5842090)
[![Downloads](https://pepy.tech/badge/vedo)](https://pepy.tech/project/vedo)
[![CircleCI](https://circleci.com/gh/marcomusy/vedo.svg?style=svg)](https://circleci.com/gh/marcomusy/vedo)


A lightweight and powerful python module
for scientific analysis and **v**isualization of **3d** **o**bjects.<br>


## 💾  Installation
```bash
pip install vedo

# For better results try:
# pip install vtk==9.0.3
```

<details>
<summary>additional installation details <i><b>[click to expand]</b></i> </summary>

- Installing VTK version 9.0.x (the latest is 9.2) will generate better visualization for transparent objects.

- To install the latest _dev_ version of `vedo`: <br>
`pip install -U git+https://github.com/marcomusy/vedo.git`

- To install from the conda-forge channel: <br>
`conda install -c conda-forge vedo`

</details>


## 📙  Documentation
The webpage of the library with documentation is available [**here**](https://vedo.embl.es).

📌 **Need help? Have a question, or wish to ask for a missing feature?**
Do not hesitate to ask any questions on the [**image.sc** forum](https://forum.image.sc/)
or by opening a [**github issue**](https://github.com/marcomusy/vedo/issues).


## 🎨  Features
The library includes a [large set of working examples](https://github.com/marcomusy/vedo/tree/master/examples)
for a wide range of functionalities

<details>
<summary>working with polygonal meshes and point clouds <i><b>[click to expand]</b></i> </summary>
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


### ⌨  Command Line Interface
Visualize a polygonal mesh or a volume from a terminal window simply with:
```bash
vedo https://vedo.embl.es/examples/data/embryo.tif
```


<details>
<summary>volumetric files (slc, tiff, DICOM...) can be visualized in different modes </summary>


|Volume 3D slicing<br>`vedo --slicer embryo.slc`| Ray-casting<br>`vedo -g`| 2D slicing<br>`vedo --slicer2d`| Colorize voxels<br>`vedo --lego`|
|:--------|:-----|:--------|:-----|
| ![slicer](https://user-images.githubusercontent.com/32848391/80292484-50757180-8757-11ea-841f-2c0c5fe2c3b4.jpg)|![isohead](https://user-images.githubusercontent.com/32848391/58336107-5a09a180-7e43-11e9-8c4e-b50e4e95ae71.gif)|![viz_slicer](https://user-images.githubusercontent.com/32848391/90966778-fc955200-e4d6-11ea-8e29-215f7aea3860.png)  |![lego](https://user-images.githubusercontent.com/32848391/56969949-71b47980-6b66-11e9-8251-4bbdb275cb22.jpg) |


</details>


Type `vedo -h` for the complete list of options.<br>

## 🐾  Gallery
`vedo` currently includes 300+ working [examples](https://github.com/marcomusy/vedo/tree/master/examples) and [notebooks](https://github.com/marcomusy/vedo/tree/master/examples/notebooks). <br>

Run any of the built-in examples. In a terminal type: `vedo -r earthquake_browser`

Check out the example galleries organized by subject here:

<a href="https://vedo.embl.es/#gallery" target="_blank">

![](https://user-images.githubusercontent.com/32848391/104370203-d1aba900-551e-11eb-876c-41e0961fcdb5.jpg)

</a>


## ✏  Contributing

Any contributions are **greatly appreciated**!
If you have a suggestion that would make this better, please fork the repo and create a pull request.
You can also simply open an issue with the tag "enhancement".



## 📜  References

**Scientific publications leveraging `vedo`:**

- X. Diego *et al.*:
*"Key features of Turing systems are determined purely by network topology"*,
Phys. Rev. X 8, 021071,
[DOI](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.021071).
- M. Musy, K. Flaherty *et al.*:
*"A Quantitative Method for Staging Mouse Limb Embryos based on Limb Morphometry"*,
Development (2018) 145 (7): dev154856,
[DOI](http://dev.biologists.org/content/145/7/dev154856).
- F. Claudi, A. L. Tyson, T. Branco, *"Brainrender. A python based software for visualisation
of neuroanatomical and morphological data."*,
eLife 2021;10:e65751,
[DOI](https://doi.org/10.7554/eLife.65751).
- J. S. Bennett, D. Sijacki,
*"Resolving shocks and filaments in galaxy formation simulations: effects on gas properties and
star formation in the circumgalactic medium"*,
Monthly Notices of the Royal Astronomical Society, Volume 499, Issue 1,
[DOI](https://doi.org/10.1093/mnras/staa2835).
- J.D.P. Deshapriya *et al.*,
*"Spectral analysis of craters on (101955) Bennu"*.
Icarus 2020,
[DOI](https://doi.org/10.1016/j.icarus.2020.114252).
- A. Pollack *et al.*,
*"Stochastic inversion of gravity, magnetic, tracer, lithology, and fault data
for geologically realistic structural models: Patua Geothermal Field case study"*,
Geothermics, Volume 95, September 2021,
[DOI](https://doi.org/10.1016/j.geothermics.2021.102129).
- X. Lu *et al.*,
*"3D electromagnetic modeling of graphitic faults in the Athabasca
 Basin using a finite-volume time-domain approach with unstructured grids"*,
Geophysics,
[DOI](https://doi.org/10.1190/geo2020-0657.1).
- M. Deepa Maheshvare *et al.*,
*"A Graph-Based Framework for Multiscale Modeling of Physiological Transport"*,
Front. Netw. Physiol. 1:802881,
[DOI](https://www.frontiersin.org/articles/10.3389/fnetp.2021.802881/full).
- F. Claudi, T. Branco,
*"Differential geometry methods for constructing manifold-targeted recurrent neural networks"*,
bioRxiv 2021.10.07.463479,
[DOI](https://doi.org/10.1101/2021.10.07.463479).
- J. Klatzow, G. Dalmasso, N. Martínez-Abadías, J. Sharpe, V. Uhlmann,
*"µMatch: 3D shape correspondence for microscopy data"*,
Front. Comput. Sci., 15 February 2022.
[DOI](https://doi.org/10.3389/fcomp.2022.777615)
- G. Dalmasso *et al.*, *"4D reconstruction of murine developmental trajectories using spherical harmonics"*,
Developmental Cell 57, 1–11 September 2022,
[DOI](https://doi.org/10.1016/j.devcel.2022.08.005).

**Have you found this software useful for your research? Star ✨ the project and cite it as:**

M. Musy  <em>et al.</em>,
"<code>vedo</code>, a python module for scientific analysis and visualization of 3D objects and point clouds",
Zenodo, 2021, <a href="https://doi.org/10.5281/zenodo.7019968">doi: 10.5281/zenodo.7019968</a>.


[![embl_logo](https://user-images.githubusercontent.com/32848391/58046204-e9157180-7b44-11e9-81c9-e916cdf9ba84.gif)](https://www.embl.es)


