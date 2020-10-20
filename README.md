
![vlogo](https://user-images.githubusercontent.com/32848391/90966721-1e420980-e4d6-11ea-998f-3285d512541f.png)

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/4acbc84816a540bfb9dc67bbff520d38)](https://www.codacy.com/manual/marcomusy/vedo?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=marcomusy/vedo&amp;utm_campaign=Badge_Grade)
[![Downloads](https://pepy.tech/badge/vedo)](https://pepy.tech/project/vedo)
[![lics](https://img.shields.io/badge/license-MIT-blue.svg)](https://en.wikipedia.org/wiki/MIT_License)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/vedo/badges/version.svg)](https://anaconda.org/conda-forge/vedo)
[![Ubuntu 20.10](https://repology.org/badge/version-for-repo/ubuntu_20_10/vedo.svg)](https://repology.org/project/vedo/versions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2561402.svg)](https://doi.org/10.5281/zenodo.2561402)
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

# To get the latest dev version use instead:
pip install -U git+https://github.com/marcomusy/vedo.git

# Or from the conda-forge channel:
conda install -c conda-forge vedo
```

---------------------------------------------------------------------

📌 **Done?** Run any of the built-in examples. In a terminal, type:

**`vedo -r covid19`**

Visualize a file from web URL (or your dropbox!), type:

**`vedo https://vedo.embl.es/examples/data/panther.stl.gz`**

Visualize a whole scene, type:

**`vedo https://vedo.embl.es/examples/geo_scene.npy`**


## 📙  Documentation
[Automatically generated documentation is available here](https://vedo.embl.es).

📌 **Need help?** Have any question, or wish to ask for a missing feature?
Do not hesitate to open a [**issue**](https://github.com/marcomusy/vedo/issues)
(or send an [email](mailto:marco.musy@embl.es)).


## 🎨  Features
The `vedo` library includes a [large set of working examples](https://github.com/marcomusy/vedo/tree/master/examples)
for a wide range of functionalities:

<details>
<summary>working with polygonal meshes and point clouds (click to expand)</summary>
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
<summary>working with volumetric data, and tetrahedral meshes</summary>
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
- Interoperability with the [trimesh](https://trimsh.org/) and [pyvista](https://github.com/pyvista/pyvista) libraries.
- Export 3D scenes and embed them into a [web page](https://vedo.embl.es/examples/fenics_elasticity.html).
- Embed 3D scenes in *jupyter* notebooks with [K3D](https://github.com/K3D-tools/K3D-jupyter) (can export an interactive 3D-snapshot page [here](https://vedo.embl.es/examples/geo_scene.html)).

</i>
</details>


## ⌨  Command Line Interface
Visualize a polygonal mesh from a terminal window simply with:
```bash
vedo my_mesh.obj
# valid formats: [vtk,vtu,vts,vtp,vtm,ply,obj,stl,3ds,dolfin-xml,neutral,gmsh,
#                 pcd,xyz,txt,byu,tif,off,slc,vti,mhd,dcm,dem,nrrd,nii,bmp,png,jpg]
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
| [![airplanes](https://user-images.githubusercontent.com/32848391/57341963-b8910900-713c-11e9-898a-84b6d3712bce.gif)](https://github.com/marcomusy/vedo/tree/master/examples/simulations/airplanes.py) | [![greyscott](https://user-images.githubusercontent.com/32848391/80291855-87e11f80-8751-11ea-9428-12e193a2a66e.gif)](https://github.com/marcomusy/vedo/tree/master/examples/simulations/grayscott.py)| [![quatumsine](https://user-images.githubusercontent.com/32848391/47751431-06aae880-dc92-11e8-9fcf-6659123edbfa.gif)](https://github.com/marcomusy/vedo/tree/master/examples/simulations/tunnelling2.py) |
| *Create a simple 3D animation in exactly 10 lines of code.*          | *Turing system of reaction-diffusion between two molecules.*                                                                   |  *Quantum-tunnelling of a particle in a box hitting a sinusoidal potential.*  |
| [![trimesh](https://user-images.githubusercontent.com/32848391/91164151-e8b44080-e6ce-11ea-8213-cf5b12aa4d16.png)](https://github.com/marcomusy/vedo/blob/master/examples/other/trimesh)              | [![dolf](https://user-images.githubusercontent.com/32848391/58368591-8b3fab80-7eef-11e9-882f-8b8eaef43567.gif)](https://vedo.embl.es/content/vedo/dolfin.html)| [![whisker](https://user-images.githubusercontent.com/32848391/95772479-170cd000-0cbd-11eb-98c4-20c5ca342cb8.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/whiskers.py) |
| *Interoperability with the [trimesh](https://trimsh.org/) library.*  |  *Support for the [FEniCS/Dolfin](https://fenicsproject.org/) library for PDE and finite element solutions.*                   | *Advanced 2D histogramming and plotting capablities.* |


### Gallery
Run any of the _built-in_ examples from command line by typing:

**`vedo --run covid19`**

|     |     |     |     |
|:---:|:---:|:---:|:---:|
| [![covid](https://user-images.githubusercontent.com/32848391/77330206-4824b380-6d1f-11ea-9bc3-e3aef970dcc2.gif)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/covid19.py) <br>`covid19`                           | [![caption](https://user-images.githubusercontent.com/32848391/90437536-dc2e5780-e0d2-11ea-8951-f905ffb54f54.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/caption.py) <br>`caption`                                 | [![fonts3d](https://user-images.githubusercontent.com/32848391/90437540-dd5f8480-e0d2-11ea-8ddc-8839688979d0.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/fonts3d.py) <br>`font`                        | [![fonts](https://user-images.githubusercontent.com/32848391/90966829-9bba4980-e4d7-11ea-9ec8-23bac4b7448e.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/fonts3d.py) <br>`fonts` |
| [![wave_eq](https://user-images.githubusercontent.com/32848391/39360796-ea5f9ef0-4a1f-11e8-85cb-f3e21072c7d5.gif)](https://github.com/marcomusy/vedo/blob/master/examples/simulations/wave_equation.py) <br>`wave_equation`        | [![doubleslit](https://user-images.githubusercontent.com/32848391/96374703-86c70300-1174-11eb-9bfb-431a1ae5346d.png)](https://github.com/marcomusy/vedo/blob/master/examples/simulations/doubleslit.py) <br>`doubleslit`                   | [![tun1](https://user-images.githubusercontent.com/32848391/96375030-e0c8c800-1176-11eb-8fde-83a65de41330.gif)](https://github.com/marcomusy/vedo/blob/master/examples/simulations/tunnelling1.py)<br>`tunnelling1`            | [![image](https://user-images.githubusercontent.com/32848391/96374887-dc4fdf80-1175-11eb-860a-e719558e7ed7.png)](https://github.com/marcomusy/vedo/blob/master/examples/advanced/thinplate_morphing_2d.py) <br>`morphing_2d`   |
| [![rabbits](https://user-images.githubusercontent.com/32848391/50738808-5816ad00-11d8-11e9-9854-c952be6fb941.jpg)](https://github.com/marcomusy/vedo/tree/master/examples/advanced/moving_least_squares2D.py) <br>`least_squares2d`| [![lut](https://user-images.githubusercontent.com/32848391/95255899-5c934e00-0822-11eb-9b07-fc3f31e2b6da.png)](https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_lut.py) <br> `mesh_lut`                                   | [![psimu](https://user-images.githubusercontent.com/32848391/50738891-db380300-11d8-11e9-84c2-0f55be7228f1.gif)](https://github.com/marcomusy/vedo/tree/master/examples/simulations/particle_simulator.py) <br>`particle_simulator` | [![gyro](https://user-images.githubusercontent.com/32848391/39766016-85c1c1d6-52e3-11e8-8575-d167b7ce5217.gif)](https://github.com/marcomusy/vedo/tree/master/examples/simulations/gyroscope1.py) <br> `gyroscope1`    |
| [![customaxes](https://user-images.githubusercontent.com/32848391/90966973-0750e680-e4d9-11ea-8e56-d75a1ad523dd.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/customAxes.py) <br>`customaxes`                | [![intersect](https://user-images.githubusercontent.com/32848391/90437548-de90b180-e0d2-11ea-8e0c-d821db4da8a9.png)](https://github.com/marcomusy/vedo/tree/master/examples/advanced/intersect2d.py) <br>`intersect2d`                     | [![goniom](https://user-images.githubusercontent.com/32848391/90437544-dd5f8480-e0d2-11ea-8321-b52d073444c4.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/goniometer.py) <br>`goniometer`                | [![](https://user-images.githubusercontent.com/32848391/82767103-2500a800-9e25-11ea-8506-e583e8ec4b01.jpg)](https://github.com/marcomusy/vedo/tree/master/examples/tetmesh/tet_threshold.py) <br>`tet_threshold`     |
| [![density](https://user-images.githubusercontent.com/32848391/90437537-dc2e5780-e0d2-11ea-982c-8dafd467c3cd.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_density3d.py) <br>`plot_density3d`           | [![mquality](https://user-images.githubusercontent.com/32848391/90976242-91319b80-e53b-11ea-8ff6-77ec78330657.png)](https://github.com/marcomusy/vedo/tree/master/examples/advanced/meshquality.py) <br>`meshquality`                      | [![levelterrain](https://user-images.githubusercontent.com/32848391/72433087-f00a8780-3798-11ea-9778-991f0abeca70.png)](https://github.com/marcomusy/vedo/tree/master/examples/basic/isolines.py) <br>`isolines`               | [![](https://user-images.githubusercontent.com/32848391/82767107-2631d500-9e25-11ea-967c-42558f98f721.jpg)](https://github.com/marcomusy/vedo/tree/master/examples/tetmesh/tet_cutMesh1.py) <br>`tet_cutmesh1`     |
| [![geologic](https://user-images.githubusercontent.com/32848391/81397531-d2867280-9127-11ea-8cc8-0effbbbebf2d.jpg)](https://github.com/marcomusy/vedo/tree/master/examples/advanced/geological_model.py) <br>`geological`          | [![multirender](https://user-images.githubusercontent.com/32848391/81459297-80813380-919f-11ea-89b1-39a305dd9897.png)](https://github.com/marcomusy/vedo/tree/master/examples/basic/multirenderers.py) <br>`multirender`                   | [![cartoony](https://user-images.githubusercontent.com/32848391/81459306-8840d800-919f-11ea-859e-d9c0b432e644.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/cartoony.py) <br>`cartoony`                  | [![](https://user-images.githubusercontent.com/32848391/81459343-b9210d00-919f-11ea-846c-152d62cba06e.png)](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/streamlines4.py) <br>`streamlines4`|
| [![graph1](https://user-images.githubusercontent.com/32848391/90437546-ddf81b00-e0d2-11ea-84d5-e4356a5c5f85.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/graph_network.py) <br>`graph_network`              | [![lineage_graph](https://user-images.githubusercontent.com/32848391/80291851-8152a800-8751-11ea-893e-4a0bb85397b1.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/graph_lineage.py) <br>`graph_lineage`               | [![siluette](https://user-images.githubusercontent.com/32848391/57179369-8e5df380-6e7d-11e9-99b4-3b1a120dd375.png)](https://github.com/marcomusy/vedo/tree/master/examples/basic/silhouette1.py) <br>`silhouette1`             | [![](https://user-images.githubusercontent.com/32848391/90298998-a734c180-de94-11ea-8dbe-bf68d451b1d6.png)](https://github.com/marcomusy/vedo/tree/master/examples/basic/silhouette2.py) <br>`silhouette2`        |
| [![gyro](https://user-images.githubusercontent.com/32848391/50738942-687b5780-11d9-11e9-97f0-72bbd63f7d6e.gif)](https://github.com/marcomusy/vedo/tree/master/examples/simulations/gyroscope2.py) <br>`gyroscope2`                 | [![thinplate_grid](https://user-images.githubusercontent.com/32848391/51433540-d188b380-1c4c-11e9-81e7-a1cf4642c54b.png)](https://github.com/marcomusy/vedo/tree/master/examples/advanced/thinplate_grid.py) <br>`thinplate_grid`          | [![trail](https://user-images.githubusercontent.com/32848391/58370826-4aee2680-7f0b-11e9-91e6-3120770cfede.gif)](https://github.com/marcomusy/vedo/tree/master/examples/simulations/trail.py) <br>`trail`                      | [![quadmorph](https://user-images.githubusercontent.com/32848391/96375928-5aaf8000-117c-11eb-83a9-bcae5c425877.png)](https://github.com/marcomusy/vedo/tree/master/examples/advanced/quadratic_morphing.py) <br>`quadratic_morphing`  |
| [![shrink](https://user-images.githubusercontent.com/32848391/46819143-41042280-cd83-11e8-9492-4f53679887fa.png)](https://github.com/marcomusy/vedo/tree/master/examples/basic/shrink.py) <br>`shrink`                             | [![mesh_custom](https://user-images.githubusercontent.com/32848391/51390972-20d9c180-1b31-11e9-955d-025f1ef24cb7.png)](https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_custom.py) <br>`mesh_custom`                      | [![spring](https://user-images.githubusercontent.com/32848391/50738955-7e891800-11d9-11e9-85cd-02bd4f3f13ea.gif)](https://github.com/marcomusy/vedo/tree/master/examples/simulations/aspring.py) <br>`spring`                  | [![lorenz](https://user-images.githubusercontent.com/32848391/46818115-be7a6380-cd80-11e8-8ffb-60af2631bf71.png)](https://github.com/marcomusy/vedo/tree/master/examples/basic/lorenz.py) <br>`lorentz`   |
| [![sliders](https://user-images.githubusercontent.com/32848391/50738848-be033480-11d8-11e9-9b1a-c13105423a79.jpg)](https://github.com/marcomusy/vedo/tree/master/examples/basic/sliders1.py) <br>`sliders1`                        | [![fitspheres1](https://user-images.githubusercontent.com/32848391/50738943-687b5780-11d9-11e9-87a6-054e0fe76241.jpg)](https://github.com/marcomusy/vedo/tree/master/examples/advanced/fitspheres1.py) <br>`fitspheres1`                   | [![fxy](https://user-images.githubusercontent.com/32848391/36611824-fd524fac-18d4-11e8-8c76-d3d1b1bb3954.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot4_fxy.py) <br>`fxy`                           | [![histogram](https://user-images.githubusercontent.com/32848391/68141260-77cc4e00-ff2d-11e9-9280-0efc5b87314d.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_1D.py) <br>`histo_1D`   |
| [![plot_err_band](https://user-images.githubusercontent.com/32848391/96375277-449fc080-1178-11eb-9a0f-3a4f9efe0d76.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot2_errband.py) <br>`plot2_errband`       | [![histogram2D](https://user-images.githubusercontent.com/32848391/72452359-b5671600-37bd-11ea-8b1d-c44d884496ed.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_2D.py) <br>`histo_2D`                           | [![histoHexagonal.py](https://user-images.githubusercontent.com/32848391/72434748-b471bc80-379c-11ea-95d7-d70333770582.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_hexagonal.py) <br>`histo_hex` | [![sphericPlot](https://user-images.githubusercontent.com/32848391/72433091-f0a31e00-3798-11ea-86bd-6c522e23ec61.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot5_spheric.py) <br>`plot5_spheric`    |
| [![boolean](https://user-images.githubusercontent.com/32848391/50738871-c0fe2500-11d8-11e9-8812-442b69be6db9.png)](https://github.com/marcomusy/vedo/tree/master/examples/basic/boolean.py) <br>`boolean`                          | [![brownian2D](https://user-images.githubusercontent.com/32848391/50738948-73ce8300-11d9-11e9-8ef6-fc4f64c4a9ce.gif)](https://github.com/marcomusy/vedo/tree/master/examples/simulations/brownian2D.py) <br>`brownian2D`                   | [![gas](https://user-images.githubusercontent.com/32848391/50738954-7e891800-11d9-11e9-95aa-67c92ca6476b.gif)](https://github.com/marcomusy/vedo/tree/master/examples/simulations/gas.py) <br>`gas`                            | [![self_org_maps2d](https://user-images.githubusercontent.com/32848391/54557310-1ade5080-49bb-11e9-9b97-1b53a7689a9b.gif)](https://github.com/marcomusy/vedo/tree/master/examples/other/self_org_maps2d.py)  <br>`self_org_maps2d`    |
| [![geodesic](https://user-images.githubusercontent.com/32848391/51855637-015f4780-232e-11e9-92ca-053a558e7f70.png)](https://github.com/marcomusy/vedo/tree/master/examples/advanced/geodesic.py) <br>`geodesic`                    | [![convexHull](https://user-images.githubusercontent.com/32848391/51932732-068cc700-2400-11e9-9b68-30294a4fa4e3.png)](https://github.com/marcomusy/vedo/tree/master/examples/advanced/convexHull.py)  <br>`convexHull`                     | [![flatarrow](https://user-images.githubusercontent.com/32848391/54612632-97c00780-4a59-11e9-8532-940c25a5dfd8.png)](https://github.com/marcomusy/vedo/tree/master/examples/basic/flatarrow.py) <br>`flatarrow`                | [![latex](https://user-images.githubusercontent.com/32848391/55568648-6190b200-5700-11e9-9547-0798c588a7a5.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/latex.py) <br>`latex`  |
| [![legosurface](https://user-images.githubusercontent.com/32848391/56820682-da40e500-684c-11e9-8ea3-91cbcba24b3a.png)](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/legosurface.py) <br>`legosurface`         | [![streamlines2](https://user-images.githubusercontent.com/32848391/56964001-9145a500-6b5a-11e9-935b-1b2425bd7dd2.png)](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/streamlines2.py) <br>`streamlines2`              | [![office](https://user-images.githubusercontent.com/32848391/56964003-9145a500-6b5a-11e9-9d9e-9736d90e1900.png)](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/office.py) <br>`office.py`                 | [![value-iteration](https://user-images.githubusercontent.com/32848391/56964055-afaba080-6b5a-11e9-99cf-3fac99df9878.jpg)](https://github.com/marcomusy/vedo/tree/master/examples/other/value-iteration.py)  <br>`value-iteration`  |
| [![shadow](https://user-images.githubusercontent.com/32848391/57312574-1d714280-70ee-11e9-8741-04fc5386d692.png)](https://github.com/marcomusy/vedo/tree/master/examples/basic/shadow.py) <br>`shadow`                             | [![multiple_pendulum](https://user-images.githubusercontent.com/32848391/50738892-db380300-11d8-11e9-807c-fb320c7b7917.gif)](https://github.com/marcomusy/vedo/tree/master/examples/simulations/multiple_pendulum.py ) <br>`multiple_pend` | [![](https://user-images.githubusercontent.com/32848391/59095175-1ec5a300-8918-11e9-8bc0-fd35c8981e2b.jpg)](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/interpolateVolume.py) <br>`interpolateVolume`    | [![histo_polar](https://user-images.githubusercontent.com/32848391/64912717-5754f400-d733-11e9-8a1f-612165955f23.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_polar.py)  <br>`histo_polar`  |
| [![streamplot](https://user-images.githubusercontent.com/32848391/73614123-93162a80-45fc-11ea-969b-9a3293b26f35.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot7_stream.py) <br>`plot7_stream`            | [![violin](https://user-images.githubusercontent.com/32848391/73481240-b55d3d80-439b-11ea-89a4-6c35ecc84b0d.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_violin.py) <br>`histo_violin`                        | [![plot3_pip](https://user-images.githubusercontent.com/32848391/73393632-4ff64780-42dc-11ea-8798-45a81c067f45.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot3_pip.py) <br>`plot3_pip`               | [![histo_spheric](https://user-images.githubusercontent.com/32848391/73392901-fccfc500-42da-11ea-828a-9bad6982a823.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_spheric.py)  <br>`histo_spheric`  |
| [![readvts](https://user-images.githubusercontent.com/32848391/80862655-04568f80-8c77-11ea-8249-5b61283e04ce.png)](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/read_vts.py) <br>`read_vts`                   | [![donutPlot](https://user-images.githubusercontent.com/32848391/64998178-6f6b7580-d8e3-11e9-9bd8-8dfb9ccd90e4.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/donut.py) <br>`donut`                                   | [![extrude](https://user-images.githubusercontent.com/32848391/65963682-971e1a00-e45b-11e9-9f29-05522ae4a800.png)](https://github.com/marcomusy/vedo/tree/master/examples/basic/extrude.py) <br>`extrude`                      | [![plotxy](https://user-images.githubusercontent.com/32848391/96375341-cb549d80-1178-11eb-868f-3e7d55d989ff.png)](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot1_errbars.py) <br>`plot1_errbars`   |
| [![isohead](https://user-images.githubusercontent.com/32848391/56972083-a7f3f800-6b6a-11e9-9cb3-1047b69dcad2.gif)](https://github.com/marcomusy/vedo/tree/master/examples)                                                         | [![viz_raycast](https://user-images.githubusercontent.com/32848391/58336919-f7b1a080-7e44-11e9-9106-f574371093a8.gif)](https://github.com/marcomusy/vedo/tree/master/examples)                                                             | [![viz_slicer](https://user-images.githubusercontent.com/32848391/80866479-3bd13600-8c8f-11ea-83c7-5f5b4fccb29d.png)](https://github.com/marcomusy/vedo/tree/master/examples/)                                                 | [![lego](https://user-images.githubusercontent.com/32848391/59788744-aaeaa980-92cc-11e9-825d-58da26ca21ff.gif)](https://github.com/marcomusy/vedo/tree/master/examples/)                                               |
| [![elastodyn](https://user-images.githubusercontent.com/32848391/54932788-bd4a8680-4f1b-11e9-9326-33645171a45e.gif)](https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/elastodynamics.py) <br> `elastodynamics` | [![heatconv](https://user-images.githubusercontent.com/32848391/57455107-b200af80-726a-11e9-897d-9c7bcb9854ac.gif)](https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/heatconv.py) <br>`heatconv`                        | [![stokes](https://user-images.githubusercontent.com/32848391/73683666-f36f9f80-46c2-11ea-9dca-2b559d2f458d.png)](https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/stokes.py) <br>`stokes`                  | [![lshape](https://user-images.githubusercontent.com/32848391/56671156-6bc91f00-66b4-11e9-8c58-e6b71e2ad1d0.gif)](https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/stokes_lshape.py) <br>`stokes_lshape`|

<br>


## 📜  References

**Presentations at interantional conferences:**

- M. Musy, G. Dalmasso, J. Sharpe and N. Sime, "`vedo`*: plotting in FEniCS with python*", ([link](https://github.com/marcomusy/vedo/blob/master/docs/fenics_poster.pdf)).
Poster at the [FEniCS'2019](https://fenicsproject.org/fenics19/) Conference,
Carnegie Institution for Science Department of Terrestrial Magnetism, Washington DC, June 2019.
- G. Dalmasso, *"Evolution in space and time of 3D volumetric images"*. Talk at the Conference for [Image-based Modeling and Simulation of Morphogenesis](https://www.pks.mpg.de/imsm19/).
Max Planck Institute for the Physics of Complex Systems, Dresden, Germany, March 2019.

**Scientific publications leveraging `vedo` (formerly known as `vtkplotter`):**

- X. Diego *et al.*:
*"Key features of Turing systems are determined purely by network topology"*,
[Physical Review X, 20 June 2018](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.021071).
- M. Musy, K. Flaherty *et al.*:
*"A Quantitative Method for Staging Mouse Limb Embryos based on Limb Morphometry"*,
Development, 5 April 2018, [doi: 10.1242/dev.154856](http://dev.biologists.org/content/145/7/dev154856).
- G. Dalmasso *et al.*, *"Evolution in space and time of 3D volumetric images"*, in preparation.
- F. Claudi, A. L. Tyson, T. Branco, *"Brainrender. A python based software for visualisation of neuroanatomical and morphological data."*
bioRxiv 2020.02.23.961748; doi: https://doi.org/10.1101/2020.02.23.961748

**Have you found this software useful for your research? Star ✨ the project and cite it as:**

M. Musy  <em>et al.</em>, "<code>vedo</code>, a python module for scientific visualization and analysis of 3D objects
and point clouds based on VTK (Visualization Toolkit)",
Zenodo, 10 February 2019, <a href="http://doi.org/10.5281/zenodo.2561402">doi: 10.5281/zenodo.2561402</a>.

[![embl_logo](https://user-images.githubusercontent.com/32848391/58046204-e9157180-7b44-11e9-81c9-e916cdf9ba84.gif)](https://www.embl.es)