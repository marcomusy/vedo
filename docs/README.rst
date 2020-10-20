.. role:: raw-html-m2r(raw)
   :format: html

.. image:: https://user-images.githubusercontent.com/32848391/90966721-1e420980-e4d6-11ea-998f-3285d512541f.png

:raw-html-m2r:`<br />`

.. image:: https://pepy.tech/badge/vtkplotter
   :target: https://pepy.tech/project/vtkplotter
   :alt: Downloads

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://en.wikipedia.org/wiki/MIT_License
   :alt: lics

.. image:: https://img.shields.io/badge/python-2.7%7C3.6-brightgreen.svg
   :target: https://pypi.org/project/vedo
   :alt: pythvers

.. image:: https://img.shields.io/badge/docs%20by-gendocs-blue.svg
   :target: https://gendocs.readthedocs.io/en/latest/
   :alt: Documentation Built by gendocs

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2561402.svg
   :target: https://doi.org/10.5281/zenodo.2561402

---------------------

`vedo` is a fast and lightweight python module
for scientific analysis and visualization of 3d objects.

Not limited to 3d, `vedo` can also generate high quality renderings in 2d (scatter plots, histograms etc).


Philosophy
----------

Inspired by the `vpython <https://vpython.org/>`_ *manifesto* "3D programming for ordinary mortals",
*vedo* makes it easy to work wth three-dimensional objects, create displays and animations
in just a few lines of code, even for those with less programming experience.

`vedo` is based on `VTK <https://www.vtk.org/>`_ and `numpy <http://www.numpy.org/>`_,
with no other dependencies.


Download and Install:
---------------------

.. code-block:: bash

   pip install vedo

Check out the **Git repository** here: https://github.com/marcomusy/vedo

*Windows-10 users* can manually place this file
`vedo.bat <https://github.com/marcomusy/vedo/blob/master/vedo.bat>`_
on the desktop to *drag&drop* files to visualize.
(Need to edit the path of their local python installation).


Features:
---------

The module includes a
`large set (300+) of working examples <https://github.com/marcomusy/vedo/tree/master/vedo/examples>`_
for the all following functionalities:

- Import meshes from VTK format, STL, Wavefront OBJ, 3DS, XML, Neutral, GMSH, PCD (PointCloud), volumetric TIFF stacks, SLC, MHD, 2D images PNG, JPEG.
- Export meshes as ASCII or binary to VTK, STL, PLY formats with command `vtkconvert`.
- Mesh analysis through the built-in methods of VTK package. Additional analysis tools like *Moving Least Squares*, mesh morphing.
- Tools to visualize and edit meshes (cutting a mesh with another mesh, slicing, normalizing, moving vertex positions, etc..). Interactive cutter widget.
- Split mesh based on surface connectivity. Extract the largest connected area.
- Calculate mass properties, like area, volume, center of mass, average size etc.
- Calculate vertex and face normals, curvatures, feature edges. Fill mesh holes.
- Subdivide faces of a mesh, increasing the number of vertex points. Mesh simplification.
- Coloring and thresholding of meshes based on associated scalar or vectorial data.
- Point-surface operations: find nearest points, check if a point lies inside or outside a mesh.
- Create primitive objects like: spheres, arrows, cubes, torus, ellipsoids...
- Generate *glyphs* (associating a mesh to each vertex of a source mesh).
- Create animations easily by just defining the position of the displayed objects in the 3D scene. Add trailing lines to moving objects automatically.
- Straightforward support for multiple `sync-ed` or independent renderers in  the same window.
- Registration (alignment) of meshes with different techniques.
- Mesh smoothing with `Laplacian` and `WindowedSinc` algorithms.
- Delaunay triangulation in 2D and 3D.
- Generate meshes by joining nearby lines in space.
- Find the closest path from one point to another, travelling along the edges of a mesh.
- Find the intersection of a mesh with a line (or with another mesh).
- Analysis of `Point Clouds`:

    - `Moving Least Squares` smoothing of 2D, 3D and 4D clouds
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
- Fully customizable axis style.
- Examples with `SHTools <https://shtools.oca.eu>`_ package for *spherical harmonics* expansion of a mesh shape.
- Integration with the *Qt5* framework.
- Draw `latex`-formatted formulas on the rending window.
- Export a 3D scene and embed it into a html page.
- Support for `FEniCS/dolfin <https://fenicsproject.org/>`_ package.
- Visualization of tensors.
- Embed the 3D rendering in a jupyter notebook with the *K3D* backend.
- Export a 3D scene and embed it into a `web page <https://vedo.embl.es/examples/fenics_elasticity.html>`_.
- Interoperability with the `trimesh <https://trimsh.org/>`_ library.


Hello World examples
--------------------
Use **command line** to un any of the built-in examples. In a terminal, type:

.. code-block:: bash

    vedo -r covid19

.. image:: https://user-images.githubusercontent.com/32848391/77330206-4824b380-6d1f-11ea-9bc3-e3aef970dcc2.gif


Visualize a file from web URL (or your dropbox!), type:

.. code-block:: bash

    vedo https://vedo.embl.es/examples/data/panther.stl.gz

.. image:: https://user-images.githubusercontent.com/32848391/91658688-c05f8400-eaca-11ea-9cb9-7c4896e116f9.png


Visualize a whole scene, type:

.. code-block:: bash

    vedo https://vedo.embl.es/examples/geo_scene.npy

.. image:: https://user-images.githubusercontent.com/32848391/81397531-d2867280-9127-11ea-8cc8-0effbbbebf2d.jpg


Visualize any polygonal mesh.
Voxel-data (*mhd, vti, slc, tiff, dicom*) files can also be visualized
with options `-g`, e.g.:

.. code-block:: bash

    vedo -g  https://vedo.embl.es/examples/data/embryo.slc
    # valid formats: [vtk,vtu,vts,vtp,vtm,ply,obj,stl,3ds,dolfin-xml,neutral,gmsh,
    #                 pcd,xyz,txt,byu,tif,off,slc,vti,mhd,DICOM,dem,nrrd,bmp,png,jpg]

.. image:: https://user-images.githubusercontent.com/32848391/58336107-5a09a180-7e43-11e9-8c4e-b50e4e95ae71.gif

To visualize multiple files or files time-sequences try `-n` or `-s` options. Use `-h` for the complete list of options.


Use a slider to control isosurfacing of a volume:

.. code-block:: bash

    vedo https://vedo.embl.es/examples/data/head.vti

.. image:: https://user-images.githubusercontent.com/32848391/56972083-a7f3f800-6b6a-11e9-9cb3-1047b69dcad2.gif


Visualize colorized voxels:

.. code-block:: bash

    vedo --lego https://vedo.embl.es/examples/data/embryo.tif

.. image:: https://user-images.githubusercontent.com/32848391/56969949-71b47980-6b66-11e9-8251-4bbdb275cb22.jpg



Examples
--------

Run any of the available scripts from with:

.. code-block:: bash

    vedo --list
    vedo -r covid19


More than 300 examples can be found in directories:

- `examples/basic <https://github.com/marcomusy/vedo/tree/master/examples/basic>`_
- `examples/advanced <https://github.com/marcomusy/vedo/tree/master/examples/advanced>`_
- `examples/volumetric <https://github.com/marcomusy/vedo/tree/master/examples/volumetric>`_
- `examples/simulations <https://github.com/marcomusy/vedo/tree/master/examples/simulations>`_
- `examples/tetmesh <https://github.com/marcomusy/vedo/tree/master/examples/tetmesh>`_
- `examples/other <https://github.com/marcomusy/vedo/tree/master/examples/other>`_
- `examples/other/dolfin <https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin>`_
- `examples/other/trimesh <https://github.com/marcomusy/vedo/tree/master/examples/other/trimesh>`_
- `examples/notebooks <https://github.com/marcomusy/vedo/blob/master/examples/notebooks>`_

Apply a *Moving Least Squares* algorithm to obtain a smooth surface from a to a
large cloud of scattered points in space
(`moving_least_squares2D.py <https://github.com/marcomusy/vedo/blob/master/examples/advanced/moving_least_squares2D.py>`_):

.. image:: https://user-images.githubusercontent.com/32848391/50738808-5816ad00-11d8-11e9-9854-c952be6fb941.jpg
   :target: https://github.com/marcomusy/vedo/blob/master/examples/advanced/moving_least_squares2D.py
   :alt: rabbit


Simulation of a gyroscope hanging from a spring
(`gyroscope1.py <https://github.com/marcomusy/vedo/blob/master/examples/simulations/gyroscope1.py>`_):

.. image:: https://user-images.githubusercontent.com/32848391/39766016-85c1c1d6-52e3-11e8-8575-d167b7ce5217.gif
   :target: https://github.com/marcomusy/vedo/blob/master/examples/simulations/gyroscope1.py
   :alt: gyro


Quantum-tunnelling effect integrating the Schroedinger equation with 4th order Runge-Kutta method.
The animation shows the evolution of a particle in a box hitting a sinusoidal potential barrier
(`tunnelling2.py <https://github.com/marcomusy/vedo/blob/master/examples/simulations/tunnelling2.py>`_):

.. image:: https://user-images.githubusercontent.com/32848391/47751431-06aae880-dc92-11e8-9fcf-6659123edbfa.gif
   :target: https://github.com/marcomusy/vedo/blob/master/examples/simulations/tunnelling2.py
   :alt: qsine2



Visualizing a Turing system of reaction-diffusion between two molecules
(`grayscott.py <https://github.com/marcomusy/vedo/blob/master/examples/simulations/grayscott.py>`_):

.. image:: https://user-images.githubusercontent.com/32848391/80291855-87e11f80-8751-11ea-9428-12e193a2a66e.gif
   :target: https://github.com/marcomusy/vedo/blob/master/examples/simulations/grayscott.py
   :alt: turing



Support for the `FEniCS/dolfin <https://fenicsproject.org/>`_ platform for visualization of PDE and
finite element solutions
(`see here <https://github.com/marcomusy/vedo/blob/master/examples/other/dolfin>`_.

.. image:: https://user-images.githubusercontent.com/32848391/58368591-8b3fab80-7eef-11e9-882f-8b8eaef43567.gif



Mesh format conversion
^^^^^^^^^^^^^^^^^^^^^^

The command ``vedo-convert`` can be used to convert multiple files from a format to a different one:

.. code-block:: bash

   Usage: vedo-convert [-h] [-to] [files [files ...]]
   allowed targets formats: [vtk, vtp, vtu, vts, ply, stl, byu, xml]

   Example: > vedo-convert myfile.vtk -to ply
