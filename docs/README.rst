.. role:: raw-html-m2r(raw)
   :format: html

.. image:: https://user-images.githubusercontent.com/32848391/52522718-50d83880-2c89-11e9-80ff-df1b5618a84a.png

:raw-html-m2r:`<br />`

.. image:: https://pepy.tech/badge/vtkplotter
   :target: https://pepy.tech/project/vtkplotter
   :alt: Downloads

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://en.wikipedia.org/wiki/MIT_License
   :alt: lics

.. image:: https://img.shields.io/badge/python-2.7%7C3.6-brightgreen.svg
   :target: https://pypi.org/project/vtkplotter
   :alt: pythvers

.. image:: https://img.shields.io/badge/docs%20by-gendocs-blue.svg
   :target: https://gendocs.readthedocs.io/en/latest/
   :alt: Documentation Built by gendocs
   
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2561402.svg
   :target: https://doi.org/10.5281/zenodo.2561402
   
---------------------

A python module for scientific visualization, analysis and animation of 3D objects 
and `point clouds` based on `VTK <https://www.vtk.org/>`_
and `numpy <http://www.numpy.org/>`_.


Download and Install:
---------------------

.. code-block:: bash

   pip install vtkplotter

Check out the **Git repository** here: https://github.com/marcomusy/vtkplotter



Features:
---------


Intuitive and straightforward API which can be combined with VTK seamlessly 
in a program, whilst mantaining access to the full range of VTK native classes.

It includes a 
`large set of working examples <https://github.com/marcomusy/vtkplotter/tree/master/examples>`_ 
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
- Examples with `SHTools <https://shtools.oca.eu/shtools>`_ package for *spherical harmonics* expansion of a mesh shape.
- Integration with the *Qt5* framework.


Hello World example
-------------------

In your python script, load a simple ``3DS`` file and display it:

.. code-block:: python

    from vtkplotter import show
    
    show('data/shapes/flamingo.3ds') 

.. image:: https://user-images.githubusercontent.com/32848391/50738813-58af4380-11d8-11e9-84ce-53579c1dba65.png
   :alt: flam

Allowed input objects to the ``show()`` command are: \ :raw-html-m2r:`<br>`
``filename``, ``vtkPolyData``, ``vtkActor``, 
``vtkActor2D``, ``vtkImageActor``, ``vtkAssembly`` or ``vtkVolume``.

Supported ``filename`` extensions are: \ :raw-html-m2r:`<br>`
`vtk, vtu, vts, vtp, ply, obj, stl, 3ds, xml, neutral, gmsh, pcd, xyz, txt, byu,
tif, slc, vti, mhd, png, jpg`.



Command-line usage
------------------

.. code-block:: bash

    vtkplotter data/shapes/flamingo.3ds

to visualize multiple files or files time-sequences try ``-n`` or ``-s`` options:

.. code-block:: bash

    vtkplotter -s data/timecourse1d/*vtk
    # or
    vtkplotter -n data/timecourse1d/*vtk

Try ``-h`` for help.\ :raw-html-m2r:`<br>`

Voxel-data (`vti, slc, mhd, tif`) files can also be visualized 
with options ``-g`` or ``--slicer``, e.g.:

.. code-block:: bash

    vtkplotter            examples/data/embryo.tif  # shows a 3D scan of a mouse embryo
    vtkplotter -g -c blue examples/data/embryo.slc  #  with sliders to control isosurfacing
    vtkplotter --slicer   examples/data/embryo.slc 

.. image:: https://user-images.githubusercontent.com/32848391/50738810-58af4380-11d8-11e9-8fc7-6c6959207224.jpg
   :target: https://user-images.githubusercontent.com/32848391/50738810-58af4380-11d8-11e9-8fc7-6c6959207224.jpg
   :alt: e2



Examples Gallery
----------------

A get-started `tutorial <https://github.com/marcomusy/vtkplotter/blob/master/examples>`_ 
script is available for download:

.. code-block:: bash

    git clone https://github.com/marcomusy/vtkplotter.git
    cd vtkplotter/examples
    python tutorial.py

More than 100 examples can be found in directories:

- `examples/basic <https://github.com/marcomusy/vtkplotter/blob/master/examples/basic>`_ ,
- `examples/advanced <https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced>`_ ,
- `examples/volumetric <https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric>`_,
- `examples/simulations <https://github.com/marcomusy/vtkplotter/blob/master/examples/simulations>`_,
- `examples/others <https://github.com/marcomusy/vtkplotter/blob/master/examples/other>`_.


Apply a *Moving Least Squares* algorithm to obtain a smooth surface from a to a
large cloud of scattered points in space 
(`moving_least_squares2D.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/moving_least_squares2D.py>`_):

.. image:: https://user-images.githubusercontent.com/32848391/50738808-5816ad00-11d8-11e9-9854-c952be6fb941.jpg
   :target: https://user-images.githubusercontent.com/32848391/50738808-5816ad00-11d8-11e9-9854-c952be6fb941.jpg
   :alt: rabbit


Simulation of a gyroscope hanging from a spring 
(`gyroscope1.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/simulations/gyroscope1.py>`_):

.. image:: https://user-images.githubusercontent.com/32848391/39766016-85c1c1d6-52e3-11e8-8575-d167b7ce5217.gif
   :target: https://user-images.githubusercontent.com/32848391/39766016-85c1c1d6-52e3-11e8-8575-d167b7ce5217.gif
   :alt: gyro


Simulation of `Rutherford scattering <https://en.wikipedia.org/wiki/Rutherford_scattering>`_ 
of charged particles on a fixed target 
(`particle_simulator.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/simulations/particle_simulator.py>`_):

.. image:: https://user-images.githubusercontent.com/32848391/43984362-5c545a0e-9d00-11e8-8ce5-572b96bb91d1.gif
   :target: https://user-images.githubusercontent.com/32848391/43984362-5c545a0e-9d00-11e8-8ce5-572b96bb91d1.gif
   :alt: ruth


Quantum-tunnelling effect integrating the Schroedinger equation with 4th order Runge-Kutta method. 
The animation shows the evolution of a particle in a box hitting a sinusoidal potential barrier
(`tunnelling2.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/simulations/tunnelling2.py>`_):

.. image:: https://user-images.githubusercontent.com/32848391/47751431-06aae880-dc92-11e8-9fcf-6659123edbfa.gif
   :target: https://user-images.githubusercontent.com/32848391/47751431-06aae880-dc92-11e8-9fcf-6659123edbfa.gif
   :alt: qsine2



Visualizing a Turing system of reaction-diffusion between two molecules
(`turing.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/simulations/turing.py>`_):

.. image:: https://user-images.githubusercontent.com/32848391/40665257-1412a30e-635d-11e8-9536-4c73bf6bdd92.gif
   :target: https://user-images.githubusercontent.com/32848391/40665257-1412a30e-635d-11e8-9536-4c73bf6bdd92.gif
   :alt: turing



Some useful ``Plotter`` attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Remember that you always have full access to all standard VTK native objects
(e.g. `vtkRenderWindowInteractor`, `vtkRenderer` and `vtkActor` through `vp.interactor`,
`vp.renderer`, `vp.actors`... etc).

.. code-block:: python

   vp = vtkplotter.Plotter() #e.g.
   vp.actors       # holds the current list of vtkActors to be shown
   vp.renderer     # holds the current vtkRenderer
   vp.renderers    # holds the list of renderers
   vp.interactor   # holds the vtkWindowInteractor object
   vp.interactive  # (True) allows to interact with renderer after show()
   vp.camera       # holds the current vtkCamera
   vp.sharecam     # (True) share the same camera in multiple renderers


Frequently used methods to manage 3D objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These methods return the ``Actor(vtkActor)`` object so that they can be concatenated,
check out ``Actor`` `methods here <https://vtkplotter.embl.es/actors.m.html>`_. :raw-html-m2r:`<br />`
(E.g.: ``actor.scale(3).pos([1,2,3]).color('blue').alpha(0.5)`` etc..).

.. code-block:: python

   actor.pos()               # set/get position vector (setters, and getters if no argument is given)
   actor.scale()             # set/get scaling factor of actor
   actor.normalize()         # sets actor at origin and scales its average size to 1
   actor.rotate(angle, axis) # rotate actor around axis
   actor.color(name)         # sets/gets color
   actor.alpha(value)        # sets/gets opacity
   actor.N()                 # get number of vertex points defining the actor's mesh
   actor.polydata()          # get the actor's mesh polydata in its current transformation
   actor.coordinates()       # get a copy of vertex points coordinates (copy=False to get references)
   actor.normals()           # get the list of normals at the vertices of the surface
   actor.clone()             # get a copy of actor
   ...


Mesh format conversion
^^^^^^^^^^^^^^^^^^^^^^

The command ``vtkconvert`` can be used to convert multiple files from a format to a different one:

.. code-block:: bash

   Usage: vtkconvert [-h] [-to] [files [files ...]]
   allowed targets formats: [vtk, vtp, vtu, vts, ply, stl, byu, xml]

   Example: > vtkconvert myfile.vtk -to ply

Available color maps from ``matplotlib`` and ``vtkNamedColors``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Example: transform a scalar value between -10.2 and 123 into a (R,G,B) color using the 'jet' map:
   from vtkplotter import colorMap
   r, g, b = colorMap(value, name='jet', vmin=-10.2, vmax=123)


.. image:: https://user-images.githubusercontent.com/32848391/50738804-577e1680-11d8-11e9-929e-fca17a8ac6f3.jpg
   :target: https://user-images.githubusercontent.com/32848391/50738804-577e1680-11d8-11e9-929e-fca17a8ac6f3.jpg
   :alt: colormaps


A list of available `vtk color names is given here <https://vtkplotter.embl.es/vtkcolors.html>`_.
:raw-html-m2r:`<br />`


