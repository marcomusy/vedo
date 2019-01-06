# Advanced examples
In this directory you will find a set of examples to perform more complex operations and simulations.
```bash
git clone https://github.com/marcomusy/vtkplotter.git
cd vtkplotter/examples/advanced
python example.py  # on mac OSX try 'pythonw' instead
```
(_click thumbnail image to get to the python script_)

|    |    |
|:-------------:|:-----|
| [![aspring](https://user-images.githubusercontent.com/32848391/50738955-7e891800-11d9-11e9-85cd-02bd4f3f13ea.gif)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/aspring.py)<br/> `aspring.py` |  Simulation of a spring in a viscuos medium. |
|    |    |
| [![blackbody](https://user-images.githubusercontent.com/32848391/50738949-73ce8300-11d9-11e9-87bd-056ba8a6232e.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/blackbody.py)<br/> `blackbody.py` | Black body intensity radiation for different temperatures in the range (3000K, 9000K) for the visible range of wavelenghts (400nm, 700nm). Colors are fairly well matched to the "jet" and "rainbow" maps in `pointColors()` method.|
|    |    |
| [![brownian2d](https://user-images.githubusercontent.com/32848391/50738948-73ce8300-11d9-11e9-8ef6-fc4f64c4a9ce.gif)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/brownian2D.py)<br/> `brownian2D.py` | Simulation of the Brownian motion of a large red particle in a swarm of small particles. <br/>The spheres collide elastically with themselves and with the walls of the box.|
|    |    |
| [![cell_main](https://user-images.githubusercontent.com/32848391/50738950-73ce8300-11d9-11e9-9d9d-960a032e0aae.gif)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/cell_main.py)<br/> `cell_main.py` |  Simulation of three bacteria types that divide at a specified rate. As they divide they occupy more and more space. |
|    |    |
| [![doubleslit](https://user-images.githubusercontent.com/32848391/50738946-7335ec80-11d9-11e9-93db-f34f853ed759.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/doubleslit.py)<br/> `doubleslit.py` | Simulation of the classic *double slit experiment*. <br/>Any number of slits of any geometry can be described. Slit sources are placed on the plane shown as a thin grid. <br/>Can simulate the [*Arago spot*](https://en.wikipedia.org/wiki/Arago_spot), the bright point at the center of a circular object shadow.|
|    |    |
| [![fatlimb](https://user-images.githubusercontent.com/32848391/50738945-7335ec80-11d9-11e9-9d3f-c6c19df8f10d.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fatlimb.py)<br/> `fatlimb.py` | Modify a mesh by moving the points along the normals to the surface and along the radius of a sphere centered at the center of mass of the mesh. At each step we redefine the actor so that the normals are recalculated for the underlying polydata.|
|    |    |
| [![fitspheres1](https://user-images.githubusercontent.com/32848391/50738943-687b5780-11d9-11e9-87a6-054e0fe76241.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fitspheres1.py)<br/> `fitspheres1.py` | Fit spheres to a region of a surface defined by N points that are closest to a given point of the surface. For some of these points we show the fitting sphere.<br/>Red lines join the center of the sphere to the surface point. <br/>Black points are the N points used for fitting. |
|    |    |
| [![gas](https://user-images.githubusercontent.com/32848391/50738954-7e891800-11d9-11e9-95aa-67c92ca6476b.gif)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/gas.py)<br/> `gas.py` | A model of an ideal gas with hard-sphere collisions.|
|    |    |
| [![gyro2](https://user-images.githubusercontent.com/32848391/50738942-687b5780-11d9-11e9-97f0-72bbd63f7d6e.gif)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/gyro2.py)<br/> `gyroscope2.py` | Gyroscope sitting on a pedestal at one end. <br/>The analysis is made in terms of Lagrangian mechanics. |
|    |    |
| [![interactor](https://user-images.githubusercontent.com/32848391/50738941-687b5780-11d9-11e9-8b00-7af4d1d93027.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/interactor.py)<br/> `interactor.py` | How to keep interacting with the 3D scene while the program is running (by rotating 3D scene with mouse and remain responsive to key press). While using the mouse, calls to custom loop function are suspended.|
|    |    |
| [![interpolate](https://user-images.githubusercontent.com/32848391/50738940-687b5780-11d9-11e9-9739-b084c5cfffaa.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/interpolate.py)<br/> `interpolate.py` | Interpolate the value of a scalar only known on a set of points to a new set of points where the scalar is not defined. <br>Two interpolation methods are possible: *Radial Basis Function* and *Nearest point*.|
|    |    |
| [![mesh_smoothers](https://user-images.githubusercontent.com/32848391/50738939-67e2c100-11d9-11e9-90cb-716ff3f03f67.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/mesh_smoothers.py)<br/> `mesh_smoothers.py` | Mesh smoothing with `smoothLaplacian()` and `smoothWSinc()` methods. |
|    |    |
| [![mls1](https://user-images.githubusercontent.com/32848391/50738937-61544980-11d9-11e9-8be8-8826032b8baf.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/moving_least_squares1D.py)<br/> `moving_least_squares1D.py` | Use a variant of the *Moving Least Squares* algorithm for a cloud of scattered points to become a smooth line. The input actor's polydata is modified by the method so more than one pass is possible. |
|    |    |
| [![mls2](https://user-images.githubusercontent.com/32848391/50738936-61544980-11d9-11e9-9efb-e2a923762b72.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/moving_least_squares2D.py)<br/> `moving_least_squares2D.py` | Use a variant of the *Moving Least Squares* algorithm for a cloud of points to become a smooth surface. The input actor's polydata is modified by the method so more than one pass is possible.|
|    |    |
| [![mls3](https://user-images.githubusercontent.com/32848391/50738935-61544980-11d9-11e9-9c20-f2ce944d2238.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/moving_least_squares3D.py)<br/> `moving_least_squares3D.py` | Generate a time sequence of 3D shapes (from a sphere to a tetrahedron) as noisy cloud points, and smooth it with *Moving Least Squares*. This make a simultaneus fit in 4D (space+time). <br>`smoothMLS3D()` method returns an actor where points are color coded in bins of fitted time. |
|    |    |
| [![mpend](https://user-images.githubusercontent.com/32848391/50738892-db380300-11d8-11e9-807c-fb320c7b7917.gif)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/multiple_pendulum.py)<br/> `multiple_pendulum.py` | Simulation of an elastic multiple pendulum with viscuos friction. |
|    |    |
| [![ruth](https://user-images.githubusercontent.com/32848391/50738891-db380300-11d8-11e9-84c2-0f55be7228f1.gif)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/particle_simulator.py)<br/> `particle_simulator.py` | Simulates interacting charged particles in 3D space. |
|    |    |
| [![qmorph](https://user-images.githubusercontent.com/32848391/50738890-db380300-11d8-11e9-9cef-4c1276cca334.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/quadratic_morphing.py)<br/> `quadratic_morphing.py` | Takes 2 shapes, source and target, and morphs source on target. This is obtained by fitting 18 parameters of a non linear, quadratic, transformation defined in `transform()`. The fitting minimizes the distance to the target surface.|
|    |    |
| [![recosurface](https://user-images.githubusercontent.com/32848391/50738889-db380300-11d8-11e9-8854-2e3c70aefeb9.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/recosurface.py)<br/> `recosurface.py` | Reconstruct a surface from a point cloud.<br>1. An mesh is loaded and noise is added to its vertices. <br>2. the point cloud is smoothened with MLS (see `moving_least_squares2D.py`) <br>3. `mesh.clean()` imposes a minimum distance among mesh points where 'tol' is the fraction of the actor size. <br>4. a triangular mesh is extracted from this set of sparse points `bins'`is the number of voxels of the subdivision.|
|    |    |
| [![skeletonize](https://user-images.githubusercontent.com/32848391/50738888-db380300-11d8-11e9-86dd-742c1b887337.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/skeletonize.py)<br/> `skeletonize.py` | Using 1D *Moving Least Squares* to skeletonize a surface. |
|    |    |
| [![tunneling1](https://vtkplotter.embl.es/gifs/tunnelling2.gif)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/tunnelling1.py)<br/> `tunnelling1.py` | Quantum Tunnelling effect using 4th order Runge-Kutta method with arbitrary potential shape. <br>The animation shows the evolution of a particle of well defined momentum (hence undefined position) in a box hitting a potential barrier. The wave function is forced to be zero at the box walls. |
|    |    |
| [![turing](https://user-images.githubusercontent.com/32848391/50738887-da9f6c80-11d8-11e9-83a6-fb002c0613bd.gif)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/turing.py)<br/> `turing.py` | Scalar values are read from a file and represented on a green scale on a mesh as a function of time. The difference between one time point and the next is shown as a blue component.|
|    |    |
| [![wave](https://user-images.githubusercontent.com/32848391/50738956-7e891800-11d9-11e9-92d7-fa109b1b8551.gif)](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/turing.py)<br/> `wave.py` | Simulate a discrete collection of oscillators. We will use this as a model of a vibrating string and compare two methods of integration: Euler and Runge-Kutta4.<br>|



