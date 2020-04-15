# Advanced examples
In this directory you will find a set of examples to perform more complex operations and simulations.
```bash
git clone https://github.com/marcomusy/vtkplotter.git
cd vtkplotter/vtkplotter/examples/advanced
python example.py
```
(_click thumbnail image to get to the python script_)

|    |    |
|:-------------:|:-----|
| [![convexhull](https://user-images.githubusercontent.com/32848391/51932732-068cc700-2400-11e9-9b68-30294a4fa4e3.png)](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced/convexHull.py)<br/> `convexHull.py`                      | Create a 3D Delaunay triangulation of a set of input points. |
|    |    |
| [![cutwithmesh](https://user-images.githubusercontent.com/32848391/51858160-4d14ef80-2334-11e9-8b9a-f76e1816966a.png)](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced/cutWithMesh.py)<br/> `cutWithMesh.py`                   | Cut a mesh with another mesh. |
|    |    |
| [![fatlimb](https://user-images.githubusercontent.com/32848391/50738945-7335ec80-11d9-11e9-9d3f-c6c19df8f10d.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced/fatlimb.py)<br/> `fatlimb.py`                               | Modify a mesh by moving the points along the normals to the surface and along the radius of a sphere centered at the center of mass of the mesh. At each step we redefine the actor so that the normals are recalculated for the underlying polydata.|
|    |    |
| [![fitline](https://user-images.githubusercontent.com/32848391/50738864-c0658e80-11d8-11e9-8754-c670f1f331d6.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced/fitline.py)<br/> `fitline.py`           | Draw a line in 3D that fits a cloud of points. <br/>The example also fits a plane to the same points.|
|    |    |
| [![fitspheres1](https://user-images.githubusercontent.com/32848391/50738943-687b5780-11d9-11e9-87a6-054e0fe76241.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced/fitspheres1.py)<br/> `fitspheres1.py`                   | Fit spheres to a region of a surface defined by N points that are closest to a given point of the surface. For some of these points we show the fitting sphere.<br/>Red lines join the center of the sphere to the surface point. <br/>Black points are the N points used for fitting. |
|    |    |
| [![geodesic](https://user-images.githubusercontent.com/32848391/51855637-015f4780-232e-11e9-92ca-053a558e7f70.png)](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced/geodesic.py)<br/> `geodesic.py`                            | Dijkstra algorithm to compute the mesh geodesic shortest path. |
|    |    |
| [![interpolateField](https://user-images.githubusercontent.com/32848391/52416117-25b6e300-2ae9-11e9-8d86-575b97e543c0.png)](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced/interpolateField.py)<br/> `interpolateField.py`    | Interpolate the value of a vectorial field only known on a small set of points to a whole region space where the field is not defined. <br>Two interpolation methods are shown: _Radial Basis Function_ and _Thin Plate Splines_.|
|    |    |
| [![interpolateScalar](https://user-images.githubusercontent.com/32848391/50738940-687b5780-11d9-11e9-9739-b084c5cfffaa.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced/interpolateScalar.py)<br/> `interpolateScalar.py` | Interpolate the value of a scalar only known on a set of points to a new set of points where the scalar is not defined. <br>Two interpolation methods are possible: _Radial Basis Function_ and _Nearest point_.|
|    |    |
| [![mesh_smoothers](https://user-images.githubusercontent.com/32848391/50738939-67e2c100-11d9-11e9-90cb-716ff3f03f67.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced/mesh_smoothers.py)<br/> `mesh_smoothers.py`          | Mesh smoothing with `smoothLaplacian()` and `smoothWSinc()` methods. |
|    |    |
| [![mls1](https://user-images.githubusercontent.com/32848391/50738937-61544980-11d9-11e9-8be8-8826032b8baf.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced/moving_least_squares1D.py)<br/> `moving_least_squares1D.py`    | Use a variant of the _Moving Least Squares_ algorithm for a cloud of scattered points to become a smooth line. The input actor's polydata is modified by the method so more than one pass is possible. |
|    |    |
| [![mls2](https://user-images.githubusercontent.com/32848391/50738936-61544980-11d9-11e9-9efb-e2a923762b72.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced/moving_least_squares2D.py)<br/> `moving_least_squares2D.py`    | Use a variant of the _Moving Least Squares_ algorithm for a cloud of points to become a smooth surface. The input actor's polydata is modified by the method so more than one pass is possible.|
|    |    |
| [![mls3](https://user-images.githubusercontent.com/32848391/50738935-61544980-11d9-11e9-9c20-f2ce944d2238.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced/moving_least_squares3D.py)<br/> `moving_least_squares3D.py`    | Generate a time sequence of 3D shapes (from a blue sphere to a red tetrahedron) as noisy cloud points, and smooth it with _Moving Least Squares_. This make a simultaneus fit in 4D (space+time). <br>`smoothMLS3D()` method returns an actor where points are color coded in bins of fitted time. |
|    |    |
| [![qmorph](https://user-images.githubusercontent.com/32848391/50738890-db380300-11d8-11e9-9cef-4c1276cca334.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced/quadratic_morphing.py)<br/> `quadratic_morphing.py`          | Takes two meshes, source and target, and morphs source to target. This is obtained by fitting 18 parameters of a non linear, quadratic, transformation defined in `transform()`. The fitting minimizes the distance to the target surface.|
|    |    |
| [![recosurface](https://user-images.githubusercontent.com/32848391/50738889-db380300-11d8-11e9-8854-2e3c70aefeb9.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced/recosurface.py)<br/> `recosurface.py`                   | Reconstruct a surface from a point cloud.<br>1. An mesh is loaded and noise is added to its vertices. <br>2. the point cloud is smoothened with MLS (see `moving_least_squares2D.py`) <br>3. `mesh.clean()` imposes a minimum distance among mesh points. <br>4. a triangular mesh is extracted from this smoother point cloud.|
|    |    |
| [![skeletonize](https://user-images.githubusercontent.com/32848391/50738888-db380300-11d8-11e9-86dd-742c1b887337.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced/skeletonize.py)<br/> `skeletonize.py`                   | Using 1D _Moving Least Squares_ to skeletonize a surface. |
|    |    |
| [![splitmesh](https://user-images.githubusercontent.com/32848391/52141626-97f57680-2656-11e9-80ea-fcd3571a6422.png)](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced/splitmesh.py)<br/> `splitmesh.py`                         | Split a mesh by connectivity and order the pieces by increasing value of the area. |
|    |    |
| [![thinplate](https://user-images.githubusercontent.com/32848391/51403917-34495480-1b52-11e9-956c-918c7805a9b5.png)](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced/thinplate.py)<br/> `thinplate.py`                         | Perform a nonlinear warp transformation defined by a set of source and target landmarks. |
|    |    |
| [![thinplate_grid](https://user-images.githubusercontent.com/32848391/51433540-d188b380-1c4c-11e9-81e7-a1cf4642c54b.png)](https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/examples/advanced/thinplate_grid.py)<br/> `thinplate_grid.py`          | Move a set of control points to warp space in their vicinity. Warping is visualised through a set of horizontal grids. |
