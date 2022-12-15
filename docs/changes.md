## Main changes

---
### `addons.py`

---
### `applications.py`

---
### `base.py`
- added `basegrid.extract_cells_on_plane()`
- added `basegrid.extract_cells_on_sphere()`
- added `basegrid.extract_cells_on_cylinder()`
- added `count_vertices()` method
- added `copy_data_from()` method to transfer all cell and point data from an object to another.
- fixed `metadata` association

---
### `pointcloud.py`
- added `cut_with_planes()` to cut at once with multiple planes
- added `generate_random_data()` for testing purposes
- renamed `vignette()` to `flagpole()`
- added new `flagpost()` similar to the above
- added new property `pointcloud.cellcolors`and `pointcolors`
to access and modify colors by RGBA assignment.
- added `compute_camera_distance()` to calculate the distance from points to the camera.
  A pointdata array is created with name 'DistanceToCamera'.
- added `cut_with_scalars()` to cut polygonal data with some input scalar data.


---
### `mesh.py`
- improved `binarize()` method thanks to @vfmatzkin
- added `collide_with()` to fix crashing of `intersect_with()` in special cases
- added `check_validity()`
- added method=4 for `subdivide()`
- added `intersect_with_plane()`
- added `boolean(..., method=1)`
- added `intersect_with_multiplanes()` to generate a set of lines from cutting a mesh in n intervals
  between a minimum and maximum distance from a plane of given origin and normal.

---
### `plotter.py`
- fixed key bindings for uppercase hits.
- added method `add_hint()` to generate a pop-up message on hovering an object
- fix output of template camera when pressing "C"
- improved `move_camera()`
- added key-press `R` to reset camera viewup to closest orthogonal viewup
  added corresponding method `plotter.reset_viewup()`
- added press `.` to fly to last clicked point and zoom in

---
### `picture.py`
-  generate false colors for an image with Picture().cmap()

---
### `pyplot.py`
- clicking on a histogram shows the bin value
- add `as2d()` method to freeze a plot in 2d canvas, without mouse interaction (experimental)

---
### `shapes.py`
- Added support for chinese and japanese chars
- Added font "ComicMono"
- added possibility to create a disc sector in `Disc(angle_range=...)`

---
### `tetmesh.py`
- added `compute_tets_volume()`
- added `check_validity()`


---
### `settings.py`

---
### `utils.py`

---
### `volume.py`
- added `vtkFlyingEdges3D` instead of contouring, which is faster.


-------------------------
## Examples

### New/Revised
examples/basic/color_mesh_cells1.py
examples/basic/color_mesh_cells2.py
examples/pyplot/fit_curve.py
examples/pyplot/histo_2d_a.py
examples/pyplot/histo_2d_b.py
examples/other/flag_labels2.py
examples/volumetric/image_false_colors.py

### Broken
examples/simulations/lorenz.py
examples/simulations/orbitals.py
examples/other/meshio_read.py
examples/other/dolfin/ex06_elasticity3.py
-------------------------
vtk9.2:
interactionstyle.py
cutter.py
histo_manual.py
tet_explode.py
tet_isos_slice.py # stuck
makeVideo.py
lorenz.py



