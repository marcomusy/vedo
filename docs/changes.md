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


---
### `pointcloud.py`
- added `cut_with_planes()` to cut at once with multiple planes
- added `generate_random_data()` for testing purposes

---
### `mesh.py`
- improved `binarize()` method thanks to @vfmatzkin
- added `collide_with()` to fix crashing of `intersect_with()` in special cases

---
### `plotter.py`
- fixed key bindings for uppercase hits.
- added method `add_hint()` to generate a pop-up message on hovering an object
- fix output of template camera when pressing "C"

---
### `picture.py`
-  generate false colors for an image with Picture().cmap()

---
### `pyplot.py`
- clicking on a histogram shows the bin value

---
### `shapes.py`
- Added support for chinese and japanese chars
- Added font "ComicMono"

---
### `tetmesh.py`


---
### `settings.py`

---
### `utils.py`

---
### `volume.py`

-------------------------
## Examples

### New/Revised
examples/pyplot/fit_curve.py
examples/pyplot/histo_2d_a.py
examples/pyplot/histo_2d_b.py

-------------------------
### Deleted

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

plot_errbars.py



