## Main changes

- `pep8` parsing with pylint and general cleanup of the code.
- `Volume(np_array)` does not need array to be transposed anymore. **Warning: this may cause flipping of the x-z axes!**

---
### `applications.py`
- added `SplinePlotter` object

---
### `base.py`

---
### `pointcloud.py`
- bug fix in `clone()`
- added `labels2D()`
- added logscale for colormapping in method `cmap(..., logscale=True)`
- added explicit label formatting for scalarbars, @XushanLu

---
### `mesh.py`

---
### `plotter.py`
- added `breakInteraction()` to break window interaction and return to the python execution flow

---
### `picture.py`
- added class `MatplotlibPicture` to embed matplotlib figures as 2d objects in the scene.

---
### `pyplot.py`

---
### `shapes.py`

- Added `text()` method to update 3d text on the fly, by @mkerrin
- Added `pcaEllipse()` analogous to `pcaEllipsoid()` for 2D problems.

---
### `tetmesh.py`


---
### `settings.py`

---
### `utils.py`
- added `getUV()` method to get the texture coordinates of a 3D point

---
### `volume.py`
- `Volume(np_array)` does not need array to be transposed anymore. **Warning: this may cause flipping of the x-z axes!**
- added `slicePlane(autocrop=False)`

-------------------------
## Examples

### New/Revised
examples/basic/input_box.py
examples/basic/pca_ellipse.py
examples/advanced/capping_mesh.py
examples/volumetric/numpy2volume2.py
examples/volumetric/numpy2volume1.py
examples/pyplot/fourier_epicycles.py
examples/pyplot/histo_pca.py
examples/pyplot/histo_1d_b.py
examples/other/flag_labels.py
examples/other/remesh_tetgen.py
examples/other/pymeshlab2.py

### Deleted

### Broken
examples/basic/multiwindows2.py
examples/simulations/lorenz.py
examples/simulations/orbitals.py





