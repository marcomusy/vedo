## Main changes

- pep8 parsing with pylint and general cleanup of the code.
- `Volume(np_array)` does not need array to be transposed anymore. **Warning: this may cause flipping of the x-z axes!**

---
### `applications.py`
- added `SplinePlotter` object

---
### `base.py`

---
### `pointcloud.py`
- bug fix in `clone()`

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
### `volume.py`
- `Volume(np_array)` does not need array to be transposed anymore. **Warning: this may cause flipping of the x-z axes!**
- added `slicePlane(autocrop=False)`

-------------------------
## Examples

### New/Revised

### Deleted

### Broken
examples/basic/multiwindows2.py
examples/simulations/lorenz.py
examples/basic/pca_ellipse.py











