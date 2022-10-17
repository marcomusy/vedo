## Main changes

- NOTE: Convention has moved from `camelCase` to `snake_case`.

E.g.
```python
mesh = Mesh("bunny.obj")
mesh.cutWithPlane()
show(mesh)
```

now becomes

```python
mesh = Mesh("bunny.obj")
mesh.cut_with_plane()     ### <--
show(mesh)
```

Some backward incompatibility may occur!


- removed requirement on vtk version

---
### `addons.py`
- improved slider callback interface

---
### `applications.py`

---
### `base.py`

---
### `pointcloud.py`

---
### `mesh.py`
- fixed issue #593

---
### `plotter.py`

---
### `picture.py`

---
### `pyplot.py`

---
### `shapes.py`

---
### `tetmesh.py`


---
### `settings.py`

---
### `utils.py`

---
### `volume.py`
- can warp scalars in a volumetric dataset with `warp()`

-------------------------
## Examples

### New/Revised
examples/pyplot/fill_gap.py
examples/basic/sliders_hsv.py
examples/volumetric/slicer2.py
examples/volumetric/warp_scalars.py
examples/other/qt_window3.py


### Deleted

### Broken
examples/basic/multiwindows2.py
examples/simulations/lorenz.py
examples/simulations/orbitals.py
scatter2.py

/home/musy/Projects/vedo/examples/volumetric/image_probe.py
/home/musy/Projects/vedo/examples/volumetric/volumeOperations.py
/home/musy/Projects/vedo/examples/other/meshio_read.py
/home/musy/Projects/vedo/examples/other/dolfin/ex06_elasticity3.py
untitled5.py
untitled6.py

alias pylint='pylint -d C0103 -d C0321 -d E1101 -d R0913 -d C0301 -d R0914 -d R0912 -d R0915 -d W1203 -d R1705 -d C0302 -d R0902 '



