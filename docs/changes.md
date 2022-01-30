## Main changes

---
### `base.py`

---
### `colors.py`

---
### `cli.py`


---
### `mesh.py`

- faces of a mesh can be reversed individually with `reverse(face_list)`.


---
### `plotter.py`
- fix `closeWindow()`

---
### `picture.py`

-- added `binarize()` method.
-- added `invert()` method.

---
### `pyplot.py`

---
### `pointcloud.py`

---
### `shapes.py`
- `Line.lineColor()` overrides `mesh.lineColor()` to avoid ambiguities.
- added `line.curvature()` method and example in docs.
- added `line.tangents()` method and example in docs.

---
### `volume.py`

---
### `utils.py`

-  function has new keyword `sortByColumn(invert=False)`


-------------------------

## New/Revised examples:

`examples/pyplot/fourier_epicycles.py`
`examples/other/ellipt_fourier_desc.py`
`examples/volumetric/image_probe.py`
`examples/volumetric/image_mask.py`
`examples/pyplot/histo_3D.py`
`examples/other/napari1.py`


## broken examples
```
cartoony.py
legendbox.py

makeVideo.py
pymeshlab1.py
pymeshlab2.py

explore5d.py
scatter_large.py

cell_colony.py
hanoi3d.py

legosurface.py
slicer1.py
vedo https://vedo.embl.es/examples/geo_scene.npz
```
