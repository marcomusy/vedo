## Main changes

- fixed bug to `distanceTo()` method


---
### `base.py`

---
### `colors.py`

---
### `cli.py`

---
### `io.py`

- improved `Video.action(cameras=[...])` to interpolate cameras positions


---
### `mesh.py`

- faces of a mesh can be reversed individually with `reverse(face_list)`.
- fixed bug to `distanceTo()` method

---
### `plotter.py`
- fix `closeWindow()`
- add `show(..., zoom="tight")` to automatically zoom in as close as possible when 2D.

---
### `picture.py`

-- added `binarize()` method.
-- added `invert()` method.

---
### `pyplot.py`
- name change `cornerPlot` -> `CornerPlot`
- name change `cornerHistogram` -> `CornerHistogram`

---
### `pointcloud.py`

---
### `shapes.py`
- `Line.lineColor()` overrides `mesh.lineColor()` to avoid ambiguities.
- added `line.curvature()` method and example in docs.
- added `line.tangents()` method and example in docs.

---
### `volume.py`
- `volume.mesh2Volume()` function moved to `vedo.mesh.binarize()`
- `volume.signedDistanceFromPointCloud()` function moved to `Points.signedDistance`

---
### `utils.py`
- function has new keyword `sortByColumn(invert=False)`


-------------------------

## New/Revised examples:

`examples/pyplot/fourier_epicycles.py`
`examples/other/ellipt_fourier_desc.py`
`examples/volumetric/image_probe.py`
`examples/volumetric/image_mask.py`
`examples/pyplot/histo_3D.py`
`examples/other/napari1.py`
`examples/other/makeVideo.py`
`examples/volumetric/volumeFromMesh.py`
`examples/volumetric/mesh2volume.py`

## broken examples
```
legosurface.py
slicer1.py
vedo https://vedo.embl.es/examples/geo_scene.npz
```
