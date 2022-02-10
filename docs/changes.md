## Main changes

- Internal global variable `settings.plotter_instance` must become `vedo.plotter_instance`.

- `vedo.settings` is now a dotted dictionary

- implemented logging module for managing errors and warnings

- fixed bug to `distanceTo()` method


---
### `applications.py`
- `Brower`, `IsosurfaceBrowser`, `Slicer2DPlotter` and `RayCastPlotter` are now `Plotter` derived classes (not functions)
- improved `IsosurfaceBrowser` for speed

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
- `legosurface()` changed interface

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
interpolateMeshArray.py export cannot pickle 'Spline' object, phong is not preserved
```
