## Main changes

---
### `base.py`

---
### `addons.py`

---
### `colors.py`

---
### `cli.py`

New CLI mode to emulate `eog` for convenient visualization of common images :

`vedo --eog https://corepetfood.com/files/2019/07/kitten-e1568311742288-1440x900.jpg`
```
Press:
  up/down     to modify level (or drag mouse)
  left/right  to modify window
  m           to mirror image
  t           to rotate image by 90 deg
  k           to enhance b&w image
  s           to apply gaussian smoothing
  S           to save image as png
  h           to print this help banner
```

---
### `mesh.py`

---
### `plotter.py`

- added `enableErase()` `enableRenderer()` `useDepthPeeling(at)` methods
- added `addScaleIndicator()` to add to the scene an indicator of absolute size of objects
(needs `settings.useParallelProjection = True`)

---
### `picture.py`

---
### `pyplot.py`

---
### `pointcloud.py`

- added `smoothLloyd2D()` for smoothing pointclouds in 2D
- vtkCellLocator seems to have a problem with single cell meshes (#558), fixed using vtkStaticCellLocator
which behaves normally

---
### `shapes.py`

- added `Line().pattern()` to create a dashed line with a user defined pattern.
- fixed bug in Text2D()

---
### `volume.py`

---
### `utils.py`


---
### `cli.py`

-------------------------

## New/Revised examples:
`examples/basic/shadow2.py`


