## Main changes

---
### `base.py`

---
### `addons.py`

---
### `colors.py`

---
### `cli.py`

New CLI mode to emulate `eog` for convenient image visualization:

`vedo --eog https://corepetfood.com/files/2019/07/kitten-e1568311742288-1440x900.jpg`
```
    printc('Press:')
    printc('  up/down     to modify level (or drag mouse)')
    printc('  left/right  to modify window')
    printc('  m           to mirror image')
    printc('  t           to rotate image by 90 deg')
    printc('  k           to enhance b&w image')
    printc('  s           to apply gaussian smoothing')
    printc('  S           to save image as png')
```

---
### `mesh.py`

---
### `plotter.py`

- added `enableErase()` `enableRenderer()` `useDepthPeeling(at)` methods

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

---
### `volume.py`

---
### `utils.py`


---
### `cli.py`

-------------------------

## New/Revised examples:
`examples/basic/shadow2.py`


