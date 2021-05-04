## Main changes


- general improvements to the `vedo` command line
- `colorcet` [colormaps](https://colorcet.holoviz.org) are directly usable in `vedo`.
Check example `basic/mesh_custom.py`

- general tool for free-hand cutting a mesh can be invoked from command line:
`vedo --edit https://vedo.embl.es/examples/data/porsche.ply`

- added example search by command line e.g.:
`vedo --search triangle -i`  (`-i` to ignore capital letters)

- added file(s) info dump by command line e.g.:
`vedo --info some_image.jpg https://vedo.embl.es/examples/data/porsche.ply`

- pressing `shift-A` toggles antialiasing for the whole rendering window
- pressing `shift-D` toggles depth-peeling algorithm for the current renderer

- antialiasing set by default on vtk9
- balloon-style flags are disabled  because of a bug in vtk9.

---
### `base.py`
- corrected bug on `diagonalSize()` returning a wrong value

---
### `addons.py`
- added `addSplineTool()` to interactively spline points in space
- added `labelRotation` in addScalarBar3D
- added `xShiftAlongY` keywords in `Axes` to slide the whole axis position along another axis
- added `xAxisRotation` to rotate the whole axis (ticks and labels)
- `addScalarBar3D()` can now render categorical data

---
### `colors.py`
- fixed small issue in `printc` to support different terminals

---
### `mesh.py`
- `computeNormals()` is no more changing the nr of mesh points unless `featureAngle` is specified
    - added keywords: `featureAngle=None, consistency=True`
- `intersectWithLine()` can now return cell ids, not just points

---
### `plotter.py`
- improved automatic text management in `show("some text")`
- added `computeWorldPosition(point2d)` to get the 3d point in the scene from a screen 2d point
- added `addSplineTool()` to interactively spline points in space
- small fix in `plotter.add()` for offscreen mode.
- added `topicture()` to render a scene into a `Picture` object (to crop, mirror etc)

---
### `picture.py`
- added FFT and RFFT, added example `fft2d.py`
- can save `Picture` obj to file as jpg or png

---
### `pointcloud.py`
- added `cutWithBox()`, `cutWithLine()`, `cutWithSphere()` and `cutWithCylinder()` methods

---
### `shapes.py`
- fixed small bug in `Glyph`

---
### `volume.py`
- added class `VolumeSlice` for volume-sliced visualization.

---
### `utils.py`
- added `roundToDigit(x,p)`, round number x to significant digit


-------------------------

## New/Revised examples:
- `vedo -r koch_fractal`
- `vedo -r mesh_custom`
- `vedo -r fft2d`
- `vedo -r lines_intersect`
- `vedo -r cutFreeHand`
- `vedo -r spline_tool`
- `vedo -r legendbox`
- `vedo -r read_volume3`
- `vedo -r multi_viewer2`




