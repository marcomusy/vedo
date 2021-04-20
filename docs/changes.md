## Main changes

- general improvements to the `vedo` command line
- `colorcet` [colormaps](https://colorcet.holoviz.org) are directly usable in `vedo`. Check example `basic/mesh_custom.py`

- general tool for free-hand cutting a mesh can be invoked from command line:
`vedo --edit https://vedo.embl.es/examples/data/porsche.ply`

- added example search by command line e.g.:
`vedo --search triangle -i`  (`-i` to ignore capital letters)

- added file(s) info dump by command line e.g.:
`vedo --info some_image.jpg https://vedo.embl.es/examples/data/porsche.ply`

---
### `base.py`
- corected bug on `diagonalSize()` returning a wrong value

---
### `addons.py`
- added `addSplineTool()` to interactively spline points in space
- added `labelRotation` in addScalarBar3D

---
### `colors.py`

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

---
### `picture.py`
- added FFT and RFFT, added example `fft2d.py`
- can save `Picture` obj to file as jpg or png

---
### `pointcloud.py`

---
### `pyplot.py`

---
### `shapes.py`

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




