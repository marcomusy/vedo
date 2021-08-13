## Main changes

- Added support for `ipygany` in jupyter notebooks.
- Command line interface `vedo ...` should now work on windows 10 too.
- added skybox environment with `show(..., bg="path/to/file.hdr")`
- added (kind of limited) support to `wxpython` embedding (analogous to Qt, but open-source)


---
### `base.py`

- added `.lighting(metallicity=1, roughness=0.1)` to include PBR (physics based rendering).

---
### `addons.py`

- added `SplineTool.nodes()` to retrieve current node positions.

---
### `colors.py`

---
### `mesh.py`

---
### `plotter.py`

---
### `picture.py`

---
### `pyplot.py`

- added `plot(mode="bar")`

---
### `pointcloud.py`

- added `hausdorffDistance()` to compute the Hausdorff distance of two point sets

---
### `shapes.py`

---
### `volume.py`

---
### `utils.py`


---
### `cli.py`

- removed `bin/vedo` and created entry point from `vedo/cli.py` (command line interface).
This works better on windows systems.

-------------------------

## New/Revised examples:
- `vedo -r plot_bars`
- `vedo -r alien_life`
- `vedo -r pendulum_ode`
- `vedo -r earth_model`
- `vedo -r spline_tool`
- `vedo -r wx_window1`



