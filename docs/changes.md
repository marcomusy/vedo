## Main changes

- Added support for `ipygany` in jupyter notebooks.
- Command line interface `vedo ...` should now work on windows 10 too.
- added skybox environment with `show(..., bg="path/to/file.hdr")`
- added (kind of limited) support to `wxpython` embedding (analogous to Qt, but open-source)
- updated pymeshlab examples


---
### `base.py`

- added `.lighting(metallicity=1, roughness=0.1)` to include PBR (physics based rendering).
- method `.printInfo()` is now simply `.print()`

---
### `addons.py`

- added `SplineTool.nodes()` to retrieve current node positions.

---
### `colors.py`

---
### `mesh.py`

---
### `plotter.py`

- added `record()` and `play()` to store and playback camera and all other events
- keyword `interactorStyle` becomes now `mode`

---
### `picture.py`

- added `level()` and `window()` methods to control brightness and contrast of a Picture.
- added `smooth()` to make gaussian smoothing of a Picture object.
- added `channels` keyword to specify rgba channels to use (useful to remove alpha)

---
### `pyplot.py`

- added `plot(mode="bar")`

---
### `pointcloud.py`

- added `hausdorffDistance()` to compute the Hausdorff distance of two point sets or meshes

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
- `vedo -r ` colorizes code dump.

-------------------------

## New/Revised examples:
- `vedo -r plot_bars`
- `vedo -r alien_life`
- `vedo -r pendulum_ode`
- `vedo -r earth_model`
- `vedo -r qt_window2`
- `vedo -r spline_tool`
- `vedo -r wx_window1`
- `vedo -r picture2mesh`
- `vedo -r record_play`
- `vedo -r pymeshlab1`
- `vedo -r pymeshlab2`


