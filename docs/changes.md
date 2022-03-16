## Main changes

- New documentation pages based on `pdoc3`.
- general cleanup of the examples
- simplified licence file by splitting fonts and general MIT licence
- `vedo/texture/*` files removed
- a new syntax is allowed for swapping or indicating the renderer:
  `plotter.at(ren_number).show(mymesh)`
   instead of the (still valid):
  `plotter.show(mymesh, at=ren_number)`
- reverted `settings` to be a simple import instead of a dictionary (because of pdoc3).

---
### `applications.py`

---
### `io.py`

---
### `pointcloud.py`
- `cmap(arrayName=...)` -> `cmap(name=...)`
- added `chamferDistance()` of pointclouds
- added `cutWithMesh(self, keep=True)` to return an Assembly of the cut & discarded parts.

---
### `mesh.py`
- can now recover the edges of a mesh with `mymesh.edges()`

---
### `plotter.py`

- keyword change `show(interactorStyle=...)` -> `show(mode=...)`
- added function `vedo.close()` to close the latest Plotter
- added `chamferDistance()` metric

---
### `picture.py`

- remove by default alpha channel so that images are pickable

---
### `pyplot.py`

---
### `shapes.py`

---
### `volume.py`


---
### `settings.py`

- new polygonal fonts added: 'Vogue', 'Brachium', 'Dalim', 'Miro', 'Ubuntu'

---
### `utils.py`


-------------------------

## New/Revised examples:
`examples/basic/mousehover2.py`
`examples/simulations/wave_equation2d.py`
`examples/advanced/interpolateScalar4.py`
`examples/basic/multirenderers.py`
`examples/advanced/spline_draw.py`
`examples/simulations/museum_problem.py`
`examples/volumetric/tet_explode.py`
`examples/other/remesh_tetgen.py`



