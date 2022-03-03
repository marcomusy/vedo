## Main changes

- general cleanup of the examples
- simplified licence file by splitting fonts and general MIT licence
- `vedo/texture/*` files removed
- a new syntax is allowed for swapping or indicating the renderer:
  `plotter.at(ren_number).show(mymesh)`
   instead of the (still valid):
  `plotter.show(mymesh, at=ren_number)`

---
### `applications.py`

---
### `io.py`

---
### `pointcloud.py`
- `cmap(arrayName=...)` -> `cmap(name=...)`

---
### `mesh.py`
- can now recover the edges of a mesh with `mymesh.edges()`

---
### `plotter.py`

- keyword change `show(interactorStyle=...)` -> `show(mode=...)`
- added function `vedo.close()` to close the latest Plotter

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







