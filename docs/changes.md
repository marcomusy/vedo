## Main changes


---
### `base.py`

- introduced new sintax to set/retrieve a mesh point data array: `myobj.pointdata["arrayname"]`
same for cell/face data: `myobj.celldata["arrayname"]`

---
### `addons.py`

- added kword `LegendBox(markers=...)`

---
### `colors.py`

- fixed small bug in colorMap (only affecting systems without matplotlib)

---
### `mesh.py`

- fixed problem in `geodesic()`.
- added tolerance keyword in `intersectWithLine()`. Also, now `returnIds` returns `[[pt,cellid]]`
- added kword flag to function `merge(..., flag=True)` to optionally keep track of the original meshes ids

---
### `plotter.py`

---
### `picture.py`

- added `.enhance()` method.
- method `.tonumpy()` reshape to [nx,ny,nchannels] to match matplotlib standard
- added `.append()` to stitch images the current to the left or to the top.
- added `.extent()` to specify physical extention of an image.

---
### `pyplot.py`


---
### `pointcloud.py`

- `cluster()` renamed to `pointcloud.addClustering()`

---
### `shapes.py`

---
### `volume.py`

---
### `utils.py`


---
### `cli.py`

-------------------------

## New/Revised examples:

`vedo -r optics_main1`
`vedo -r optics_main2`
`vedo -r morphomatics_tube`

