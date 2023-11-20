## Main changes

- fixes to `extrude()` thanks to @JeffreyWardman
- filter out triangle strips in Ribbon and extrude()
- improvements in doc strings

## Breaking changes
- improvements to `shapes.Ellipsoid()` and bug fixes in #978 by @daniel-a-diaz
- improvements to `pointcloud.pca_ellipsoid()` and bug fixes
- improvements to `pointcloud.pca_ellipse()` and bug fixes

### Renaming

### Other changes


### Bug Fixes

## New/Revised Examples
```
```

### Broken Examples
```
tests/issues/discussion_800.py
tests/issues/issue_905.py
gyroscope1.py broken
markpoint.py
examples/other/pygmsh_cut.py ust cut tetmesh to gen ugrid
```

#### Broken Projects
umap_viewer3d
trackviewer (some problems with removing a track)


#### Broken Exports to .npz:
boolean.py
cartoony.py
mesh_lut.py
mesh_map2cell.py
texturecubes.py
meshquality.py
volumetric/streamlines1.py


