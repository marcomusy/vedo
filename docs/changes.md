
## Main changes

- fixes to `extrude()` thanks to @JeffreyWardman
- filter out triangle strips in Ribbon and extrude()
- improvements in doc strings
- add `utils.madcad2vedo` conversion as per #976 by @JeffreyWardman
- add `utils.camera_to_dict()`
- add `Axes(title_backface_color=...)` keyword
- improvements on `plotter.__init__()`
- fix `labels()` and `labels2d()`



## Breaking changes

- improvements to `shapes.Ellipsoid()` and bug fixes in #978 by @daniel-a-diaz
- improvements to `pointcloud.pca_ellipsoid()` and bug fixes
- improvements to `pointcloud.pca_ellipse()` and bug fixes


### Renaming


### Other changes
- add `core.apply_transform_from_actor()`
- add `add volume.slab()`



### Bug Fixes
- fix plotter `a` toggle


## New/Revised Examples
```
examples/volumetric/slab_vol.py
examples/other/madcad1.py
```

### Broken Examples
```
tests/issues/discussion_800.py
tests/issues/issue_905.py
gyroscope1.py broken
markpoint.py
cut_and_cap.py

volumetric/streamlines1.py
mousehover1.py
mousehover2.py (unstable hovering?)
```

#### Broken Projects
umap_viewer3d
trackviewer (some problems with removing a track, and z spacing)


#### Broken Exports to .npz:
boolean.py
cartoony.py
mesh_lut.py
mesh_map2cell.py
texturecubes.py
meshquality.py
volumetric/streamlines1.py


