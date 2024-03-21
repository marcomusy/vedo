
## Changes and Fixes

- add typing annotations
- add magic-class example
- fix bug in `IsosurfaceBrowser` in #1064
- partial fix of bug in 1066
- add `mesh.adjacency_list()` and `graph_ball()` by @sergei9838
- add `utils.circle_from_3points()` function.
- add example `examples/other/iminuit2.py`
- add `rotation=..` to `Arrow2D()` class
- improvements to `applications.MorphPlotter`
- add `FlyOverSurface` class and  `examples/basic/interaction_modes3.py`
- address #1072 for pyinstaller 
- add `mesh.extrude_and_trim_with()` method out of #1077


## Soft-breaking Changes
Changes that will break existing code whose fixing is trivial:

- None


## Hard-breaking Changes
Changes that will break existing code and need active thinking and work to adapt

- None


## New/Revised Examples
```
examples/basic/interaction_modes3.py
examples/advanced/warp4b.py
examples/other/magic-class1.py
examples/other/iminuit2.py
```

### Broken Examples
Examples that are not fully functional and need some fixing:
```
examples/basic/light_sources.py

earth_model.py (wrong colormapping?)
markpoint.py (misplaced leader indicator)
cut_and_cap.py (incomplete capping)
gyroscope1.py (broken physics)
tests/issues/discussion_800.py (incomplete capping of tube)
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
streamlines1.py
