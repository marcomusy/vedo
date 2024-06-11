
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
- fix reset clipping range in thumbnail generation as per #1085
- add `mesh.euler_characteristic()`, `mesh.genus()` and `mesh.to_reeb_graph()` as per #1084
- fix `reset_camera()` by @sergei9838 and Eric
- fix handle empty axis for rotation #1113 by @JeffreyWardman 
- fix set backend to '2d' in IPython REPLs #1108 by @paulbrodersen
- fix add tolerance to contains #1105 by @JeffreyWardman
- fix minor bug in RoundedLine #1104 by @PinkMushroom
- fix avoid overwriting screenshots with "S" key #1100 by @j042
- add support for meshlib
- add more kw options to `core.probe()`
- increase precision in writing obj files as per #1119 by @ManuGraiph
- add `plotter.freeze()` to freeze interaction of current renderer in #1122 by @sergei9838
- add class `addons.ButtonWidget` to address issue #1138


## Soft-breaking Changes
Changes that will break existing code whose fixing is trivial:

- remove concatenate=True flag from `apply_transform()` discussed in  #1111


## Hard-breaking Changes
Changes that will break existing code and need active thinking and work to adapt

- None


## New/Revised Examples
```
examples/basic/interaction_modes3.py
examples/basic/interaction_modes4.py
examples/basic/buttons3.py
examples/advanced/warp4b.py
examples/other/magic-class1.py
examples/other/iminuit2.py
examples/other/meshlib1.py
```

### Broken Examples
Examples that are not fully functional and need some fixing:
```
markpoint.py (misplaced leader indicator)
cut_and_cap.py (incomplete capping)
gyroscope1.py (broken physics)
tests/issues/discussion_800.py (incomplete capping of tube)
```

#### Broken Projects and Known Issues
umap_viewer3d should be revised
trackviewer (some problems with removing a track, and z spacing)
pyplot.plot cannot plot constant line or single point

#### Broken Exports to .npz:
boolean.py
cartoony.py
mesh_lut.py
mesh_map2cell.py
texturecubes.py
meshquality.py
streamlines1.py
