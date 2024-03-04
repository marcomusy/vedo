
## Changes and Fixes

- add typing annotations
- add magic-class example
- fix bug in `IsosurfaceBrowser` in #1064
- partial fix of bug in 1066

## Soft Breaking Changes
Changes that will break existing code whose fixing is trivial:

- None


## Hard Breaking Changes
Changes that will break existing code and need active thinking and work to adapt

- None


## New/Revised Examples
```
examples/other/magic-class1.py
```

### Broken Examples
Examples that are not fully functional and need some fixing:
```
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
