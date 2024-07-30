
## Changes and Fixes

- add `settings.force_single_precision_points = False` in #1137 by @JeffreyWardman and @sean-d-zydex
- fix Volume masking in #1146 by @ivishalanand 
- fix `LegendBox` in #1153 by @GerritFischer
- add `mesh.laplacian_diffusion()`
- fix `DistanceTool` in #1158
- fix `shapes.Plane.normal` in #1159 by @smoothumut
- add `Arrow.top_point()` and `Arrow.base_point()` to extract current arrow position #1163 @smoothumut
- fix `Arrow.top_index` to produce the correct index value
- add `assembly.Group.objects` by @smoothumut
- add `addons.DrawingWidget` class for tracing on planar props
- add `Video(..., scale=1)` keyword in #1168 by @YongcaiHuang
- modify `legosurface(boundary=True)` default in #1166


## Soft-breaking Changes
Changes that will break existing code whose fixing is trivial:

- None


## Hard-breaking Changes
Changes that will break existing code and need active thinking and work to adapt

- None


## New/Revised Examples
```
tests/issues/issue_1146.py
examples/advanced/spline_draw2.py
```


### Broken Examples
Examples that are not fully functional and need some fixing:

```
markpoint.py (misplaced leader indicator)
cut_and_cap.py (incomplete capping)
tests/issues/discussion_800.py (incomplete capping of tube)
examples/volumetric/numpy2volume1.py ()
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
