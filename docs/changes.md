
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
- make load functions compatible with pathlib #1176 by @Louis-Pujol
- fixed broken link to example #1175 by @jo-mueller
- add documentation to Mesh.boolean() #1173 by @jkunimune
- raise an error when calling cell_normals before compute_normals() #1172 by @jkunimune
- add documentation warning as computing normals can affect appearence of the mesh #1174 by @jkunimune
- add documentation about `Cube` and `Box` having duplicated vertices to allow defining normals #1171
- add documentation do address the behaviour of `mesh.volume()` and `mesh.is_closed()` wrt duplicated vertices.
- add `plotter.reset_clipping_range()` to reset the camera clipping range based on the bounds of the visible actors #1170
- fix issue with find_cell() in #1095
- improvements to `volume.isosurface_discrete()` in #1180 by @snownontrace
- fix bug on video frame by resetting camera clipping range in #1180 by @snownontrace
- changes in the scalarbar2d object.


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
examples/volumetric/isosurfaces2.py
examples/pyplot/fit_curve2.py
```

## To Do
- improve 4/5 keys to show a scalarbar
- add interpolate_scalar5 to webpage
- fix draw_spline1,2 in webpage


### Broken Examples
Examples that are not fully functional and need some fixing:

```
markpoint.py (misplaced leader indicator)
cut_and_cap.py (incomplete capping)
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
