## Main changes
Major internal refactoring.

## Breaking changes

### Renaming
- rename internal list `plt.actors` which now become `plt.objects`
- rename `.points()` to property `.vertices`. Hence:
    `mesh.points() -> mesh.vertices`
    `mesh.points(newpoints) -> mesh.vertices = newpoints`
- rename `.cell_centers()` to property `.cell_centers`
- rename `.faces()` to property `.cells`
- rename `.lines()` to property `.lines`
- rename `.edges()` to property `.edges`
- rename `.normals()` and split it into property `.vertex_normals` and property `.cell_normals`
- rename `picture.Picture2D(...)` which becomes `Image(...).clone2d()` (see `examples/pyplot/embed_matplotlib.py`).
- rename `Volume.probe_points()` which becomes `points.probe(volume)`
- rename `Volume.probe_line()` which becomes `line.probe(volume)`
- rename `Volume.probe_plane()` which becomes `plane.probe(volume)`
- rename `file_io.load_transform()`. `LinearTransform("file.mat")` substitutes it.
- rename `transform_with_landmarks()` to `align_with_landmarks()`
- rename `find_cells_in()` to `find_cells_in_bounds()`
- rename `mesh.is_inside(pt)` moved to `mesh.contains(pt)`
- rename `Slicer2DPlotter` moved to `application module.Slicer2DPlotter`
- rename and moved method `voronoi()` to `points.generate_voronoi()`
- rename class `Ruler` to `Ruler3D`

### Other changes
- improvements in how vtk classes are imported (allow lazy import)
- improvements to method `mesh.clone2d()`
- improvements in `Slicer3DPlotter` thanks to @daniel-a-diaz in #925
- improvements in `applications.Browser`

- add new `vedo.transformations.py` module.
- add `plotter.pick_area()` thanks to @ZiguoAtGitHub and @RubendeBruin feedback.
- add texture to npz files thanks to @zhouzq-thu in #918
- add background radial gradients
- add `utils.line_line_distance()` function
- add `utils.segment_segment_distance()` function
- add `plotter.initialize_interactor()` method
- add object hinting by hovering mouse (see `flag_labels1.py`)
- add `colors.lut_color_at(value)` the color of the lookup table at value.
- add `.show(..., screenshot="myfile.png")` keyword
- add `object.coordinates` same as `object.vertices`
- add `move()` to move single points or objects
- add `copy()` as alias to `clone()`
- add "Roll" to camera dictionary (thanks @baba-yaga )
- add `applications.Slicer3DTwinPlotter` thanks to @daniel-a-diaz
- add radii feature to `smooth_mls_2d()` by @jo-mueller (now store results in arrays `mesh.pointdata['MLSVariance']` and `mesh.pointdata['MLSValidPoint']`)
- passing a `vtkCamera` to `show(camera=...)` triggers a copy of the input which is therefore not muted by any subsequent interaction (thanks @baba-yaga )

### Bug Fixes
- bug fix in `closest_point()` thanks to @goncalo-pt
- bug fix in tformat thanks to @JohnsWor in #913
- bug fix in windows OS in timers callbacks thanks to @jonaslindemann
- bug fix to non linear tranforms mode. Now it can be instantiated with a dictionary
- bug fix in meshlab interface thanks to @JeffreyWardman in #924
- bug fix changed `mp = matplotlib.colormaps[name]` in colors.py


## New/Revised Examples
```
examples/basic/buttons.py
examples/basic/input_box.py
examples/basic/sliders2.py
examples/basic/spline_tool.py
examples/basic/interaction_modes2.py
examples/advanced/timer_callback1.py
examples/advanced/timer_callback2.py
examples/advanced/warp4a.py
examples/advanced/warp4b.py
examples/pyplot/embed_matplotlib.py
examples/pyplot/plot_fxy2.py
examples/simulations/springs_fem.py
examples/simulations/lorenz.py
examples/volumetric/numpy2volume0.py
examples/volumetric/slicer1.py
examples/volumetric/tet_astyle.py
examples/volumetric/tet_cut1.py
examples/volumetric/tet_cut2.py
examples/other/flag_labels1.py
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
flatarrow.py
mesh_lut.py
mesh_map2cell.py
texturecubes.py
meshquality.py
volumetric/streamlines1.py


