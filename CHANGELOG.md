# Changelog

All notable changes to this project will be documented in this file.


===================================================================================
# Development Version 

## Changes and Fixes
- general code cleanup with pylint.
- add `utils.compute_hessian()`
- fix issue #1224 for cells coloring in jupyter notebook for k3d
- fix nan case in scalarbar3d()






## Soft-breaking Changes
Changes that may break existing code whose fixing is trivial:


## Hard-breaking Changes
Changes that will break existing code and need active thinking and some work to adapt

- None

## New/Revised Examples
```
```

## To Do
- fix trasform in image.tomesh() is not transmitted to mesh


#### (Internal) Broken Examples
Examples that are not fully functional and need some fixing:
```
markpoint.py (misplaced leader indicator)
cut_and_cap.py (incomplete capping)
tests/issues/discussion_800.py (incomplete capping of tube)
advanced/warp4b.py (probs with picker?)
```

#### (Internal) Known issues
umap_viewer3d should be revised
trackviewer (some problems with removing a track, and z spacing)
pyplot.plot cannot plot constant line or single point
numpy2volume0.py  (volume alphas look bad)

##### (Internal) Broken exports to .npz:
Fails to export correctly to npz format
```
boolean.py
cartoony.py
mesh_lut.py
mesh_map2cell.py
texturecubes.py
meshquality.py
streamlines1.py
```







===================================================================================

# Version 2024.5.3

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
- add documentation to `Mesh.boolean()` #1173 by @jkunimune
- raise an error when calling cell_normals before compute_normals() #1172 by @jkunimune
- add documentation warning as computing normals can affect appearence of the mesh #1174 by @jkunimune
- add documentation about `Cube` and `Box` having duplicated vertices to allow defining normals #1171
- add documentation do address the behaviour of `mesh.volume()` and `mesh.is_closed()` wrt duplicated vertices.
- add `plotter.reset_clipping_range()` to reset the camera clipping range based on the bounds of the visible actors #1170
- fix issue with find_cell() in #1095
- improvements to `volume.isosurface_discrete()` in #1180 by @snownontrace
- fix bug on video frame by resetting camera clipping range in #1180 by @snownontrace
- changes in the scalarbar2d object.
- fix purging of nan in pyplot.plot()
- fix line trace to skip first point
- adjust volume transfer function for transparency @Poisoned
- fixing axes type 10 by @Poisoned
- improvements to input/output functionality for Assembly @ttsesm
- added `mesh.remove_all_lines()` method
- added keyword `Plane(edge_direction=...)` by @smoothumut
- added `isolines(..., n=list())` option to pass specific values.
- in `file_io.screenshot()` add fourth channel representing trasparency @miek0tube
- remove obsolete class `CellCenters` which is now function `object.cell_centers()`


## Soft-breaking Changes
Changes that may break existing code whose fixing is trivial:

- change `object.points()` to `object.points` everywhere.
- change `object.cell_centers` to `object.cell_centers().points` everywhere.
- change `Picture()` to `Image()` everywhere.

## Hard-breaking Changes
Changes that will break existing code and need active thinking and some work to adapt

- None


## New/Revised Examples
```
examples/advanced/spline_draw2.py
examples/volumetric/isosurfaces2.py
examples/pyplot/fit_curve2.py

tests/issues/issue_1146.py
tests/issues/discussion_1190.py
tests/issues/test_sph_harm2.py
tests/issues/issue_1218.py

tests/snippets/test_interactive_plotxy1.py
tests/snippets/test_interactive_plotxy2.py
tests/snippets/test_elastic_pendulum.py
```


===================================================================================
# Version 2024.5.2

## Changes and Fixes
- add `magic-class` example
- fix bug in `IsosurfaceBrowser` in #1064
- add `mesh.adjacency_list()` and `graph_ball()` methods by @sergei9838
- add `utils.circle_from_3points()` function.
- add example `examples/other/iminuit2.py`
- add `rotation=..` to `Arrow2D()` class
- improvements to `applications.MorphPlotter` class
- add `FlyOverSurface` class and  `examples/basic/interaction_modes3.py`
- add `mesh.extrude_and_trim_with()` method out of #1077
- fix reset clipping range in thumbnail generation in #1085
- add `mesh.euler_characteristic()`, `mesh.genus()` and `mesh.to_reeb_graph()` in #1084
- fix `reset_camera()` by @sergei9838 and Eric
- fix handle empty axis for rotation #1113 by @JeffreyWardman 
- fix minor bug in RoundedLine #1104 by @PinkMushroom
- fix avoid overwriting screenshots with "S" key #1100 by @j042
- add support for `meshlib` in https://doc.meshinspector.com/index.html
- add relevant keyword options to `core.probe()` method
- increase precision in writing obj files in #1119 by @ManuGraiph
- add `plotter.freeze()` to freeze interaction of current renderer in #1122 by @sergei9838
- add class `addons.ButtonWidget` to address issue #1138
- add typing annotations in submodules

* allow for dictionary input in Group and Assembly by @JeffreyWardman in https://github.com/marcomusy/vedo/pull/1057
* allow assembly to correctly index objects by @JeffreyWardman in https://github.com/marcomusy/vedo/pull/1062
* backwards compatibility in typing with python < 3.11 by @JeffreyWardman in https://github.com/marcomusy/vedo/pull/1093
* avoid overwriting screenshots with `"S"` key by @j042 in https://github.com/marcomusy/vedo/pull/1100
* minor bug in RoundedLine by @PinkMushroom in https://github.com/marcomusy/vedo/pull/1104
* bugfix: add tolerance to contains by @JeffreyWardman in https://github.com/marcomusy/vedo/pull/1105
* Mitigate issue #769: don't set backend to '2d' in IPython REPLs by @paulbrodersen in https://github.com/marcomusy/vedo/pull/1108
* handle empty axis for rotation by @JeffreyWardman in https://github.com/marcomusy/vedo/pull/1113
* Print position parameter as 'pos' by @adamltyson in https://github.com/marcomusy/vedo/pull/1134

## Soft-breaking Changes
Changes that will break existing code whose fixing is trivial:

- remove `concatenate=True` keyword from `apply_transform()` discussed in  #1111


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
tests/issues/issue_1077.py
```


===================================================================================
# Version 2024.5.1

## Changes and Fixes

- fixes to `extrude()` method thanks to @JeffreyWardman
- add `utils.madcad2vedo` conversion as per #976 by @JeffreyWardman
- add `utils.camera_to_dict()`
- add `Axes(title_backface_color=...)` keyword
- fix `labels()` and `labels2d()`
- add `shapes.plot_scalar()` plot a scalar along a line.
- add support for `tetgenpy`
- add `transformations.compute_main_axes()`
- add `transformations.__call__()` to apply a transformation
- fix small bug in `pointcloud.distance_to()`
- add `applications.MorphPlotter()` to morph a polygonal mesh to a target mesh
- add `smooth_data()` to smooth/diffuse data attributes
- add `shapes.Tubes()`
- add `utils.Minimizer()` class
- add `CellCenters(Points)` class
- add `core.apply_transform_from_actor()`
- add `add volume.slab()`
- add `mesh.generate_random_points()` to generate random points onto a surface
- add `tetmesh.generate_random_points()` to generate random points in a tet mesh
- rename `integrate_arrays_over_domain()` to `integrate_data`
- extend `volume.operation()` to support logic operations as per #1002
- add `pointcloud.relax_point_positions()` method
- add `pointcloud.auto_distance()` method calculates the distance to the closest point in the same cloud of points.
- fixed `mesh.collapse_edges()` after #992
- add `mesh.cut_closed_surface()`
- fix `image.clone()` in #1011
- add `transformations.TransformInterpolator` class
- add `Line.find_index_at_position()` finds the index of the line vertex that is closest to a point
- add `visual.LightKit` class which provides "natural" lighting from 4 sources.
- add `fast-simplification` example by @Louis-Pujol in #992
- add metadata "shape" to `volume.slice_plane()` in #1018
- fix `core.mark_boundaries()` method
- add callbacks for cutters in #1020 and `examples/volumetric/slice_plane3.py`
- add `utils.andrews_curves()` function.
- add `transformations.LinearTransform.transpose()` method.
- add `pointcloud.generate_segments()` to generate a continous line from un-ordered points in 3d
- fix `assembly.__add__()` by @j042 in #1036
- small fix to `Ruler3D` class.
- add `plotter.render_hidden_lines()` method
- add slot for triangle strips in constructor `Mesh([verts, faces, lines, strips])` in #1019
- internally use "import vedo.vtkclasses as vtki" instead of "vtk" to avoid confusion
- add `join_with_strips()` in #1043
- improvements to `shapes.Ellipsoid()` and bug fixes in #978 by @daniel-a-diaz
- improvements to `pointcloud.pca_ellipsoid()` and bug fixes
- improvements to `pointcloud.pca_ellipse()` and bug fixes
- fix Plotter `a` toggle keypress
- fix viz on jupyter notebook as per #994
- fix `mesh.imprint()`
- small fix to `applications.Slicer2DPlotter`
- automatically apply the actor transform to an object actor that was moved manually (via eg "InteractorStyleTrackballActor") in #1045 and #1046 by @sergei9838
- add support to  `RectilinearGrid` and `StructuredGrid` data (experimental).
- improvements to `RayCastPlotter`
- add `visual.scalar_range()` to control mesh coloring.
- fix `shapes.Text3D.text()` by @gioda
- add `volume.isosurface_discrete()` method


## Soft Breaking Changes
Changes that can break existing code whose fixing is trivial:

- change `clone2d(scale=...)` to `clone2d(size=...)`
- remove `shapes.StreamLines(object)` becomes `object.compute_streamlines()`
- split `mesh.decimate()` into `mesh.decimate()`, `mesh.decimate_pro()` and `mesh.decimate_binned()` as per #992
- modified `core.clean()` after #992
- remove `import_window()` for obj files and create `load_obj()` by @zhouzq-thu in #891
- add `smooth_mls_12d(..., n=0)` to fix the number of neighbors in the smoothing
- modified API for `mesh.binarize()`
- `plotter.add_hover_legend()` now returns the id of the callback.
- removed `settings.render_lines_as_tubes` and `settings.hidden_line_removal`, add `plotter.render_hidden_lines()` method
- fix `close()`, `close_window()` is now obsolete and removed.


## Hard Breaking Changes
Changes that will break existing code and need active thinking and work to adapt

- None


## New/Revised Examples
```
examples/basic/sliders_hsv.py
examples/basic/buttons1.py
examples/basic/buttons2.py
examples/basic/input_box.py

examples/advanced/warp4b.py
examples/advanced/diffuse_data.py
examples/advanced/moving_least_squares1D.py

examples/volumetric/slab_vol.py
examples/volumetric/streamlines1.py
examples/volumetric/streamlines2.py
examples/volumetric/streamlines3.py
examples/volumetric/streamlines4.py
examples/volumetric/office.py
examples/volumetric/slice_plane1.py
examples/volumetric/slice_plane3.py
examples/volumetric/mesh2volume.py
examples/volumetric/read_volume3.py
examples/volumetric/rectl_grid1.py
examples/volumetric/struc_grid1.py
examples/volumetric/app_raycaster.py
examples/volumetric/isosurfaces1.py
examples/volumetric/isosurfaces2.py

examples/simulations/mag_field1.py
examples/pyplot/andrews_cluster.py

examples/other/madcad1.py
examples/other/tetgen1.py
examples/other/nelder-mead.py
examples/other/fast_simpl.py

tests/issues/issue_968.py
tests/issues/issue_1025.py
tests/issues/test_force_anim.py
tests/snippets/test_discourse_1956.py
tests/snippets/test_ellipsoid_main_axes.py
tests/snippets/test_compare_fit1.py
```


===================================================================================
# Version 2023.5.0

## Main changes
Major internal refactoring.

## Breaking changes

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

## Other changes
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

## Bug Fixes
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


===================================================================================
# Version 2023.4.5

## Main changes

- Rename module `io.py` to `file_io.py` to avoid overriding stlib `io`.
- Complete revision of cutter widgets functionality
- Integration in napari thanks to @jo-mueller


### Breaking changes
- method `plotter.add()` no more accepts keyword `render=True/False`. Please use `plotter.add().render()` explicitly. Same thing for `plotter.remove()`.

### Other fixes and improvements
- added gpu acceleration for CLI volumetric visualization in #832 by @zhang-qiang-github
- fixes for `k3d` jupyter backend
- added `plotter.fov(value)` (field of view, the so called "fish-eye" effect)
- fix `ploter.get_meshes()`
- fix for `plotter.remove(unpack_assemblies=False)` method
- fix for `io.import_window()` method
- added `cut_with_cookiecutter()` to cut 2D contours.
- fix `shapes.NormalLines()` class
- added `vedo.interactor_modes` module
- added `vedo.interactor_modes.BlenderStyle` class
- added `base.pointdata.clear()` to remove all associated data 
- added `volume.hide_voxels()` for visualization
- added `Event.timerid` attribute
- fix to `Volume.operation` by @DanKrsi
- fix links in `pyplot` examples by @androbaza
- fix `screenshot_scale` and remove it from settings.
- allow initializing `ScalarBar` with a tuple range `(min, max)`
- Update API Documentation for Changing Backend by @bhacha
- Add `application.Browser().play()` to autoplay a slider
- Add `pad()` to padding a Volume with zeros voxels (useful to `dilate()`)
- Add `ProgressBarWidget()` to show a progress bar in the rendering window
- Fix Scalarbar3D logscale and change separator symbol by @XushanLu
- Fix `vedo/interactor_modes.mouse_left_move()` by @MiticoDan
- Added `applications.AnimationPlayer` class by @mikaeltulldahl
- fix convex hull in 2D by @ManuGraiph


-------------------------
## New/Revised Examples
```
examples/basic/sliders_range.py
examples/basic/interaction_modes.py
examples/advanced/timer_callback3.py
examples/advanced/warp6.py
examples/pyplot/histo_1d_e.py
examples/other/tensor_grid2.py
examples/simulations/airplane1.py
examples/simulations/lorenz.py
examples/simulations/gas.py
examples/simulations/aspring2_player.py
```
