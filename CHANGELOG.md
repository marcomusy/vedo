# Changelog

All notable changes to the project will be documented in this file.

===================================================================================
# Development Version 
- add interactive widget wrappers for line, sphere, point and cylinder placement,
  together with new streamlines examples showing their usage
- add Three.js scene export support and reorganize file I/O exporters
- add quaternion utilities, explicit structured-grid methods, chemistry helpers,
  and Gaussian cube loading support
- improve CLI startup and terminal output with Rich logging, better lazy imports,
  improved VTK load feedback, and better trame backend compatibility
- fix `backends.py`: trame component imports now distinguish missing packages from
  internal import failures and report incompatible layouts cleanly; notebook backend
  dispatch now validates and normalizes `settings.default_backend`; the 2D backend
  guards renderer autoclose against empty renderer lists; the k3d backend avoids
  mutating the original mesh when mapping cell colors to points, handles degenerate
  lookup tables safely, tightens line export limits, and hardens object-name lookup;
  the trame backend now uses per-plotter server/view ids to avoid state collisions
  across multiple notebook views; `_rgb2int()` now clamps inputs, ignores any alpha
  component, validates underspecified colors, and rounds instead of truncating when
  converting normalized colors to packed integers
- improve Slicer3DPlotter, plotting/runtime behavior, scene object lookup,
  follow-camera handling for Text3D, and fix a range of runtime/API issues
- fix `colors.py`: `_setup_colormaps` now raises distinct `RuntimeError` messages for missing
  vs outdated (< 3.5) matplotlib; `_has_colors` no longer uses fragile `from builtins import
  get_ipython` (replaced with `getattr(builtins, "get_ipython", None)`) and simplified to a
  single return; `_hex_to_rgb_cached` removes the intermediate list allocation; unknown VTK
  named colors in `_get_color_from_string` now warn and return gray consistently instead of
  silently returning black; `get_color` removes stale hard-coded fast-paths for "r"/"g"/"b"
  (covered by the lru-cached string resolver), replaces `.isdigit()` with `int()` try/except
  so negative-integer strings like `"-7"` are handled correctly, and unifies the 0-255 RGBA
  branch to divide all components (including alpha) by 255; `get_color_name` removes
  redundant `str()` wrap on an already-string dict key; `hsv2rgb`/`rgb2hsv` now share a
  lazy module-level `vtkMath` singleton instead of constructing a new VTK object per call;
  `rgb2hex` uses `round()` instead of `int()` to avoid truncation bias (e.g. 0.5 → #80 not #7f);
  `color_map`: rename `cut` → `is_array`, fix degenerate `vmax==vmin` scalar path to return
  `[0.0]` instead of `[value-vmin]`, and return a plain tuple `(0.5,0.5,0.5)` on error for
  scalar input (was `np.array`); `build_palette`: replace misleading `get_color()` passthrough
  on already-converted HSV values with plain `np.asarray`, fix docstring `N` → `n`;
  `build_lut`: remove leftover commented-out debug print;
  `printc`: replace bare `except:` with `except Exception:` to avoid swallowing
  `KeyboardInterrupt` and `SystemExit`;
  `printd`: guard `GetPosition()` call with `hasattr(obj, "GetPosition")` so non-VTK
  objects with a `.name` attribute no longer crash; use `str(obj.name)` to handle
  non-string name values; wrap `min`/`max` stats print in `try/except` to skip
  non-numeric sequences cleanly
- fix `OperationNode`: safe early-return when pipeline disabled, stable graphviz node IDs,
  cycle detection in tree traversal, explicit `__str__`, removed dead `counts` attribute
- fix `ProgressBar`: restore cursor + newline on `__del__`, correct `_fit_line` width
  accounting, fix `progressbar()` docstring to match actual signature
- fix `grid_corners`: `yflip` used `n` (columns) instead of `m` (rows), producing negative
  row indices and wrong box positions for non-square grids; row index now computed with `//`
- fix `make_ticks`: `raise RuntimeError` now carries a message; `if not n` guard changed
  to `if n is None` to avoid silently overriding a caller-supplied zero; custom-label path
  uses `np.isclose` instead of `==` for x1 boundary; loop variable renamed from `s` to
  `tok` to avoid shadowing the numeric step size; `useformat` branch skipped when logscale
  is active to avoid building a discarded string
- fix `cart2cyl`: replace `np.sqrt(x*x + y*y)` with `np.hypot(x, y)` for consistency
  with `cart2spher` and overflow safety on large inputs
- fix `NonLinearTransform`: `sigma` getter/setter docstrings were swapped; `invert` return
  annotation changed from `NonLinearTransform` to `Self`; `source_points`/`target_points`
  setters simplified (drop `if …: pass else:` idiom); `compute_main_axes` now subtracts the
  base transformed position `p0 = transform_point(pt)` so that the finite-difference Jacobian
  is correct at any pivot, not just the origin; `sqrt` of eigenvalues guarded with `np.abs`
  against tiny negative values from floating-point noise
- fix `LinearTransform`: sequence input now validated as a square 2D array (raises
  `ValueError` for flat or non-square inputs); JSON loading uses `.get()` for `name`/`comment`
  so missing optional keys no longer raise `KeyError`; legacy-format parser switches from
  `split(" ")` to `split()` to handle tabs and multiple spaces; `is_identity` simplified to
  `np.allclose(M, np.eye(4))`; `reorient` antiparallel branch now picks a proper perpendicular
  axis instead of perturbing `newaxis`, clamps the `arccos` argument to `[-1, 1]`, and replaces
  the magic constant `1.4142` with `np.sqrt(2)`
- fix standalone utils: `is_sequence` drops dead Py2 `__getslice__` check; `point_in_triangle`
  converts all inputs to arrays; `intersection_ray_triangle` degenerate check now tests the
  cross-product `n` instead of edge vector `v`; `triangle_solver` angle-zero guards use
  `is not None`; `get_uv` replaces deprecated `np.matrix` with `np.array`; `grep` drops
  redundant newline strip; `print_histogram` removes dead vtkImageData/vtkPolyData branches;
  `make_ticks` saves/restores numpy print options in a `finally` block; `camera_from_dict`
  uses `is not None` guard for `modify_inplace`
- fix `Minimizer`: convergence flag no longer calls `Iterate()` after `Minimize()`;
  `eval()` passes list not dict to user function; `set_parameters()` guards scalar values;
  `minimize()` resets paths on repeated calls; `_summary_rows` uses enumerate and avoids
  recomputing the Hessian on every `__str__` call
- fix `lazy_imports`: remove redundant `seen` set and `ordered` list in `build_attr_map`
  (replaced with `list(attr_map)`, relying on Python 3.7+ dict insertion-order guarantee);
  `getattr_lazy` now wraps the import+getattr in a `try/except` and raises a clear
  `ImportError` with context instead of a bare traceback; `if attr_map`/`if module_map`
  truthiness guards replaced with `is not None` throughout
- migrate the project documentation to MkDocs and refresh a large set of examples
- refactor `Settings` class: `__str__` now generates output from live values grouped
  by category (General, Rendering, Lighting, …) instead of scraping the class docstring;
  `init_colab()` and `start_xvfb()` moved to module-level functions and exported from
  `vedo` directly; `__getitem__` now raises `KeyError` (not `AttributeError`) for unknown
  keys; `__contains__` added so `"key" in settings` works; `dry_run_mode` included in
  `keys()` / `values()` / `items()` and visible in `print(settings)`; `clear_cache` path
  construction made explicit for absolute vs. relative `cache_directory`; `set_vtk_verbosity`
  now imports through `vedo.vtkclasses` instead of `vtkmodules` directly


## Soft-breaking Changes
- examples under `examples/other/` were moved to `examples/extras/`
- `settings.init_colab()` and `Settings.start_xvfb()` are now module-level functions;
  use `vedo.init_colab()` and `vedo.start_xvfb()` instead


## Hard-breaking Changes
- None



## New/Revised Examples
```
examples/extras/export_threejs.py
examples/extras/quaternion_tutorial.py
examples/extras/chemistry2.py
examples/animation/aizawa_attractor.py
examples/volumetric/slicer_set_volume.py
examples/volumetric/streamlines2_linewidget.py
examples/volumetric/streamlines2_spherewidget.py
examples/volumetric/streamlines2_pointwidget.py
examples/volumetric/streamlines2_cylinderwidget.py
```

#### (Internal) Broken Examples
Examples that are not fully functional and need some fixing:
```
advanced/warp4b.py (problem with picker?)
interpolate_scalar4.py misplaced scalarbar
markpoint.py (misplaced leader indicator)
cut_and_cap.py (incomplete capping)
tests/issues/discussion_800.py (incomplete capping of tube)
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
# Version 2026.6.1

## Changes and Fixes
- added example with embedding matplotlib figures in a scene
- add `Line.eval2d()` useful for plotting
- improvements to the `addons.BaseCutter` as per #1263 by @natabma
- add `applications.MorphByLandmarkPlotter` class
- add `applications.MorphBySplinesPlotter` class
- fix memory leaks in Text2D and Assembly classes #1269 by @shBLOCK
- add `volume.extract_components()`
- add `applications.ImageEditor` Plotter-derived class 
- add `pointcloud.project_point_on_variety()` function
- objects can now be removed also by substring e.g. `plt.remove("*_mesh")` will remove "some_mesh"


## Soft-breaking Changes
- small changes to the API of the `addons.BaseCutter` class
- `CornerAnnotation` class removed (substituted by Text2D)
- changes in `Button` class (check examples/basic/buttons2.py)



## Hard-breaking Changes
- None


===================================================================================
# Version 2025.5.4

## Changes and Fixes
- general code cleanup with pylint.
- add `utils.compute_hessian()`
- fix issue #1224 for cells coloring in jupyter notebook for k3d
- add support for STEP files format (needs opencascade lib) #1222
- fix nan case in scalarbar3d()
- add `.rename()` method to set any object name.
- fix bug #1230 in `line.find_index_at_position()` by @natabma
- Add lazy initialization for normals (#1231) by @CorpsSansOrganes
- add chemistry module to represent molecules and proteins
- fix to Box class by @dbotwinick


## Soft-breaking Changes
Changes that may break existing code whose fixing is trivial:


## Hard-breaking Changes
Changes that will break existing code and need active thinking and some work to adapt

- None

## New/Revised Examples
```
examples/pyplot/plot_fxy0.py
examples/volumetric/image_editor.py
examples/extras/pysr_regression.py
examples/extras/chemistry1.py
tests/issues/issue_1230.py
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
- add example `examples/extras/iminuit2.py`
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
examples/extras/magic-class1.py
examples/extras/iminuit2.py
examples/extras/meshlib1.py
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

examples/animation/mag_field1.py
examples/pyplot/andrews_cluster.py

examples/extras/madcad1.py
examples/extras/tetgen1.py
examples/extras/nelder-mead.py
examples/extras/fast_simpl.py

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
examples/animation/springs_fem.py
examples/animation/lorenz.py
examples/volumetric/numpy2volume0.py
examples/volumetric/slicer1.py
examples/volumetric/tet_astyle.py
examples/volumetric/tet_cut1.py
examples/volumetric/tet_cut2.py
examples/extras/flag_labels1.py
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
examples/extras/tensor_grid2.py
examples/animation/airplane1.py
examples/animation/lorenz.py
examples/animation/gas.py
examples/animation/aspring2_player.py
```
