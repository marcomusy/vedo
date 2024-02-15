
## Changes and Fixes

- fixes to `extrude()` thanks to @JeffreyWardman
- filter out triangle strips in Ribbon and `extrude()`
- improvements in doc strings
- add `utils.madcad2vedo` conversion as per #976 by @JeffreyWardman
- add `utils.camera_to_dict()`
- add `Axes(title_backface_color=...)` keyword
- improvements on `plotter.__init__()`
- fix `labels()` and `labels2d()`
- add `shapes.plot_scalar()` plot a scalar along a line.
- add support for `tetgenpy`
- add `transformations.compute_main_axes()`
- add `transformations.__call__()` to apply it
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
- fix plotter `a` toggle
- fix viz on jupyter notebook as per #994
- fix `mesh.imprint()`
- small fix to `applications.Slicer2DPlotter`
- automatically apply the actor transform to an object actor that was moved manually (via eg "InteractorStyleTrackballActor") in #1045 and #1046 by @sergei9838
- add support to `StructuredGrid` data (experimental).
- improvements to `RayCastPlotter`
- add `visual.scalar_range()` to control mesh coloring.
- fix `shapes.Text3D.text()` by @gioda
- add `volume.isosurface_discrete()` method


## Soft Breaking Changes
Changes that will break existing code whose fixing is trivial:

- change `clone2d(scale=...)` to `clone2d(size=...)`
- remove `shapes.StreamLines()` becoming `object.compute_streamlines()`
- split `mesh.decimate()` into `mesh.decimate()`, `mesh.decimate_pro()` and `mesh.decimate_binned()` as per #992
- modified `core.clean()` after #992
- rename `vedo.tetmesh` to `vedo.grids` and support for `RectilinearGrid`
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

examples/pyplot/plot_stream.py
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

### Broken Examples
Examples that are not fully functional and need some fixing:
```
markpoint.py (misplaced leader indicator)
cut_and_cap.py (incomplete capping)
gyroscope1.py (broken physics)
mousehover1.py (long indicator)
tests/issues/discussion_800.py  (incomplete capping of tube)
examples/volumetric/earth_model.py (overburden wron color)
```

#### Broken Projects
umap_viewer3d
trackviewer (some problems with removing a track, and z spacing)
./clone_viewer3d.py and  ~/Projects/umap_viewer3d
    [vedo.plotter] INFO: object 'FlagPole' was manually moved. Updated to its current position.
    [vedo.plotter] INFO: object 'Text3D' was manually moved. Updated to its current position.
napari-vedo-bridge interactor frozen

rio_organoid
    too slow??


#### Broken Exports to .npz:
boolean.py
cartoony.py
mesh_lut.py
mesh_map2cell.py
texturecubes.py
meshquality.py
streamlines1.py
