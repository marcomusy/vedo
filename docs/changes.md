## Main changes

- added `plotter.pick_area()` thanks to @ZiguoAtGitHub and @RubendeBruin feedback.
- bug fix in `closest_point()` thanks to @goncalo-pt
- bug fix in tformat thanks to @JohnsWor in #913
- add texture to npz files thanks to @zhouzq-thu in #918

- Fix meshlab interface thanks to @JeffreyWardman in #924
- Update `Slicer3DPlotter` thanks to @daniel-a-diaz in #925
- Improvemnets on `applications.Slicer3DPlotter`
- Improvements on `applications.Browser`
- add background radial gradients
- add `utils.line_line_distance()`
- add `utils.segment_segment_distance()`
- addressed bug on windows OS in timers callbacks thanks to @jonaslindemann
- add `plotter.initialize_interactor()`
- add object hinting (flag_labels1.py) by hovering mouse
- add `colors.lut_color_at(value)` the color of the lookup table at value.
- remove `picture.Picture2D(...)` which becomes `Image(...).clone2d()`
see `examples/pyplot/embed_matplotlib.py`.
- improvements to method `mesh.clone2d()`
- name change from `Picture` to `Image`, renamed `vedo.picture` to `vedo.image`
- reformat how vtk classes are imported (allow some laziness)
- add `.show(..., screenshot="myfile.png")` keyword
- add `object.coordinates` same as `object.vertices`
- fixing to non linear tranforms mode. Now it can be instantiated with a dictionary
    add `move()` to move single points or objects
- add `copy()` as alias to `clone()`
- remove `file_io.load_transform()` LinearTransform("file.mat") substitutes this
- add "Roll" to camera settings (thanks @baba-yaga )


### Breaking changes
- requires vtk=>9.0
- plt.actors must become plt.objects
- change .points() to .vertices
- change .cell_centers() to .cell_centers
- change .faces() to .cells
- change .lines() to .lines
- change .edges() to .edges
- change .normals() to .vertex_normals and .cell_normals
- removed `Volume.probe_points()`
- removed `Volume.probe_line()`
- removed `Volume.probe_plane()`
- `Slicer2DPlotter` moved to application module
- `mesh.is_inside(pt)` moved to `mesh.contains(pt)`
- added `applications.Slicer3DTwinPlotter` thanks to @daniel-a-diaz
- passing a `vtkCamera` to `show(camera=...)` triggers a copy of the input which is
therefore not muted by any subsequent interaction (thanks @baba-yaga )


-------------------------
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
examples/simulations/lorenz.py
examples/volumetric/slicer1.py
examples/other/flag_labels1.py
examples/pyplot/embed_matplotlib.py
examples/pyplot/plot_fxy2.py
examples/simulations/springs_fem.py
```


### Broken Examples
```
tests/issues/discussion_800.py
tests/issues/issue_905.py

gyroscope1.py broken
tet_cut2.py broken
markpoint.py
plot_spheric.py
```

### broken in npz dump:
boolean.py
cartoony.py
flatarrow.py
mesh_lut.py
mesh_map2cell.py
rotate_image.py (miss transform)
texturecubes.py
meshquality.py
volumetric/streamlines1.py

test offline screenshot
