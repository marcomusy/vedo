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



### Breaking changes
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

```


### Broken Examples
```
tests/issues/discussion_751.py
tests/issues/discussion_800.py
tests/issues/issue_905.py

slice_plane1.py
```

### TODO
- TextBase maybe useless can go into Actor2D
- Mesh([points, faces, lines])
- reimplement actor rotations, 
    try disable .position .rotations to check
- revisit splines and other widgets
- merge does something strange with flagpost
- analysis_plots.visualize_clones_as_timecourse_with_fit not working


