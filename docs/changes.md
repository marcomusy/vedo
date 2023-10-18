## Main changes

- added `plotter.pick_area()` thanks to @ZiguoAtGitHub and @RubendeBruin feedback.
- bug fix in `closest_point()` thanks to @goncalo-pt
- bug fix in tformat thanks to @JohnsWor  https://github.com/marcomusy/vedo/pull/913
- add texture to npz files thanks to @zhouzq-thu https://github.com/marcomusy/vedo/pull/918

- Fix meshlab interface thanks to @JeffreyWardman in #924
- Update `Slicer3DPlotter` thanks to @daniel-a-diaz in #925
- Improvemnets on `applications.Slicer3DPlotter`
- Improvements on `applications.Browser`
- add background radial gradients
- add `utils.line_line_distance()`
- add `utils.segment_segment_distance()`


### Breaking changes
- plt.actors must become plt.objects
- in `plotter.add_button(func)`, must use `func(event)` instead of `func()`
(thanks to @smoothumut for spotting the bug)
- change .points() to .vertices everywhere
- change .cell_centers() to .cell_centers everywhere
- change .faces() to .cells everywhere
- change .lines() to .lines everywhere
- change .edges() to .edges everywhere
- change .normals() to .vertex_normals and .cell_normals everywhere
- removed `Volume.probe_points()`
- removed `Volume.probe_line()`
- removed `Volume.probe_plane()`
- `Slicer2DPlotter` moved to application module
- `mesh.is_inside(pt)` moved to `mesh.contains(pt)`
- added `applications.Slicer3DTwinPlotter` thanks to @daniel-a-diaz




-------------------------
## New/Revised Examples
```
examples/advanced/timer_callback1.py
examples/advanced/timer_callback2.py
examples/basic/buttons.py
examples/basic/input_box.py
examples/basic/sliders2.py
examples/basic/interaction_modes2.py
examples/volumetric/slicer1.py
```



### Broken Examples
```
~/Projects/vedo/examples/basic
background_image.py
glyphs2.py


~/Projects/vedo/examples/advanced
warp4.py


~/Projects/vedo/examples/pyplot
caption.py
goniometer.py
histo_2d_b.py
histo_hexagonal.py
isolines.py


~/Projects/vedo/examples/simulations
airplane1.py
aspring1.py
brownian2d.py
gyroscope1.py
gyroscope2.py
lorenz.py
pendulum_3d.py
trail.py


~/Projects/vedo/examples/other
ellipt_fourier_desc.py
export_numpy.py
flag_labels1.py
ex06_elasticity2.py


release on master as branch?
tests/issues/issue_871a.py


staging
clonala analysis
```



