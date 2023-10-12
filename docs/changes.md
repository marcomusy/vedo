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
- in `plotter.add_button(func)`, must use `func(event)` instead of `func()`
(thanks to @smoothumut for spotting the bug)
- change .points() to .vertices everywhere
- change .cell_centers() to .cell_centers everywhere
- change .faces() to .cells everywhere
- change .lines() to .lines everywhere



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
align5.py
background_image.py
cut_freehand.py
cut_interactive.py
glyphs2.py
largestregion.py
rotate_image.py
slider_browser.py
ssao.py


~/Projects/vedo/examples/advanced
interpolate_scalar3.py
recosurface.py
spline_draw.py
timer_callback2.py
warp4.py
warp6.py

~/Projects/vedo/examples/pyplot
glyphs2.py
explore5d.py
goniometer.py
histo_2d_a.py
histo_2d_b.py
isolines.py


~/Projects/vedo/examples/simulations
airplane1.py
aspring1.py
brownian2d.py
gyroscope1.py
gyroscope2.py

```



