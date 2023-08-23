## Main changes

- added `plotter.pick_area()` thanks to @ZiguoAtGitHub and @RubendeBruin feedback.
- bug fix in `closest_point()` thanks to @goncalo-pt
- bug fix in tformat thanks to @JohnsWor  https://github.com/marcomusy/vedo/pull/913


### Breaking changes
- in `plotter.add_button(func)`, must use `func(event)` instead of `func()`
(thanks to @smoothumut for spotting the bug)


-------------------------
## New/Revised Examples
```
examples/advanced/timer_callback1.py
examples/advanced/timer_callback2.py
examples/basic/buttons.py
examples/basic/input_box.py
examples/basic/sliders2.py
examples/basic/interaction_modes2.py
```

### Broken Examples
```
meshio_read.py
navier-stokes_lshape.py
```
earth_model.py
tet_explode.py
tetralize_surface.py
histo_hexagonal.py
meshio_read.py
nearest.py



