## Main changes

- rename module `io.py` to `file_io.py` to avoid override stlib io.

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
- Add `ProgressBarWindow()` to show a progress bar in the rendering window
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
examples/simulations/lorenz.py
examples/simulations/gas.py
examples/simulations/aspring2_player.py
```

### Broken Examples
```
airplane1.py
examples/simulations/trail.py
```








