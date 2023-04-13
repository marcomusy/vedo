## Main changes

### Breaking changes
- method `plotter.add()` no more accepts keyword `render=True/False`. Please use `plotter.add().render()` explicitly. Same thing for `plotter.remove()`.

### Other fixes and improvements
- added gpu acceleration for CLI volumetric visualization in #832 by @zhang-qiang-github
- fixes for `k3d`
- added `plotter.fov(value)` (field of view)
- fix `ploter.get_meshes()`
- fix for `plotter.remove(unpack_assemblies=False)` method
- fix for `io.import_window()` method
- added `cut_with_cookiecutter()` to cut 2D contours.
- fix `shapes.NormalLines()` class
- added `vedo.interactor_modes` module
- added `vedo.interactor_modes.BlenderStyle` class
- added `base.pointdata.clear()` to remove all associated data 
- added `volume.hide_voxels()` for visualization


-------------------------
## New/Revised Examples
```
examples/basic/sliders_range.py
examples/basic/interaction_modes.py
examples/advanced/warp6.py
examples/pyplot/histo_1d_e.py
```

### Broken Examples

