## Main changes

- new API docs
- added cli functionality with `vedo --search-code` to inpect the code for keywords
- method `tomesh()` changed to `generate_mesh()`
- fix for `mark_boundaries()`
- use `vtkStaticCellLocator` instead of `vtkCellLocator`
- improved `pointcloud.density()` function
- add `utils.progressbar()`
- add `trame` examples
- add `utils.is_ragged()` to check if an array is homogeneuos (not ragged)
- fix `shapes.Lines()` class 
- add `shapes.ThickTube()` as per #800
- add `settings.backend_autoclose` (default is True) to automatically close `Plotter` after show() in jupyter notebooks, #802
- fix k3d backend - remove pinning to 2.7.4 #808
- in `Text2D` and `Text3D` shortcut to symbols are now formatted as `:gamma` and no more as `\gamma`
- fix `mesh.stretch()` thanks to @mikaeltulldahl in #807
- fixes to `cmap()`
- added `mesh.is_manifold()` method #813
- added `utils.open3d2vedo()` and `utils.vedo2open3d` converters as per #663
- added `mesh.non_manifold_faces()` to detect and (try to) remove non-manifold faces of a triangular mesh #813 #663 and https://github.com/SlicerIGT/SlicerBoneReconstructionPlanner/issues/35
- added `pointcloud.compute_acoplanarity()` method
- add pipeline visualization with syntax `any_vedo_obj.pipeline.show()` or interactively by clicking an object and pressing `y`.
- fixes to `Button` class
- in `map_points_to_cells()` added keyword `move` in #819 @19kdc3
- fix `flagpost()` by creating a new class `addons.Flagpost()` in #821 @ZiguoAtGitHub
- add `vedo.io.Video(backend='imageio')` support for video generation
- add `vedo.io.Video.split_frames()` of an already existing video file.


-------------------------
## New/Revised Examples
```
examples/other/trame_ex1.py
examples/other/trame_ex2.py
examples/other/trame_ex3.py
examples/simulations/lorenz.py
examples/notebooks/slider2d.ipynb
examples/basic/mousehover0.py.py
```

### Broken Examples


