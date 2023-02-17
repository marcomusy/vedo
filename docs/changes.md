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
- add `settings.backend_autoclose` (default is True) to automatically close `Plotter` after show 
in jupyter notebooks, #802
- fix k3d backend - remove pinning to 2.7.4 #808
- in `Text2D` and `Text3D` shortcut to symbols are now formatted as `:gamma` and no more as `\gamma`
- fix `mesh.stretch()` thanks to @mikaeltulldahl in #807
- fixes to `cmap()`

-------------------------
## New/Revised Examples
```
examples/other/trame_ex1.py
examples/other/trame_ex2.py
examples/other/trame_ex3.py
examples/simulations/lorenz.py
examples/notebooks/slider2d.ipynb
```

### Broken Examples


