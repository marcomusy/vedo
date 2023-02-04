## Main changes

- new API docs
- added cli functionality with `vedo --search-code` to inpect the code for keywords
- method `tomesh()` changed to `generate_mesh()`
- fix for `mark_boundaries()`
- use `vtkStaticCellLocator` instead of `vtkCellLocator`
- improved `pointcloud.density()` function
- add `utils.progress_bar()`
- add `trame` examples
- add `utils.is_ragged()` to check if an array is homogeneuos
- fix `shapes.Lines()` class 
- add `shapes.ThickTube()` as per #800
- add `settings.backend_autoclose` (default is True) to automatically close `Plotter` after show 
in jupyter notebooks, #802


-------------------------
## New/Revised Examples
```
examples/other/trame_ex1.py
examples/other/trame_ex2.py
examples/other/trame_ex3.py
examples/simulations/lorenz.py
```

### Broken Examples
