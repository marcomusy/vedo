## Main changes

### Breaking changes
- method `plotter.add()` no more accepts keyword `render=True/False`. Please use `plotter.add().render()` explicitly. Same thing for `plotter.remove()`.

### Other fixes and improvements
- added gpu acceleration for CLI volumetric visualization in #832 by @zhang-qiang-github
- fixes for `k3d`
- added `plotter.fov(value)` (field of view)


-------------------------
## New/Revised Examples
```
examples/advanced/warp6.py

```

### Broken Examples

