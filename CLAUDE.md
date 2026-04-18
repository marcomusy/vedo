# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is vedo

vedo is a scientific visualization library built on top of VTK, providing a Pythonic API for 3D/2D rendering, mesh processing, volume rendering, and interactive plotting.

## Commands

```bash
pip install -e .                        # Install in development mode
pytest tests/common                     # Run the main test suite
pytest tests/common/test_0_imports.py   # Run a single test file
pytest tests/issues                     # Run issue-specific regression tests
ruff check .                            # Lint
vedo <file>                             # Visualize a file from the CLI
vedo -r align1                          # Run examples/basic/align1.py by name
```

Pytest config lives in `pyproject.toml`. Default test paths: `tests/common`. `tests/issues` and `tests/snippets` are excluded by default.

Ruff silences `F401` (unused imports), `E402`, `E501`, `E701` — match this style.

## Architecture

### Module layout

```
vedo/
├── core/               # Algorithm mixins (no rendering)
│   ├── common.py       # CommonAlgorithms — geometry ops on all objects
│   ├── points.py       # PointAlgorithms mixin
│   ├── volume.py       # VolumeAlgorithms mixin
│   └── data.py         # DataArrayHelper for point/cell/metadata arrays
├── visual/             # Rendering/visual property mixins
│   └── runtime.py      # CommonVisual, PointsVisual, MeshVisual, VolumeVisual, ImageVisual
├── plotter/            # Window management
│   └── runtime.py      # Plotter class; scene, camera, interaction, lifecycle split into siblings
├── mesh/core.py        # Mesh class
├── pointcloud/core.py  # Points class
├── volume/core.py      # Volume class
├── grids/              # TetMesh, UnstructuredGrid, StructuredGrid, Image, etc.
├── shapes/             # Primitives, curves, text, glyphs
├── pyplot/             # 2D/3D plotting (Plot, histograms, etc.)
├── addons/             # UI widgets: Slider, Button, Axes, ScalarBar, cutting tools
├── applications/       # Domain tools: slicer.py, chemistry.py
├── assembly.py         # Group and Assembly containers
├── settings.py         # Global vedo.settings object
├── colors.py           # Color utilities and colormaps
├── utils.py            # General utilities (~100KB)
├── vtkclasses.py       # Centralized lazy VTK imports
└── __init__.py         # Lazy export map (nothing loads until accessed)
```

### Class hierarchy

All visual objects inherit from `CommonVisual` (in `visual/runtime.py`). Features are assembled via multiple inheritance from mixins:

```
CommonVisual
├── PointsVisual  →  Points  (point cloud)
│                     └── Mesh  (polygonal surface, also inherits MeshVisual)
├── VolumeVisual  →  Volume
└── ImageVisual   →  Image
```

Every object exposes three VTK handles:
- `obj.dataset` — the underlying VTK data object (vtkPolyData, vtkImageData, …)
- `obj.mapper`  — VTK mapper
- `obj.actor`   — VTK actor/prop for scene placement

### Data arrays

Point and cell scalar/vector data are accessed through `DataArrayHelper`:

```python
obj.pointdata["density"] = np.array([...])   # assign
values = obj.pointdata["density"]             # retrieve as numpy
obj.pointdata.select("density")              # make active scalar
# same API for obj.celldata and obj.metadata
```

### Plotter

`Plotter` (in `plotter/runtime.py`) manages the render window. Its methods are split across sibling files: `scene.py`, `camera.py`, `interaction.py`, `lifecycle.py`, `io.py`, `keymap.py`.

Multiple renderers in one window: `Plotter(shape=(2,2))`. Route objects to a renderer with `plt.show(obj, at=1)` or `plt.add(obj, render=False)`.

### Lazy loading

`vedo/__init__.py` builds a lazy attribute map — submodules load only when first accessed. `vedo/vtkclasses.py` does the same for VTK imports. Never add top-level eager imports to `__init__.py`.

### Settings

`vedo.settings` is a global config object. Set it **before** creating objects; it is applied at construction time, not retroactively. Supports both attribute and dict-style access.

### VTK wrapping convention

Always import VTK through `vedo.vtkclasses` (e.g., `from vedo.vtkclasses import vtk`), not directly from `vtkmodules`. This keeps version handling centralised and preserves lazy loading.
