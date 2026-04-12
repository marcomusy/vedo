# API Overview

<img src="https://user-images.githubusercontent.com/32848391/50738810-58af4380-11d8-11e9-8fc7-6c6959207224.jpg" alt="API overview" class="api-overview-hero">

The `vedo` API is organized by domain so you can jump directly to the part of the
library you need: geometry, visualization, interaction, file I/O, plotting, or
global configuration.

Use this page as a quick entry point before diving into the full generated
reference for each module.

## Explore the API

<div class="api-overview-grid">
  <a class="api-card" href="core/">
    <h2>core</h2>
    <p>Shared algorithms, transforms, summaries, and common data helpers.</p>
  </a>
  <a class="api-card" href="mesh/">
    <h2>mesh</h2>
    <p>Polygonal surface data structures, metrics, and mesh-specific operations.</p>
  </a>
  <a class="api-card" href="pointcloud/">
    <h2>pointcloud</h2>
    <p>Point sets, fitting tools, reconstruction, cutting, and geometric analysis.</p>
  </a>
  <a class="api-card" href="grids/">
    <h2>grids</h2>
    <p>Structured, rectilinear, explicit, unstructured, and tetrahedral datasets.</p>
  </a>
  <a class="api-card" href="volume/">
    <h2>volume</h2>
    <p>Volumetric data objects, slicing tools, and image-based processing.</p>
  </a>
  <a class="api-card" href="shapes/">
    <h2>shapes</h2>
    <p>Curves, primitives, glyphs, text, markers, and convenience geometry builders.</p>
  </a>
  <a class="api-card" href="plotter/">
    <h2>plotter</h2>
    <p>Rendering windows, scene management, interaction, camera control, and display.</p>
  </a>
  <a class="api-card" href="visual/">
    <h2>visual</h2>
    <p>Visual mixins, actor appearance, lighting, color mapping, and rendering helpers.</p>
  </a>
  <a class="api-card" href="addons/">
    <h2>addons</h2>
    <p>Axes, scalar bars, widgets, rulers, cutters, sliders, and annotation tools.</p>
  </a>
  <a class="api-card" href="applications/">
    <h2>applications</h2>
    <p>Ready-to-use slicers, browsers, editors, morphing tools, and interactive apps.</p>
  </a>
  <a class="api-card" href="file_io/">
    <h2>file_io</h2>
    <p>Readers, writers, downloads, screenshots, scene export, and video utilities.</p>
  </a>
  <a class="api-card" href="pyplot/">
    <h2>pyplot</h2>
    <p>Figures, charts, statistical plots, graphs, and high-level plotting helpers.</p>
  </a>
  <a class="api-card" href="settings/">
    <h2>settings</h2>
    <p>Global configuration for rendering behavior, defaults, interactivity, and style.</p>
  </a>
</div>

## Common Starting Points

- Start with [plotter](plotter.md) if you want to create windows, render scenes, and interact with objects.
- Start with [mesh](mesh.md), [pointcloud](pointcloud.md), [grids](grids.md), or [volume](volume.md) if you already know your data type.
- Start with [file_io](file_io.md) to load, save, export, or capture scenes.
- Start with [addons](addons.md) and [applications](applications.md) when you need widgets, slicers, rulers, or other higher-level tools.
- Start with [settings](settings.md) to understand global defaults that affect the whole library.
