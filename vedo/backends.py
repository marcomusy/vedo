#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from importlib import import_module
import os
import numpy as np

import vedo.vtkclasses as vtki

from vedo.pointcloud import Points
from vedo.mesh import Mesh
from vedo.volume import Volume

import vedo
from vedo import settings
from vedo import utils

__doc__ = """Submodule to delegate jupyter notebook rendering"""

__all__ = []


############################################################################################
def _import_trame_components():
    """Import trame modules across legacy and current package layouts."""
    try:
        from trame.app import get_server  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Trame support requires the 'trame' package. Install with:\n"
            "> pip install trame trame-vtk trame-vuetify"
        ) from exc

    layout_paths = [
        ("trame.ui.vuetify3", "VAppLayout", "vue3"),
        ("trame.ui.vuetify", "VAppLayout", "vue2"),
    ]
    vuetify_paths = [
        ("trame.widgets.vuetify3", "vue3"),
        ("trame.widgets.vuetify", "vue2"),
    ]

    def _safe_import(module_name):
        try:
            return import_module(module_name)
        except ModuleNotFoundError as exc:
            # Fall back only when the candidate package itself is missing.
            if exc.name == module_name:
                return None
            raise

    VAppLayout = None
    client_type = None
    for module_name, attr_name, layout_client_type in layout_paths:
        module = _safe_import(module_name)
        if module is None:
            continue
        try:
            VAppLayout = getattr(module, attr_name)
        except AttributeError as exc:
            raise ImportError(
                f"Trame layout module {module_name!r} does not provide {attr_name!r}. "
                "This trame installation is incompatible with vedo."
            ) from exc
        client_type = layout_client_type
        break

    t_vtk = _safe_import("trame.widgets.vtk")

    vuetify = None
    vuetify_client_type = None
    for module_name, current_client_type in vuetify_paths:
        vuetify = _safe_import(module_name)
        if vuetify is None:
            continue
        vuetify_client_type = current_client_type
        break

    if (
        VAppLayout is not None
        and t_vtk is not None
        and vuetify is not None
        and client_type is not None
    ):
        if client_type != vuetify_client_type:
            raise ImportError(
                f"Trame version mismatch: layout loaded as {client_type} but "
                f"vuetify widgets loaded as {vuetify_client_type}. "
                "Ensure trame-vuetify version matches your trame installation."
            )
        return get_server, VAppLayout, t_vtk, vuetify, client_type

    missing = []
    if VAppLayout is None or vuetify is None:
        missing.append("trame-vuetify")
    if t_vtk is None:
        missing.append("trame-vtk")
    missing = sorted(set(missing))

    raise ImportError(
        "Trame backend requires additional widget packages. Install with:\n"
        f"> pip install trame {' '.join(missing)}"
    )


############################################################################################
def _warn_trame_xopengl(render_window):
    """Warn about unstable interactive trame rendering on X/GLX offscreen VTK."""
    if not render_window:
        return
    if render_window.GetClassName() == "vtkXOpenGLRenderWindow":
        vedo.logger.warning(
            "Trame on vtkXOpenGLRenderWindow can segfault during interactive rendering. "
            "Prefer VtkLocalView or use a VTK build with EGL/OSMesa for remote rendering."
        )


############################################################################################
def get_notebook_backend(actors2show=()):
    """Return the appropriate notebook viewer.
    `actors2show` is only forwarded to the k3d backend; other backends
    read actors directly from the active Plotter.
    """
    backend = settings.default_backend

    if not backend:
        vedo.logger.error("No jupyter backend configured (settings.default_backend is empty).")
        return None
    if not isinstance(backend, str):
        vedo.logger.error(
            "Invalid jupyter backend configuration: "
            f"settings.default_backend must be a string, got {type(backend).__name__}."
        )
        return None

    backend = backend.strip().lower()

    if backend == "2d":
        return start_2d()
    if backend == "k3d":
        return start_k3d(actors2show)
    if backend.startswith("trame"):
        return start_trame()
    if backend.startswith("ipyvtk"):
        return start_ipyvtklink()
    if backend == "panel":
        return start_panel()

    vedo.logger.error(f"Unknown jupyter backend: {backend!r}")
    return None


#####################################################################################
def start_2d():
    """Start a 2D display in the notebook"""
    try:
        import PIL.Image
    except ImportError:
        vedo.logger.error("Pillow is not installed, try:\n> pip install Pillow")
        return None

    plt = vedo.current_plotter()
    if not plt:
        vedo.logger.error("No active Plotter found for the 2d backend.")
        return None

    if hasattr(plt, "window") and plt.window:
        try:
            image_array = vedo.file_io.screenshot(asarray=True, scale=1)
            pil_img = PIL.Image.fromarray(image_array)
        except Exception as e:
            vedo.logger.warning(f"2d backend screenshot failed: {e}")
            return None

        vedo.set_current_notebook_plotter(pil_img)
        renderers = getattr(plt, "renderers", ())
        if settings.backend_autoclose and renderers and plt.renderer == renderers[-1]:
            plt.close()
        return pil_img

    vedo.logger.error("No window present for the 2d backend.")
    return None


#####################################################################################
def start_panel():
    """Start a panel display in the notebook"""
    try:
        import panel as pn  # type: ignore

        pn.extension(
            "vtk", design="material", sizing_mode="stretch_width", template="material"
        )
    except ImportError:
        vedo.logger.error("panel is not installed, try:\n> conda install panel")
        return None

    vedo.logger.warning("panel backend is experimental and may not render correctly.")
    plt = vedo.current_plotter()
    if not plt:
        vedo.logger.error("No active Plotter found for the panel backend.")
        return None

    if hasattr(plt, "window") and plt.window:
        plt.renderer.ResetCamera()
        vtkpan = pn.pane.VTK(
            plt.window,
            margin=0,
            sizing_mode="stretch_both",
            min_height=600,
            orientation_widget=True,
            enable_keybindings=True,
        )
        vedo.set_current_notebook_plotter(vtkpan)
        return vtkpan

    vedo.logger.error("No window present for the panel backend.")
    return None


####################################################################################
def start_k3d(actors2show):
    """Start a k3d display in the notebook"""
    try:
        # https://github.com/K3D-tools/K3D-jupyter
        import k3d
    except ModuleNotFoundError:
        print("\nCannot find k3d, install with:  pip install k3d")
        return None

    plt = vedo.current_plotter()
    if not plt:
        vedo.logger.error("No active Plotter found for the k3d backend.")
        return None

    def _setup_scalar_metadata(polydata, mapper):
        """Build scalar metadata for k3d color mapping."""
        vtkdata = polydata.GetPointData()
        vtkscals = vtkdata.GetScalars()

        if vtkscals is None:
            vtkdata = polydata.GetCellData()
            vtkscals = vtkdata.GetScalars()
            if vtkscals is not None:
                c2p = vtki.new("CellDataToPointData")
                c2p.SetInputData(polydata)
                c2p.Update()
                polydata = c2p.GetOutput()
                vtkdata = polydata.GetPointData()
                vtkscals = vtkdata.GetScalars()

        if vtkscals is None:
            return polydata, None, None, None, None

        if not vtkscals.GetName():
            vtkscals.SetName("scalars")
        scals_min, scals_max = mapper.GetScalarRange()
        color_attribute = (vtkscals.GetName(), scals_min, scals_max)
        lut = mapper.GetLookupTable()
        if lut is None:
            return polydata, vtkscals, color_attribute, None, (scals_min, scals_max)
        lut.Build()
        kcmap = []
        nlut = lut.GetNumberOfTableValues()
        if nlut <= 0:
            return polydata, vtkscals, color_attribute, None, (scals_min, scals_max)
        lut_scale = max(nlut - 1, 1)
        for i in range(nlut):
            r, g, b, _ = lut.GetTableValue(i)
            kcmap += [i / lut_scale, r, g, b]
        return polydata, vtkscals, color_attribute, kcmap, (scals_min, scals_max)

    already_has_axes = False
    actors2show2 = []
    for ia in actors2show:
        if not ia:
            continue

        try:
            if ia.name == "Axes":
                already_has_axes = True
        except AttributeError:
            pass

        if isinstance(ia, vedo.Assembly):  # unpack assemblies
            actors2show2 += ia.recursive_unpack()
        else:
            actors2show2.append(ia)

    nbplot = k3d.plot(
        axes=["x", "y", "z"],
        menu_visibility=settings.k3d_menu_visibility,
        height=settings.k3d_plot_height,
        antialias=settings.k3d_antialias,
        background_color=_rgb2int(vedo.get_color(plt.backgrcol)),
        camera_fov=30.0,  # deg (this is the vtk default)
        lighting=settings.k3d_lighting,
        grid_color=_rgb2int(vedo.get_color(settings.k3d_axes_color)),
        label_color=_rgb2int(vedo.get_color(settings.k3d_axes_color)),
        axes_helper=settings.k3d_axes_helper,
    )

    # set k3d camera
    vedo.set_current_notebook_plotter(nbplot)
    nbplot.camera_auto_fit = settings.k3d_camera_autofit
    nbplot.axes_helper = settings.k3d_axes_helper
    nbplot.grid_auto_fit = settings.k3d_grid_autofit

    if already_has_axes:
        nbplot.grid_visible = False
    if settings.k3d_grid_visible is not None:  # override if set
        nbplot.grid_visible = settings.k3d_grid_visible

    if plt.camera:
        nbplot.camera = utils.vtkCameraToK3D(plt.camera)

    for ia in actors2show2:
        if isinstance(ia, (vtki.vtkAssembly, vtki.vtkActor2D)):
            continue

        if hasattr(ia, "actor") and isinstance(
            ia.actor, (vtki.vtkAssembly, vtki.vtkActor2D)
        ):
            continue

        iacloned = ia

        kobj = None
        kcmap = None
        color_attribute = None
        vtkscals = None
        name = None
        if hasattr(ia, "filename"):
            if ia.filename:
                name = os.path.basename(ia.filename)
            if hasattr(ia, "name") and ia.name:
                name = os.path.basename(ia.name)

        ################################################################## scalars
        # work out scalars first, Points Lines are also Mesh objs
        if isinstance(ia, Points):
            # print('scalars', ia.name, ia.npoints)
            iap = ia.properties

            if ia.dataset.GetNumberOfPolys():
                iacloned = ia.clone()
                iapoly = iacloned.clean().triangulate().compute_normals().dataset
            else:
                iapoly = ia.dataset

            if ia.mapper.GetScalarVisibility() and ia.mapper.GetColorMode() > 0:
                iapoly, vtkscals, color_attribute, kcmap, scal_range = (
                    _setup_scalar_metadata(iapoly, ia.mapper)
                )
                if scal_range is not None:
                    scals_min, scals_max = scal_range

            else:
                color_attribute = ia.color()

        #####################################################################Volume
        if isinstance(ia, Volume):
            # print('Volume', ia.name, ia.dimensions())
            kx, ky, _ = ia.dimensions()
            arr = ia.pointdata[0]
            kimage = arr.reshape(-1, ky, kx)

            color_transfer_function = ia.properties.GetRGBTransferFunction()
            kcmap = []
            for i in range(128):
                r, g, b = color_transfer_function.GetColor(i / 127)
                kcmap += [i / 127, r, g, b]

            kbounds = np.array(ia.dataset.GetBounds()) + np.repeat(
                np.array(ia.dataset.GetSpacing()) / 2.0, 2
            ) * np.array([-1, 1] * 3)

            kobj = k3d.volume(
                kimage.astype(np.float32),
                color_map=kcmap,
                # color_range=ia.dataset.GetScalarRange(),
                alpha_coef=10,
                bounds=kbounds,
                name=name,
            )
            nbplot += kobj

        ################################################################ Text2D
        elif isinstance(ia, vedo.Text2D):
            # print('Text2D', ia.GetPosition())
            pos = (ia.GetPosition()[0], 1.0 - ia.GetPosition()[1])

            kobj = k3d.text2d(
                ia.text(),
                position=pos,
                color=_rgb2int(vedo.get_color(ia.c())),
                is_html=True,
                size=ia.properties.GetFontSize() / 22.5 * 1.5,
                label_box=bool(ia.properties.GetFrame()),
                # reference_point='bl',
            )
            nbplot += kobj

        ################################################################# Lines
        elif (
            hasattr(ia, "lines")
            and ia.dataset.GetNumberOfLines()
            and ia.dataset.GetNumberOfPolys() == 0
        ):
            for i, ln_idx in enumerate(ia.lines):
                if i >= 200:
                    vedo.logger.warning("in k3d, nr. of lines is limited to 200.")
                    break

                pts = ia.coordinates[ln_idx]

                aves = ia.diagonal_size() * iap.GetLineWidth() / 100

                kobj = k3d.line(
                    pts.astype(np.float32),
                    color=_rgb2int(iap.GetColor()),
                    opacity=iap.GetOpacity(),
                    shader=settings.k3d_line_shader,
                    width=aves.astype(float),
                    name=name,
                )
                nbplot += kobj

        ################################################################## Mesh
        elif isinstance(ia, Mesh) and ia.npoints and ia.dataset.GetNumberOfPolys():
            # print('Mesh', ia.name, ia.npoints, len(ia.cells))

            if not vtkscals:
                color_attribute = None

            cols = []
            if ia.mapper.GetColorMode() == 2:  # direct RGB colors
                vcols = ia.dataset.GetPointData().GetScalars()

                if not vcols:
                    iacloned = ia.clone()
                    iacloned.map_cells_to_points()
                    vcols = iacloned.dataset.GetPointData().GetScalars()

                if vcols and vcols.GetNumberOfComponents() in (3, 4):
                    # vedo.logger.info("found RGB direct colors in Mesh")
                    cols = utils.vtk2numpy(vcols).astype(np.uint32)
                    cols = 65536 * cols[:, 0] + 256 * cols[:, 1] + cols[:, 2]

                    kobj = k3d.mesh(
                        iacloned.coordinates,
                        iacloned.cells,
                        colors=cols,
                        name=name,
                        opacity=iap.GetOpacity(),
                        side="double",
                        wireframe=(iap.GetRepresentation() == 1),
                    )

                else:
                    vedo.logger.warning("could not find RGB direct colors in Mesh")
                    kobj = k3d.mesh(
                        iacloned.coordinates,
                        iacloned.cells,
                        color=_rgb2int(iap.GetColor()),
                        name=name,
                        opacity=iap.GetOpacity(),
                        side="double",
                        wireframe=(iap.GetRepresentation() == 1),
                    )

            else:
                kobj = k3d.vtk_poly_data(
                    iapoly,
                    name=name,
                    color=_rgb2int(iap.GetColor()),
                    color_attribute=color_attribute,
                    color_map=kcmap,
                    opacity=iap.GetOpacity(),
                    side="double",
                    wireframe=(iap.GetRepresentation() == 1),
                )

            if iap.GetInterpolation() == 0:
                kobj.flat_shading = True

            nbplot += kobj

        #####################################################################Points
        elif isinstance(ia, Points):
            # print('Points', ia.name, ia.npoints)
            kcols = []
            if kcmap is not None and vtkscals:
                scals = utils.vtk2numpy(vtkscals)
                kcols = k3d.helpers.map_colors(
                    scals, kcmap, [scals_min, scals_max]
                ).astype(np.uint32)

            aves = ia.average_size() * iap.GetPointSize() / 200

            kobj = k3d.points(
                ia.coordinates.astype(np.float32),
                color=_rgb2int(iap.GetColor()),
                colors=kcols,
                opacity=iap.GetOpacity(),
                shader=settings.k3d_point_shader,
                point_size=aves.astype(float),
                name=name,
            )
            nbplot += kobj

        #####################################################################
        elif isinstance(ia, vedo.Image):
            vedo.logger.error("Sorry Image objects are not supported in k3d.")

    if plt and settings.backend_autoclose:
        plt.close()
    return nbplot


#####################################################################################
def start_trame():
    """Start a trame display in the notebook"""
    try:
        get_server, VAppLayout, t_vtk, vuetify, client_type = _import_trame_components()
    except ImportError as exc:
        print(exc)
        return None

    plt = vedo.current_plotter()
    if not plt:
        vedo.logger.error("No active Plotter found for the trame backend.")
        return None
    if hasattr(plt, "window") and plt.window:
        render_window = plt.window
        _warn_trame_xopengl(render_window)
        plt.renderer.ResetCamera()
        server_name = f"vedo-jupyter-{id(plt)}"
        view_ref = f"vedo_trame_view_{id(plt)}"
        scene_state_key = f"scene_{view_ref}"
        server = get_server(server_name, client_type=client_type)
        state, ctrl = server.state, server.controller
        plt.server = server
        plt.controller = ctrl
        plt.state = state

        with VAppLayout(server) as layout:
            with layout.root:
                with vuetify.VContainer(fluid=True, classes="pa-0 fill-height"):
                    plt.reset_camera()
                    server.state[scene_state_key] = {}
                    view = t_vtk.VtkLocalView(render_window, ref=view_ref)
                    ctrl.view_update = view.update
                    ctrl.view_reset_camera = view.reset_camera
                    ctrl.view_update()

        ctrl.on_server_exited.add(lambda **_: print("trame server exited"))
        vedo.set_current_notebook_plotter(layout)
        return layout
    vedo.logger.error("No window present for the trame backend.")
    return None


#####################################################################################
def start_ipyvtklink():
    try:
        from ipyvtklink.viewer import ViewInteractiveWidget  # type: ignore
    except ImportError:
        print("ipyvtklink is not installed, try:\n> pip install ipyvtklink")
        return None

    plt = vedo.current_plotter()
    if not plt:
        vedo.logger.error("No active Plotter found for the ipyvtklink backend.")
        return None
    if hasattr(plt, "window") and plt.window:
        plt.renderer.ResetCamera()
        nbplot = ViewInteractiveWidget(
            plt.window, allow_wheel=True, quality=100, quick_quality=50
        )
        vedo.set_current_notebook_plotter(nbplot)
        return nbplot
    vedo.logger.error("No window present for the ipyvtklink backend.")
    return None


#####################################################################################
def _rgb2int(rgb_tuple):
    """Return the packed RGB integer from normalized color components."""
    rgb = np.asarray(rgb_tuple, dtype=float).ravel()
    if rgb.size < 3:
        raise ValueError("rgb_tuple must contain at least 3 components.")
    rgb = np.clip(rgb[:3], 0.0, 1.0)
    rgb = np.rint(rgb * 255).astype(np.uint32)
    return int(65536 * rgb[0] + 256 * rgb[1] + rgb[2])
