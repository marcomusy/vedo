#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
def get_notebook_backend(actors2show=()):
    """Return the appropriate notebook viewer"""

    #########################################
    if settings.default_backend == "2d":
        return start_2d()

    #########################################
    if settings.default_backend == "k3d":
        return start_k3d(actors2show)

    #########################################
    if settings.default_backend.startswith("trame"):
        return start_trame()

    #########################################
    if settings.default_backend.startswith("ipyvtk"):
        return start_ipyvtklink()

    vedo.logger.error(f"Unknown jupyter backend: {settings.default_backend}")
    return None


#####################################################################################
def start_2d():
    try:
        import PIL.Image
        # import IPython
    except ImportError:
        print("PIL or IPython not available")
        return None

    plt = vedo.plotter_instance

    if hasattr(plt, "window") and plt.window:
        try:
            nn = vedo.file_io.screenshot(asarray=True, scale=1)
            pil_img = PIL.Image.fromarray(nn)
        except ValueError as e:
            return None

        # IPython.display.display(pil_img)
        vedo.notebook_plotter = pil_img
        if settings.backend_autoclose and plt.renderer == plt.renderers[-1]:
            plt.close()
        return pil_img


####################################################################################
def start_k3d(actors2show):

    try:
        # https://github.com/K3D-tools/K3D-jupyter
        import k3d
    except ModuleNotFoundError:
        print("Cannot find k3d, install with:  pip install k3d")
        return None

    plt = vedo.plotter_instance
    if not plt:
        return None

    actors2show2 = []
    for ia in actors2show:
        if not ia:
            continue
        if isinstance(ia, vedo.Assembly):  # unpack assemblies
            actors2show2 += ia.recursive_unpack()
        else:
            actors2show2.append(ia)

    vedo.notebook_plotter = k3d.plot(
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
    vedo.notebook_plotter.camera_auto_fit = settings.k3d_camera_autofit
    vedo.notebook_plotter.grid_auto_fit = settings.k3d_grid_autofit
    vedo.notebook_plotter.axes_helper = settings.k3d_axes_helper

    if plt.camera:
        vedo.notebook_plotter.camera = utils.vtkCameraToK3D(plt.camera)

    if not plt.axes:
        vedo.notebook_plotter.grid_visible = False

    for ia in actors2show2:

        if isinstance(ia, (vtki.vtkCornerAnnotation, vtki.vtkAssembly, vtki.vtkActor2D)):
            continue

        if hasattr(ia, "actor") and isinstance(
            ia.actor, (vtki.vtkCornerAnnotation, vtki.vtkAssembly, vtki.vtkActor2D)
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
            if ia.name:
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

                vtkdata = iapoly.GetPointData()
                vtkscals = vtkdata.GetScalars()

                if vtkscals is None:
                    vtkdata = iapoly.GetCellData()
                    vtkscals = vtkdata.GetScalars()
                    if vtkscals is not None:
                        c2p = vtki.new("CellDataToPointData")
                        c2p.SetInputData(iapoly)
                        c2p.Update()
                        iapoly = c2p.GetOutput()
                        vtkdata = iapoly.GetPointData()
                        vtkscals = vtkdata.GetScalars()

                else:

                    if not vtkscals.GetName():
                        vtkscals.SetName("scalars")
                    scals_min, scals_max = ia.mapper.GetScalarRange()
                    color_attribute = (vtkscals.GetName(), scals_min, scals_max)
                    lut = ia.mapper.GetLookupTable()
                    lut.Build()
                    kcmap = []
                    nlut = lut.GetNumberOfTableValues()
                    for i in range(nlut):
                        r, g, b, _ = lut.GetTableValue(i)
                        kcmap += [i / (nlut - 1), r, g, b]

            else:
                color_attribute = ia.color()

        #####################################################################Volume
        if isinstance(ia, Volume):
            # print('Volume', ia.name, ia.dimensions())
            kx, ky, _ = ia.dimensions()
            arr = ia.pointdata[0]
            kimage = arr.reshape(-1, ky, kx)

            colorTransferFunction = ia.properties.GetRGBTransferFunction()
            kcmap = []
            for i in range(128):
                r, g, b = colorTransferFunction.GetColor(i / 127)
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
            vedo.notebook_plotter += kobj

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
            vedo.notebook_plotter += kobj

        ################################################################# Lines
        elif (
            hasattr(ia, "lines")
            and ia.dataset.GetNumberOfLines()
            and ia.dataset.GetNumberOfPolys() == 0
        ):

            for i, ln_idx in enumerate(ia.lines):

                if i > 200:
                    vedo.logger.warning("in k3d, nr. of lines is limited to 200.")
                    break

                pts = ia.vertices[ln_idx]

                aves = ia.diagonal_size() * iap.GetLineWidth() / 100

                kobj = k3d.line(
                    pts.astype(np.float32),
                    color=_rgb2int(iap.GetColor()),
                    opacity=iap.GetOpacity(),
                    shader=settings.k3d_line_shader,
                    width=aves,
                    name=name,
                )
                vedo.notebook_plotter += kobj

        ################################################################## Mesh
        elif isinstance(ia, Mesh) and ia.npoints and ia.dataset.GetNumberOfPolys():
            # print('Mesh', ia.name, ia.npoints, len(ia.cells))

            if not vtkscals:
                color_attribute = None

            cols = []
            if ia.mapper.GetColorMode() == 0:
                # direct RGB colors

                vcols = ia.dataset.GetPointData().GetScalars()
                if vcols and vcols.GetNumberOfComponents() == 3:
                    cols = utils.vtk2numpy(vcols)
                    cols = 65536 * cols[:, 0] + 256 * cols[:, 1] + cols[:, 2]
                # print("GetColor",iap.GetColor(), _rgb2int(iap.GetColor()) )
                # print("colors", len(cols))
                # print("color_attribute", color_attribute)
                # if kcmap is not None: print("color_map", len(kcmap))
                # TODO:
                # https://k3d-jupyter.org/reference/factory.mesh.html#colormap

                kobj = k3d.mesh(
                    iacloned.vertices,
                    iacloned.cells,
                    colors=cols,
                    name=name,
                    color=_rgb2int(iap.GetColor()),
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

            vedo.notebook_plotter += kobj

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
                ia.vertices.astype(np.float32),
                color=_rgb2int(iap.GetColor()),
                colors=kcols,
                opacity=iap.GetOpacity(),
                shader=settings.k3d_point_shader,
                point_size=aves,
                name=name,
            )
            vedo.notebook_plotter += kobj

        #####################################################################
        elif isinstance(ia, vedo.Image):
            vedo.logger.error("Sorry Image objects are not supported in k3d.")

    if plt and settings.backend_autoclose:
        plt.close()
    return vedo.notebook_plotter


#####################################################################################
def start_trame():

    try:
        from trame.app import get_server, jupyter
        from trame.ui.vuetify import VAppLayout
        from trame.widgets import vtk as t_vtk, vuetify
    except ImportError:
        print("trame is not installed, try:\n> pip install trame==2.5.2")
        return None

    plt = vedo.plotter_instance
    if hasattr(plt, "window") and plt.window:
        plt.renderer.ResetCamera()
        server = get_server("jupyter-1")
        state, ctrl = server.state, server.controller
        plt.server = server
        plt.controller = ctrl
        plt.state = state

        with VAppLayout(server) as layout:

            with layout.root:

                with vuetify.VContainer(fluid=True, classes="pa-0 fill-height"):
                    plt.reset_camera()
                    view = t_vtk.VtkLocalView(plt.window)
                    ctrl.view_update = view.update
                    ctrl.view_reset_camera = view.reset_camera

        ctrl.on_server_exited.add(lambda **_: print("trame server exited"))
        vedo.notebook_plotter = jupyter.show(server)
        return vedo.notebook_plotter
    vedo.logger.error("No window present for the trame backend.")
    return None


#####################################################################################
def start_ipyvtklink():
    try:
        from ipyvtklink.viewer import ViewInteractiveWidget
    except ImportError:
        print("ipyvtklink is not installed, try:\n> pip install ipyvtklink")
        return None

    plt = vedo.plotter_instance
    if hasattr(plt, "window") and plt.window:
        plt.renderer.ResetCamera()
        vedo.notebook_plotter = ViewInteractiveWidget(
            plt.window, allow_wheel=True, quality=100, quick_quality=50
        )
        return vedo.notebook_plotter
    vedo.logger.error("No window present for the ipyvtklink backend.")
    return None


#####################################################################################
def _rgb2int(rgb_tuple):
    # Return the int number of a color from (r,g,b), with 0<r<1 etc.
    rgb = (int(rgb_tuple[0] * 255), int(rgb_tuple[1] * 255), int(rgb_tuple[2] * 255))
    return 65536 * rgb[0] + 256 * rgb[1] + rgb[2]
