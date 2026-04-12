from __future__ import annotations

"""Scene import/export, screenshots, and terminal interaction helpers."""

import os
from typing import Any

import numpy as np

import vedo
import vedo.vtkclasses as vtki
from vedo import colors, settings, utils
from vedo.assembly import Assembly
from vedo.grids.image import Image
from vedo.pointcloud import Points
from vedo.volume import Volume

from .loaders import _import_npy

__docformat__ = "google"
__all__ = [
    "export_window",
    "to_numpy",
    "_export_npy",
    "import_window",
    "screenshot",
]


def export_window(
    fileoutput: str | os.PathLike,
    binary=False,
    plt=None,
    backend: str | None = None,
    backend_options: dict | None = None,
) -> vedo.Plotter | None:
    """
    Exporter which writes out the rendered scene into an HTML, X3D or Numpy file.

    Examples:
        - [export_x3d.py](https://github.com/marcomusy/vedo/tree/master/examples/other/export_x3d.py)

        Check out the HTML generated webpage [here](https://vedo.embl.es/examples/embryo.html).

        <img src='https://user-images.githubusercontent.com/32848391/57160341-c6ffbd80-6de8-11e9-95ff-7215ce642bc5.jpg' width="600"/>

    .. note::
        the rendering window can also be exported to `numpy` file `scene.npz`
        by pressing `E` key at any moment during visualization.

        For `.html` exports, the default backend remains `k3d`.
        Pass `backend="threejs"` to generate a standalone Three.js scene page instead.
        Optional `backend_options` can tune the Three.js material mapping, e.g.
        `{"headlight_intensity": 1.1, "specular_scale": 0.55, "preserve_base_color": True}`.
    """
    fileoutput = str(fileoutput)
    if plt is None:
        plt = vedo.current_plotter()
    if plt is None:
        vedo.logger.error("export_window(): no active Plotter found")
        return None

    fr = fileoutput.lower()
    ####################################################################
    if fr.endswith(".npy") or fr.endswith(".npz"):
        _export_npy(plt, fileoutput)

    ####################################################################
    elif fr.endswith(".x3d"):
        _export_x3d(plt, fileoutput, binary=binary)

    ####################################################################
    elif fr.endswith(".html"):
        html_backend = "k3d" if backend is None else backend.lower()
        if html_backend == "threejs":
            _export_threejs(plt, fileoutput, backend_options=backend_options)
        elif html_backend == "k3d":
            savebk = vedo.current_notebook_backend()
            try:
                vedo.set_current_notebook_backend("k3d")
                vedo.settings.default_backend = "k3d"
                backend_obj = vedo.backends.get_notebook_backend(plt.objects)
                with open(fileoutput, "w", encoding="UTF-8") as fp:
                    fp.write(backend_obj.get_snapshot())
            finally:
                vedo.set_current_notebook_backend(savebk)
                vedo.settings.default_backend = savebk
        else:
            vedo.logger.error(f"Unsupported html export backend '{backend}'")

    else:
        vedo.logger.error(f"export extension {fr.split('.')[-1]} is not supported")

    return plt


#########################################################################
def to_numpy(act: Any) -> dict:
    """Encode a vedo object to numpy format."""

    ########################################################
    def _fillcommon(obj, adict):
        adict["filename"] = obj.filename
        adict["name"] = obj.name
        adict["time"] = obj.time
        adict["rendered_at"] = obj.rendered_at
        try:
            adict["transform"] = obj.transform.matrix
        except AttributeError:
            adict["transform"] = np.eye(4)

    ####################################################################
    try:
        obj = act.retrieve_object()
    except AttributeError:
        obj = act

    adict = {}
    adict["type"] = "unknown"

    ######################################################## Points/Mesh
    if isinstance(obj, (Points, vedo.UnstructuredGrid)):
        adict["type"] = "Mesh"
        _fillcommon(obj, adict)

        if isinstance(obj, vedo.UnstructuredGrid):
            # adict["type"] = "UnstructuredGrid"
            # adict["cells"] = obj.cells_as_flat_array
            poly = obj._actor.GetMapper().GetInput()
            mapper = obj._actor.GetMapper()
        else:
            poly = obj.dataset
            mapper = obj.mapper

        # Keep geometry arrays at single precision to reduce exported scene size.
        adict["points"] = obj.vertices.astype(np.float32, copy=False)

        adict["cells"] = None
        if poly.GetNumberOfPolys():
            adict["cells"] = obj.cells_as_flat_array

        adict["lines"] = None
        if poly.GetNumberOfLines():
            adict["lines"] = obj.lines  # _as_flat_array

        adict["pointdata"] = {}
        for iname in obj.pointdata.keys():
            if "normals" in iname.lower():
                continue
            adict["pointdata"][iname] = obj.pointdata[iname]

        adict["celldata"] = {}
        for iname in obj.celldata.keys():
            if "normals" in iname.lower():
                continue
            adict["celldata"][iname] = obj.celldata[iname]

        adict["metadata"] = {}
        for iname in obj.metadata.keys():
            adict["metadata"][iname] = obj.metadata[iname]

        adict["point_normals"] = None
        normals = poly.GetPointData().GetNormals()
        if normals:
            adict["point_normals"] = vedo.vtk2numpy(normals).astype(
                np.float32, copy=False
            )

        adict["texture_coordinates"] = None
        tcoords = poly.GetPointData().GetTCoords()
        if tcoords:
            adict["texture_coordinates"] = vedo.vtk2numpy(tcoords).astype(
                np.float32, copy=False
            )

        # NEW in vedo 5.0
        adict["scalar_mode"] = mapper.GetScalarMode()
        adict["array_name_to_color_by"] = mapper.GetArrayName()
        adict["color_mode"] = mapper.GetColorMode()
        adict["interpolate_scalars_before_mapping"] = (
            mapper.GetInterpolateScalarsBeforeMapping()
        )
        adict["use_lookup_table_scalar_range"] = mapper.GetUseLookupTableScalarRange()
        adict["scalar_range"] = mapper.GetScalarRange()
        adict["scalar_visibility"] = mapper.GetScalarVisibility()
        adict["pickable"] = obj.actor.GetPickable()
        adict["dragable"] = obj.actor.GetDragable()

        # adict["color_map_colors"]  = mapper.GetColorMapColors()   #vtkUnsignedCharArray
        # adict["color_coordinates"] = mapper.GetColorCoordinates() #vtkFloatArray
        texmap = mapper.GetColorTextureMap()  # vtkImageData
        if texmap:
            adict["color_texture_map"] = vedo.Image(texmap).tonumpy()
            # print("color_texture_map", adict["color_texture_map"].shape)

        adict["texture_array"] = None
        texture = obj.actor.GetTexture()
        if texture:
            adict["texture_array"] = vedo.Image(texture.GetInput()).tonumpy()
            adict["texture_interpolate"] = texture.GetInterpolate()
            adict["texture_repeat"] = texture.GetRepeat()
            adict["texture_quality"] = texture.GetQuality()
            adict["texture_color_mode"] = texture.GetColorMode()
            adict["texture_mipmap"] = texture.GetMipmap()
            adict["texture_blending_mode"] = texture.GetBlendingMode()
            adict["texture_edge_clamp"] = texture.GetEdgeClamp()
            adict["texture_border_color"] = texture.GetBorderColor()
            # print("tonumpy: texture", obj.name, adict["texture_array"].shape)

        adict["LUT"] = None
        adict["LUT_range"] = None
        lut = mapper.GetLookupTable()
        if lut:
            nlut = lut.GetNumberOfTableValues()
            lutvals = []
            for i in range(nlut):
                v4 = lut.GetTableValue(i)  # (r, g, b, alpha)
                lutvals.append(v4)
            adict["LUT"] = np.array(lutvals, dtype=np.float32)
            adict["LUT_range"] = np.array(lut.GetRange())

        prp = obj.properties
        adict["alpha"] = prp.GetOpacity()
        adict["representation"] = prp.GetRepresentation()
        adict["pointsize"] = prp.GetPointSize()

        adict["linecolor"] = None
        adict["linewidth"] = None
        adict["edge_visibility"] = prp.GetEdgeVisibility()  # new in vedo 5.0
        if prp.GetEdgeVisibility():
            adict["linewidth"] = prp.GetLineWidth()
            adict["linecolor"] = prp.GetEdgeColor()

        adict["ambient"] = prp.GetAmbient()
        adict["diffuse"] = prp.GetDiffuse()
        adict["specular"] = prp.GetSpecular()
        adict["specularpower"] = prp.GetSpecularPower()
        adict["specularcolor"] = prp.GetSpecularColor()
        adict["shading"] = prp.GetInterpolation()  # flat phong..:
        adict["color"] = prp.GetColor()
        adict["lighting_is_on"] = prp.GetLighting()
        adict["backcolor"] = None
        if obj.actor.GetBackfaceProperty():
            adict["backcolor"] = obj.actor.GetBackfaceProperty().GetColor()
        if adict["point_normals"] is None and poly.GetNumberOfPolys():
            if adict["representation"] != 1 and adict["shading"] != 0:
                # Compute normals from the already transformed world-space
                # points so exported shading follows the exact geometry that
                # the browser receives.
                world_poly = utils.buildPolyData(adict["points"], faces=adict["cells"])
                pdnorm = vtki.new("PolyDataNormals")
                pdnorm.SetInputData(world_poly)
                pdnorm.SetComputePointNormals(True)
                pdnorm.SetComputeCellNormals(False)
                pdnorm.SetConsistency(True)
                pdnorm.FlipNormalsOff()
                pdnorm.SetSplitting(False)
                pdnorm.Update()
                wn = pdnorm.GetOutput().GetPointData().GetNormals()
                if wn:
                    adict["point_normals"] = vedo.vtk2numpy(wn).astype(
                        np.float32, copy=False
                    )

    ######################################################## Volume
    elif isinstance(obj, Volume):
        adict["type"] = "Volume"
        _fillcommon(obj, adict)
        adict["array"] = obj.tonumpy()
        adict["mode"] = obj.mode()
        adict["spacing"] = obj.spacing()
        adict["origin"] = obj.origin()

        prp = obj.properties
        ctf = prp.GetRGBTransferFunction()
        otf = prp.GetScalarOpacity()
        gotf = prp.GetGradientOpacity()
        smin, smax = ctf.GetRange()
        xs = np.linspace(smin, smax, num=256, endpoint=True)
        cols, als, algrs = [], [], []
        for x in xs:
            cols.append(ctf.GetColor(x))
            als.append(otf.GetValue(x))
            if gotf:
                algrs.append(gotf.GetValue(x))
        adict["color"] = cols
        adict["alpha"] = als
        adict["alphagrad"] = algrs

    ######################################################## Image
    elif isinstance(obj, Image):
        adict["type"] = "Image"
        _fillcommon(obj, adict)
        adict["array"] = obj.tonumpy()
        adict["scale"] = obj.actor.GetScale()
        adict["position"] = obj.actor.GetPosition()
        adict["orientation"] = obj.actor.GetOrientation()
        adict["origin"] = obj.actor.GetOrigin()
        adict["alpha"] = obj.alpha()

    ######################################################## Text2D
    elif isinstance(obj, vedo.Text2D):
        adict["type"] = "Text2D"
        adict["rendered_at"] = obj.rendered_at
        adict["text"] = obj.text()
        adict["position"] = obj.actor.GetPosition()
        adict["color"] = obj.properties.GetColor()
        adict["font"] = obj.fontname
        adict["size"] = obj.properties.GetFontSize() / 22.5
        adict["bgcol"] = obj.properties.GetBackgroundColor()
        adict["alpha"] = obj.properties.GetBackgroundOpacity()
        adict["frame"] = obj.properties.GetFrame()

    ######################################################## Assembly
    elif isinstance(obj, Assembly):
        adict["type"] = "Assembly"
        _fillcommon(obj, adict)
        adict["parts"] = []
        for a in obj.unpack():
            adict["parts"].append(to_numpy(a))

    else:
        # vedo.logger.warning(f"to_numpy: cannot export object of type {type(obj)}")
        pass

    return adict


#########################################################################
def _color_to_hex(rgb) -> str | None:
    """Convert a color tuple to a CSS hex string."""
    if rgb is None:
        return None
    r, g, b = colors.get_color(rgb)
    return "#{:02x}{:02x}{:02x}".format(
        int(np.clip(r, 0, 1) * 255),
        int(np.clip(g, 0, 1) * 255),
        int(np.clip(b, 0, 1) * 255),
    )


def _json_compatible(value):
    """Convert numpy-heavy scene payloads into JSON-serializable objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {k: _json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(v) for v in value]
    return value


def _plotter_to_scene_dict(plt) -> dict:
    """Serialize the current Plotter scene into a dictionary."""
    if plt is None:
        vedo.logger.error("_plotter_to_scene_dict(): no active Plotter found")
        return {}

    def _append_scene_object(ob, index_tag) -> None:
        if isinstance(ob, Assembly):
            asse_actor = ob.actor if hasattr(ob, "actor") else ob
            asse_scale = asse_actor.GetScale()
            asse_pos = asse_actor.GetPosition()
            asse_ori = asse_actor.GetOrientation()
            asse_org = asse_actor.GetOrigin()
            for elem in ob.unpack():
                npobj = to_numpy(elem)
                npobj["name"] = f"ASSEMBLY{index_tag}_{ob.name}_{npobj.get('name', '')}"
                metadata = npobj.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                    npobj["metadata"] = metadata
                metadata["assembly"] = ob.name
                metadata["assembly_scale"] = asse_scale
                metadata["assembly_position"] = asse_pos
                metadata["assembly_orientation"] = asse_ori
                metadata["assembly_origin"] = asse_org
                sdict["objects"].append(npobj)
        else:
            sdict["objects"].append(to_numpy(ob))

    sdict = {}
    sdict["shape"] = plt.shape
    sdict["sharecam"] = plt.sharecam
    sdict["camera"] = dict(
        pos=plt.camera.GetPosition(),
        focal_point=plt.camera.GetFocalPoint(),
        viewup=plt.camera.GetViewUp(),
        distance=plt.camera.GetDistance(),
        clipping_range=plt.camera.GetClippingRange(),
        parallel_scale=plt.camera.GetParallelScale(),
    )
    sdict["position"] = plt.pos
    sdict["size"] = plt.size
    sdict["axes"] = plt.axes
    sdict["title"] = plt.title
    sdict["backgrcol"] = colors.get_color(plt.renderer.GetBackground())
    sdict["backgrcol2"] = None
    if plt.renderer.GetGradientBackground():
        sdict["backgrcol2"] = plt.renderer.GetBackground2()
    sdict["use_depth_peeling"] = plt.renderer.GetUseDepthPeeling()
    sdict["use_parallel_projection"] = plt.camera.GetParallelProjection()
    sdict["default_font"] = vedo.settings.default_font

    sdict["objects"] = []

    actors = plt.get_actors(include_non_pickables=True)
    for i, a in enumerate(actors):
        if not a.GetVisibility():
            continue

        try:
            ob = a.retrieve_object()
            _append_scene_object(ob, i)
        except AttributeError:
            pass

    for i, ax in enumerate(getattr(plt, "axes_instances", [])):
        if not ax or ax is True:
            continue
        if isinstance(ax, Assembly):
            _append_scene_object(ax, f"AX{i}")
    return sdict


#########################################################################
def _export_npy(plt, fileoutput="scene.npz") -> None:
    """Compatibility wrapper for the dedicated NPY exporter."""
    from .export_npy import _export_npy as _export_npy_impl

    _export_npy_impl(plt, fileoutput)


def _export_x3d(plt, fileoutput="scene.x3d", binary=False) -> None:
    """Compatibility wrapper for the dedicated X3D exporter."""
    from .export_x3d import _export_x3d as _export_x3d_impl

    _export_x3d_impl(plt, fileoutput, binary=binary)


def _export_threejs(
    plt, fileoutput="scene.html", backend_options: dict | None = None
) -> None:
    """Compatibility wrapper for the dedicated Three.js exporter."""
    from .export_threejs import _export_threejs as _export_threejs_impl

    _export_threejs_impl(plt, fileoutput, backend_options=backend_options)


########################################################################
def import_window(fileinput: str | os.PathLike) -> vedo.Plotter | None:
    """
    Import a whole scene from a Numpy NPZ file.

    Returns:
        `vedo.Plotter` instance
    """
    fileinput = str(fileinput)

    if fileinput.endswith(".npy") or fileinput.endswith(".npz"):
        return _import_npy(fileinput)

    # elif ".obj" in fileinput.lower():
    #     meshes = load_obj(fileinput, mtl_file, texture_path)
    #     plt = vedo.Plotter()
    #     plt.add(meshes)
    #     return plt

    # elif fileinput.endswith(".h5") or fileinput.endswith(".hdf5"):
    #     return _import_hdf5(fileinput) # in store/file_io_HDF5.py

    return None


def screenshot(
    filename="screenshot.png", scale=1, asarray=False
) -> vedo.Plotter | np.ndarray | None:
    """
    Save a screenshot of the current rendering window.

    Alternatively, press key `Shift-S` in the rendering window to save a screenshot.
    You can also use keyword `screenshot` in `show(..., screenshot="pic.png")`.

    Args:
        scale (int):
            Set image magnification as an integer multiplicative factor.
            E.g. setting a magnification of 2 produces an image twice as large,
            but 10x slower to generate.
        asarray (bool):
            Return a numpy array of the image
    """
    filename = str(filename)
    filename_lower = filename.lower()
    # print("calling screenshot", filename, scale, asarray)

    plt = vedo.current_plotter()
    if not plt or not plt.window:
        # vedo.logger.error("in screenshot(), rendering window is not present, skip.")
        return plt  ##########

    if plt.renderer:
        plt.renderer.ResetCameraClippingRange()

    if asarray and scale == 1 and not plt.offscreen:
        nx, ny = plt.window.GetSize()
        arr = vtki.vtkUnsignedCharArray()
        plt.window.GetRGBACharPixelData(0, 0, nx - 1, ny - 1, 0, arr)
        narr = vedo.vtk2numpy(arr).T[:3].T.reshape([ny, nx, 3])
        narr = np.flip(narr, axis=0)
        return narr  ##########

    ###########################
    if filename_lower.endswith(".pdf"):
        writer = vtki.new("GL2PSExporter")
        writer.SetRenderWindow(plt.window)
        writer.Write3DPropsAsRasterImageOff()
        writer.SilentOn()
        writer.SetSortToBSP()
        writer.SetFileFormatToPDF()
        writer.SetFilePrefix(filename.replace(".pdf", ""))
        writer.Write()
        return plt  ##########

    elif filename_lower.endswith(".svg"):
        writer = vtki.new("GL2PSExporter")
        writer.SetRenderWindow(plt.window)
        writer.Write3DPropsAsRasterImageOff()
        writer.SilentOn()
        writer.SetSortToBSP()
        writer.SetFileFormatToSVG()
        writer.SetFilePrefix(filename.replace(".svg", ""))
        writer.Write()
        return plt  ##########

    elif filename_lower.endswith(".eps"):
        writer = vtki.new("GL2PSExporter")
        writer.SetRenderWindow(plt.window)
        writer.Write3DPropsAsRasterImageOff()
        writer.SilentOn()
        writer.SetSortToBSP()
        writer.SetFileFormatToEPS()
        writer.SetFilePrefix(filename.replace(".eps", ""))
        writer.Write()
        return plt  ##########

    if settings.screeshot_large_image:
        w2if = vtki.new("RenderLargeImage")
        w2if.SetInput(plt.renderer)
        w2if.SetMagnification(scale)
    else:
        w2if = vtki.new("WindowToImageFilter")
        w2if.SetInput(plt.window)
        if hasattr(w2if, "SetScale"):
            w2if.SetScale(int(scale), int(scale))
        if settings.screenshot_transparent_background:
            w2if.SetInputBufferTypeToRGBA()
        w2if.ReadFrontBufferOff()  # read from the back buffer
    w2if.Update()

    if asarray:
        pd = w2if.GetOutput().GetPointData()
        npdata = utils.vtk2numpy(pd.GetArray("ImageScalars"))
        # npdata = npdata[:, [0, 1, 2]]  # remove alpha channel, issue #1199
        ydim, xdim, _ = w2if.GetOutput().GetDimensions()
        npdata = npdata.reshape([xdim, ydim, -1])
        npdata = np.flip(npdata, axis=0)
        return npdata  ###########################

    if filename.lower().endswith(".png"):
        writer = vtki.new("PNGWriter")
        writer.SetFileName(filename)
        writer.SetInputData(w2if.GetOutput())
        writer.Write()
    elif filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        writer = vtki.new("JPEGWriter")
        writer.SetFileName(filename)
        writer.SetInputData(w2if.GetOutput())
        writer.Write()
    else:  # add .png
        writer = vtki.new("PNGWriter")
        writer.SetFileName(filename + ".png")
        writer.SetInputData(w2if.GetOutput())
        writer.Write()
    return plt
