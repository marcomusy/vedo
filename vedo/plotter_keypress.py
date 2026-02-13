# -*- coding: utf-8 -*-
"""Default keyboard event handler for Plotter."""

import os
import sys

import numpy as np

import vedo
import vedo.vtkclasses as vtki
from vedo import addons, utils


__docformat__ = "google"


def _key_quit(plotter, _iren, _renderer) -> bool:
    plotter.break_interaction()
    return True


def _key_close(plotter, _iren, _renderer) -> bool:
    plotter.close()
    return True


def _key_abort(plotter, _iren, _renderer) -> bool:
    vedo.logger.info("Execution aborted. Exiting python kernel now.")
    plotter.break_interaction()
    sys.exit(0)


def _key_help(_plotter, _iren, _renderer) -> bool:
    msg = f" vedo {vedo.__version__}"
    msg += f" | vtk {vtki.vtkVersion().GetVTKVersion()}"
    msg += f" | numpy {np.__version__}"
    msg += f" | python {sys.version_info[0]}.{sys.version_info[1]}, press: "
    vedo.printc(msg.ljust(75), invert=True)
    msg = (
        "    i     print info about the last clicked object     \n"
        "    I     print color of the pixel under the mouse     \n"
        "    Y     show the pipeline for this object as a graph \n"
        "    <- -> use arrows to reduce/increase opacity        \n"
        "    x     toggle mesh visibility                       \n"
        "    w     toggle wireframe/surface style               \n"
        "    l     toggle surface edges visibility              \n"
        "    p/P   hide surface faces and show only points      \n"
        "    1-3   cycle surface color (2=light, 3=dark)        \n"
        "    4     cycle color map (press shift-4 to go back)   \n"
        "    5-6   cycle point-cell arrays (shift to go back)   \n"
        "    7-8   cycle background and gradient color          \n"
        "    09+-  cycle axes styles (on keypad, or press +/-)  \n"
        "    k     cycle available lighting styles              \n"
        "    K     toggle shading as flat or phong              \n"
        "    A     toggle anti-aliasing                         \n"
        "    D     toggle depth-peeling (for transparencies)    \n"
        "    U     toggle perspective/parallel projection       \n"
        "    o/O   toggle extra light to scene and rotate it    \n"
        "    a     toggle interaction to Actor Mode             \n"
        "    n     toggle surface normals                       \n"
        "    r     reset camera position                        \n"
        "    R     reset camera to the closest orthogonal view  \n"
        "    .     fly camera to the last clicked point         \n"
        "    C     print the current camera parameters state    \n"
        "    X     invoke a cutter widget tool                  \n"
        "    S     save a screenshot of the current scene       \n"
        "    E/F   export 3D scene to numpy file or X3D         \n"
        "    q     return control to python script              \n"
        "    Esc   abort execution and exit python kernel       "
    )
    vedo.printc(msg, dim=True, italic=True, bold=True)
    vedo.printc(
        " Check out the documentation at:  https://vedo.embl.es ".ljust(75),
        invert=True,
        bold=True,
    )
    return True


def _key_toggle_actor_mode(_plotter, iren, _renderer) -> bool:
    cur = iren.GetInteractorStyle()
    if isinstance(cur, vtki.get_class("InteractorStyleTrackballCamera")):
        msg = "Interactor style changed to TrackballActor\n"
        msg += "  you can now move and rotate individual meshes:\n"
        msg += "  press X twice to save the repositioned mesh\n"
        msg += "  press 'a' to go back to normal style"
        vedo.printc(msg)
        iren.SetInteractorStyle(vtki.new("InteractorStyleTrackballActor"))
    else:
        iren.SetInteractorStyle(vtki.new("InteractorStyleTrackballCamera"))
    return True


def _key_toggle_depth_peeling(plotter, _iren, renderer) -> bool:
    udp = not renderer.GetUseDepthPeeling()
    renderer.SetUseDepthPeeling(udp)
    if udp:
        plotter.window.SetAlphaBitPlanes(1)
        renderer.SetMaximumNumberOfPeels(vedo.settings.max_number_of_peels)
        renderer.SetOcclusionRatio(vedo.settings.occlusion_ratio)
    plotter.interactor.Render()
    wasUsed = renderer.GetLastRenderingUsedDepthPeeling()
    rnr = plotter.renderers.index(renderer)
    vedo.printc(f"Depth peeling set to {udp} for renderer nr.{rnr}", c=udp)
    if not wasUsed and udp:
        vedo.printc("\t...but last rendering did not actually used it!", c=udp, invert=True)
    return True


def _key_fly_to(plotter, _iren, _renderer) -> bool:
    if plotter.picked3d:
        plotter.fly_to(plotter.picked3d)
    return True


def _key_screenshot(plotter, _iren, _renderer) -> bool:
    fname = "screenshot.png"
    i = 1
    while os.path.isfile(fname):
        fname = f"screenshot{i}.png"
        i += 1
    for ss in plotter.sliders:
        ss[0].off()
    for bb in plotter.buttons:
        bb.off()
    vedo.file_io.screenshot(fname)
    vedo.printc(rf":camera: Saved rendering window to {fname}", c="b")
    for ss in plotter.sliders:
        ss[0].on()
        ss[0].Render()
    for bb in plotter.buttons:
        bb.on()
    return True


def _key_print_camera(_plotter, _iren, renderer) -> bool:
    cam = renderer.GetActiveCamera()
    vedo.printc("\n###################################################", c="y")
    vedo.printc("## Template python code to position this camera: ##", c="y")
    vedo.printc("cam = dict(", c="y")
    vedo.printc("    pos=" + utils.precision(cam.GetPosition(), 6) + ",", c="y")
    vedo.printc("    focal_point=" + utils.precision(cam.GetFocalPoint(), 6) + ",", c="y")
    vedo.printc("    viewup=" + utils.precision(cam.GetViewUp(), 6) + ",", c="y")
    vedo.printc("    roll=" + utils.precision(cam.GetRoll(), 6) + ",", c="y")
    if cam.GetParallelProjection():
        vedo.printc("    parallel_scale=" + utils.precision(cam.GetParallelScale(), 6) + ",", c="y")
    else:
        vedo.printc("    distance=" + utils.precision(cam.GetDistance(), 6) + ",", c="y")
    vedo.printc("    clipping_range=" + utils.precision(cam.GetClippingRange(), 6) + ",", c="y")
    vedo.printc(")", c="y")
    vedo.printc("show(mymeshes, camera=cam)", c="y")
    vedo.printc("###################################################", c="y")
    return True


def _key_export_npz(_plotter, _iren, _renderer) -> bool:
    vedo.printc(r":camera: Exporting 3D window to file scene.npz", c="b", end="")
    vedo.file_io.export_window("scene.npz")
    vedo.printc(", try:\n> vedo scene.npz  # (this is experimental!)", c="b")
    return True


def _key_export_x3d(_plotter, _iren, _renderer) -> bool:
    vedo.file_io.export_window("scene.x3d")
    vedo.printc(r":camera: Exporting 3D window to file", c="b", end="")
    vedo.file_io.export_window("scene.npz")
    vedo.printc(". Try:\n> firefox scene.html", c="b")
    return False


def _key_print_info(plotter, _iren, _renderer) -> bool:
    if plotter.clicked_object:
        print(plotter.clicked_object)
    else:
        print(plotter)
    return False


def _key_pick_color(plotter, iren, _renderer) -> bool:
    x, y = iren.GetEventPosition()
    plotter.color_picker([x, y], verbose=True)
    return False


def _key_show_pipeline(plotter, _iren, _renderer) -> bool:
    if plotter.clicked_object and plotter.clicked_object.pipeline:
        plotter.clicked_object.pipeline.show()
    return False


_KEY_DISPATCH = {
    "q": _key_quit,
    "Return": _key_quit,
    "Ctrl+q": _key_close,
    "Ctrl+w": _key_close,
    "Escape": _key_close,
    "F1": _key_abort,
    "h": _key_help,
    "a": _key_toggle_actor_mode,
    "D": _key_toggle_depth_peeling,
    "period": _key_fly_to,
    "S": _key_screenshot,
    "C": _key_print_camera,
    "E": _key_export_npz,
    "F": _key_export_x3d,
    "i": _key_print_info,
    "I": _key_pick_color,
    "Y": _key_show_pipeline,
}


def handle_default_keypress(plotter, iren, event) -> None:
    # NB: qt creates and passes a vtkGenericRenderWindowInteractor

    key = iren.GetKeySym()

    if "_L" in key or "_R" in key:
        return

    if iren.GetShiftKey():
        key = key.upper()

    if iren.GetControlKey():
        key = "Ctrl+" + key

    if iren.GetAltKey():
        key = "Alt+" + key

    #######################################################
    # utils.vedo.printc('Pressed key:', key, c='y', box='-')
    # print(key, iren.GetShiftKey(), iren.GetAltKey(), iren.GetControlKey(),
    #       iren.GetKeyCode(), iren.GetRepeatCount())
    #######################################################

    x, y = iren.GetEventPosition()
    renderer = iren.FindPokedRenderer(x, y)

    dispatch = _KEY_DISPATCH.get(key)
    if dispatch:
        if dispatch(plotter, iren, renderer):
            return

    elif key == "Down":
        if plotter.clicked_object and plotter.clicked_object in plotter.get_meshes():
            plotter.clicked_object.alpha(0.02)
            if hasattr(plotter.clicked_object, "properties_backface"):
                bfp = plotter.clicked_actor.GetBackfaceProperty()
                plotter.clicked_object.properties_backface = bfp  # save it
                plotter.clicked_actor.SetBackfaceProperty(None)
        else:
            for obj in plotter.get_meshes():
                if obj:
                    obj.alpha(0.02)
                    bfp = obj.actor.GetBackfaceProperty()
                    if bfp and hasattr(obj, "properties_backface"):
                        obj.properties_backface = bfp
                        obj.actor.SetBackfaceProperty(None)

    elif key == "Left":
        if plotter.clicked_object and plotter.clicked_object in plotter.get_meshes():
            ap = plotter.clicked_object.properties
            aal = max([ap.GetOpacity() * 0.75, 0.01])
            ap.SetOpacity(aal)
            bfp = plotter.clicked_actor.GetBackfaceProperty()
            if bfp and hasattr(plotter.clicked_object, "properties_backface"):
                plotter.clicked_object.properties_backface = bfp
                plotter.clicked_actor.SetBackfaceProperty(None)
        else:
            for a in plotter.get_meshes():
                if a:
                    ap = a.properties
                    aal = max([ap.GetOpacity() * 0.75, 0.01])
                    ap.SetOpacity(aal)
                    bfp = a.actor.GetBackfaceProperty()
                    if bfp and hasattr(a, "properties_backface"):
                        a.properties_backface = bfp
                        a.actor.SetBackfaceProperty(None)

    elif key == "Right":
        if plotter.clicked_object and plotter.clicked_object in plotter.get_meshes():
            ap = plotter.clicked_object.properties
            aal = min([ap.GetOpacity() * 1.25, 1.0])
            ap.SetOpacity(aal)
            if (
                aal == 1
                and hasattr(plotter.clicked_object, "properties_backface")
                and plotter.clicked_object.properties_backface
            ):
                # put back
                plotter.clicked_actor.SetBackfaceProperty(
                    plotter.clicked_object.properties_backface)
        else:
            for a in plotter.get_meshes():
                if a:
                    ap = a.properties
                    aal = min([ap.GetOpacity() * 1.25, 1.0])
                    ap.SetOpacity(aal)
                    if (
                        aal == 1
                        and hasattr(a, "properties_backface")
                        and a.properties_backface
                    ):
                        a.actor.SetBackfaceProperty(a.properties_backface)

    elif key == "Up":
        if plotter.clicked_object and plotter.clicked_object in plotter.get_meshes():
            plotter.clicked_object.properties.SetOpacity(1)
            if (
                hasattr(plotter.clicked_object, "properties_backface")
                and plotter.clicked_object.properties_backface
            ):
                plotter.clicked_object.actor.SetBackfaceProperty(
                    plotter.clicked_object.properties_backface
                )
        else:
            for a in plotter.get_meshes():
                if a:
                    a.properties.SetOpacity(1)
                    if hasattr(a, "properties_backface") and a.properties_backface:
                        a.actor.SetBackfaceProperty(a.properties_backface)

    elif key == "P":
        if plotter.clicked_object and plotter.clicked_object in plotter.get_meshes():
            objs = [plotter.clicked_object]
        else:
            objs = plotter.get_meshes()
        for ia in objs:
            try:
                ps = ia.properties.GetPointSize()
                if ps > 1:
                    ia.properties.SetPointSize(ps - 1)
                ia.properties.SetRepresentationToPoints()
            except AttributeError:
                pass

    elif key == "p":
        if plotter.clicked_object and plotter.clicked_object in plotter.get_meshes():
            objs = [plotter.clicked_object]
        else:
            objs = plotter.get_meshes()
        for ia in objs:
            try:
                ps = ia.properties.GetPointSize()
                ia.properties.SetPointSize(ps + 2)
                ia.properties.SetRepresentationToPoints()
            except AttributeError:
                pass

    elif key == "U":
        pval = renderer.GetActiveCamera().GetParallelProjection()
        renderer.GetActiveCamera().SetParallelProjection(not pval)
        if pval:
            renderer.ResetCamera()

    elif key == "r":
        renderer.ResetCamera()

    elif key == "A":  # toggle antialiasing
        msam = plotter.window.GetMultiSamples()
        if not msam:
            plotter.window.SetMultiSamples(16)
        else:
            plotter.window.SetMultiSamples(0)
        msam = plotter.window.GetMultiSamples()
        if msam:
            vedo.printc(f"Antialiasing set to {msam} samples", c=bool(msam))
        else:
            vedo.printc("Antialiasing disabled", c=bool(msam))

    elif key == "R":
        plotter.reset_viewup()

    elif key == "w":
        try:
            if plotter.clicked_object.properties.GetRepresentation() == 1:  # toggle
                plotter.clicked_object.properties.SetRepresentationToSurface()
            else:
                plotter.clicked_object.properties.SetRepresentationToWireframe()
        except AttributeError:
            pass

    elif key == "1":
        try:
            plotter._icol += 1
            plotter.clicked_object.mapper.ScalarVisibilityOff()
            pal = vedo.colors.palettes[vedo.settings.palette % len(vedo.colors.palettes)]
            plotter.clicked_object.c(pal[(plotter._icol) % 10])
            plotter.remove(plotter.clicked_object.scalarbar)
        except AttributeError:
            pass

    elif key == "2":  # dark colors
        try:
            bsc = ["k1", "k2", "k3", "k4",
                "b1", "b2", "b3", "b4",
                "p1", "p2", "p3", "p4",
                "g1", "g2", "g3", "g4",
                "r1", "r2", "r3", "r4",
                "o1", "o2", "o3", "o4",
                "y1", "y2", "y3", "y4"]
            plotter._icol += 1
            if plotter.clicked_object:
                plotter.clicked_object.mapper.ScalarVisibilityOff()
                newcol = vedo.get_color(bsc[(plotter._icol) % len(bsc)])
                plotter.clicked_object.c(newcol)
                plotter.remove(plotter.clicked_object.scalarbar)
        except AttributeError:
            pass

    elif key == "3":  # light colors
        try:
            bsc = ["k6", "k7", "k8", "k9",
                "b6", "b7", "b8", "b9",
                "p6", "p7", "p8", "p9",
                "g6", "g7", "g8", "g9",
                "r6", "r7", "r8", "r9",
                "o6", "o7", "o8", "o9",
                "y6", "y7", "y8", "y9"]
            plotter._icol += 1
            if plotter.clicked_object:
                plotter.clicked_object.mapper.ScalarVisibilityOff()
                newcol = vedo.get_color(bsc[(plotter._icol) % len(bsc)])
                plotter.clicked_object.c(newcol)
                plotter.remove(plotter.clicked_object.scalarbar)
        except AttributeError:
            pass

    elif key == "4":  # cmap name cycle
        ob = plotter.clicked_object
        if not isinstance(ob, (vedo.Points, vedo.UnstructuredGrid)):
            return
        if not ob.mapper.GetScalarVisibility():
            return
        onwhat = ob.mapper.GetScalarModeAsString()  # UsePointData/UseCellData

        cmap_names = [
            "Accent",
            "Paired",
            "rainbow",
            "rainbow_r",
            "Spectral",
            "Spectral_r",
            "gist_ncar",
            "gist_ncar_r",
            "viridis",
            "viridis_r",
            "hot",
            "hot_r",
            "terrain",
            "ocean",
            "coolwarm",
            "seismic",
            "PuOr",
            "RdYlGn",
        ]
        try:
            i = cmap_names.index(ob._cmap_name)
            if iren.GetShiftKey():
                i -= 1
            else:
                i += 1
            if i >= len(cmap_names):
                i = 0
            if i < 0:
                i = len(cmap_names) - 1
        except ValueError:
            i = 0

        ob._cmap_name = cmap_names[i]
        ob.cmap(ob._cmap_name, on=onwhat)
        if ob.scalarbar:
            if isinstance(ob.scalarbar, vtki.vtkActor2D):
                plotter.remove(ob.scalarbar)
                title = ob.scalarbar.GetTitle()
                ob.add_scalarbar(title=title)
                plotter.add(ob.scalarbar).render()
            elif isinstance(ob.scalarbar, vedo.Assembly):
                plotter.remove(ob.scalarbar)
                ob.add_scalarbar3d(title=ob._cmap_name)
                plotter.add(ob.scalarbar)

        vedo.printc(
            f"Name:'{ob.name}'," if ob.name else "",
            f"range:{utils.precision(ob.mapper.GetScalarRange(),3)},",
            f"colormap:'{ob._cmap_name}'", c="g", bold=False,
        )

    elif key == "5":  # cycle pointdata array
        ob = plotter.clicked_object
        if not isinstance(ob, (vedo.Points, vedo.UnstructuredGrid)):
            return

        arrnames = ob.pointdata.keys()
        arrnames = [a for a in arrnames if "normal" not in a.lower()]
        arrnames = [a for a in arrnames if "tcoord" not in a.lower()]
        arrnames = [a for a in arrnames if "textur" not in a.lower()]
        if len(arrnames) == 0:
            return
        ob.mapper.SetScalarVisibility(1)

        if not ob._cmap_name:
            ob._cmap_name = "rainbow"

        try:
            curr_name = ob.dataset.GetPointData().GetScalars().GetName()
            i = arrnames.index(curr_name)
            if "normals" in curr_name.lower():
                return
            if iren.GetShiftKey():
                i -= 1
            else:
                i += 1
            if i >= len(arrnames):
                i = 0
            if i < 0:
                i = len(arrnames) - 1
        except (ValueError, AttributeError):
            i = 0

        ob.cmap(ob._cmap_name, arrnames[i], on="points")
        if ob.scalarbar:
            if isinstance(ob.scalarbar, vtki.vtkActor2D):
                plotter.remove(ob.scalarbar)
                title = ob.scalarbar.GetTitle()
                ob.scalarbar = None
                ob.add_scalarbar(title=arrnames[i])
                plotter.add(ob.scalarbar)
            elif isinstance(ob.scalarbar, vedo.Assembly):
                plotter.remove(ob.scalarbar)
                ob.scalarbar = None
                ob.add_scalarbar3d(title=arrnames[i])
                plotter.add(ob.scalarbar)
        else:
            vedo.printc(
                f"Name:'{ob.name}'," if ob.name else "",
                f"active pointdata array: '{arrnames[i]}'",
                c="g",
                bold=False,
            )

    elif key == "6":  # cycle celldata array
        ob = plotter.clicked_object
        if not isinstance(ob, (vedo.Points, vedo.UnstructuredGrid)):
            return

        arrnames = ob.celldata.keys()
        arrnames = [a for a in arrnames if "normal" not in a.lower()]
        arrnames = [a for a in arrnames if "tcoord" not in a.lower()]
        arrnames = [a for a in arrnames if "textur" not in a.lower()]
        if len(arrnames) == 0:
            return
        ob.mapper.SetScalarVisibility(1)

        if not ob._cmap_name:
            ob._cmap_name = "rainbow"

        try:
            curr_name = ob.dataset.GetCellData().GetScalars().GetName()
            i = arrnames.index(curr_name)
            if "normals" in curr_name.lower():
                return
            if iren.GetShiftKey():
                i -= 1
            else:
                i += 1
            if i >= len(arrnames):
                i = 0
            if i < 0:
                i = len(arrnames) - 1
        except (ValueError, AttributeError):
            i = 0

        ob.cmap(ob._cmap_name, arrnames[i], on="cells")
        if ob.scalarbar:
            if isinstance(ob.scalarbar, vtki.vtkActor2D):
                plotter.remove(ob.scalarbar)
                title = ob.scalarbar.GetTitle()
                ob.scalarbar = None
                ob.add_scalarbar(title=arrnames[i])
                plotter.add(ob.scalarbar)
            elif isinstance(ob.scalarbar, vedo.Assembly):
                plotter.remove(ob.scalarbar)
                ob.scalarbar = None
                ob.add_scalarbar3d(title=arrnames[i])
                plotter.add(ob.scalarbar)
        else:
            vedo.printc(
                f"Name:'{ob.name}'," if ob.name else "",
                f"active celldata array: '{arrnames[i]}'",
                c="g", bold=False,
            )

    elif key == "7":
        bgc = np.array(renderer.GetBackground()).sum() / 3
        if bgc <= 0:
            bgc = 0.223
        elif 0 < bgc < 1:
            bgc = 1
        else:
            bgc = 0
        renderer.SetBackground(bgc, bgc, bgc)

    elif key == "8":
        bg2cols = [
            "lightyellow",
            "darkseagreen",
            "palegreen",
            "steelblue",
            "lightblue",
            "cadetblue",
            "lavender",
            "white",
            "blackboard",
            "black",
        ]
        bg2name = vedo.get_color_name(renderer.GetBackground2())
        if bg2name in bg2cols:
            idx = bg2cols.index(bg2name)
        else:
            idx = 4
        if idx is not None:
            bg2name_next = bg2cols[(idx + 1) % (len(bg2cols) - 1)]
        if not bg2name_next:
            renderer.GradientBackgroundOff()
        else:
            renderer.GradientBackgroundOn()
            renderer.SetBackground2(vedo.get_color(bg2name_next))

    elif key in ["plus", "equal", "KP_Add", "minus", "KP_Subtract"]:  # cycle axes style
        i = plotter.renderers.index(renderer)
        try:
            plotter.axes_instances[i].EnabledOff()
            plotter.axes_instances[i].SetInteractor(None)
        except AttributeError:
            # print("Cannot remove widget", [plotter.axes_instances[i]])
            try:
                plotter.remove(plotter.axes_instances[i])
            except:
                print("Cannot remove axes", [plotter.axes_instances[i]])
                return
        plotter.axes_instances[i] = None

        if not plotter.axes:
            plotter.axes = 0
        if isinstance(plotter.axes, dict):
            plotter.axes = 1

        if key in ["minus", "KP_Subtract"]:
            if not plotter.camera.GetParallelProjection() and plotter.axes == 0:
                plotter.axes -= 1  # jump ruler doesnt make sense in perspective mode
            bns = plotter.renderer.ComputeVisiblePropBounds()
            addons.add_global_axes(axtype=(plotter.axes - 1) % 15, c=None, bounds=bns)
        else:
            if not plotter.camera.GetParallelProjection() and plotter.axes == 12:
                plotter.axes += 1  # jump ruler doesnt make sense in perspective mode
            bns = plotter.renderer.ComputeVisiblePropBounds()
            addons.add_global_axes(axtype=(plotter.axes + 1) % 15, c=None, bounds=bns)
        plotter.render()

    elif "KP_" in key or key in [
            "Insert","End","Down","Next","Left","Begin","Right","Home","Up","Prior"
        ]:
        asso = {  # change axes style
            "KP_Insert": 0, "KP_0": 0, "Insert": 0,
            "KP_End":    1, "KP_1": 1, "End":    1,
            "KP_Down":   2, "KP_2": 2, "Down":   2,
            "KP_Next":   3, "KP_3": 3, "Next":   3,
            "KP_Left":   4, "KP_4": 4, "Left":   4,
            "KP_Begin":  5, "KP_5": 5, "Begin":  5,
            "KP_Right":  6, "KP_6": 6, "Right":  6,
            "KP_Home":   7, "KP_7": 7, "Home":   7,
            "KP_Up":     8, "KP_8": 8, "Up":     8,
            "Prior":     9,  # on windows OS
        }
        clickedr = plotter.renderers.index(renderer)
        if key in asso:
            if plotter.axes_instances[clickedr]:
                if hasattr(plotter.axes_instances[clickedr], "EnabledOff"):  # widget
                    plotter.axes_instances[clickedr].EnabledOff()
                else:
                    try:
                        renderer.RemoveActor(plotter.axes_instances[clickedr])
                    except:
                        pass
                plotter.axes_instances[clickedr] = None
            bounds = renderer.ComputeVisiblePropBounds()
            addons.add_global_axes(axtype=asso[key], c=None, bounds=bounds)
            plotter.interactor.Render()

    if key == "O":
        renderer.RemoveLight(plotter._extralight)
        plotter._extralight = None

    elif key == "o":
        vbb, sizes, _, _ = addons.compute_visible_bounds()
        cm = utils.vector((vbb[0] + vbb[1]) / 2, (vbb[2] + vbb[3]) / 2, (vbb[4] + vbb[5]) / 2)
        if not plotter._extralight:
            vup = renderer.GetActiveCamera().GetViewUp()
            pos = cm + utils.vector(vup) * utils.mag(sizes)
            plotter._extralight = addons.Light(pos, focal_point=cm, intensity=0.4)
            renderer.AddLight(plotter._extralight)
            vedo.printc("Press 'o' again to rotate light source, or 'O' to remove it.", c='y')
        else:
            cpos = utils.vector(plotter._extralight.GetPosition())
            x, y, z = plotter._extralight.GetPosition() - cm
            r, th, ph = vedo.transformations.cart2spher(x, y, z)
            th += 0.2
            if th > np.pi:
                th = np.random.random() * np.pi / 2
            ph += 0.3
            cpos = vedo.transformations.spher2cart(r, th, ph).T + cm
            plotter._extralight.SetPosition(cpos)

    elif key == "l":
        if plotter.clicked_object in plotter.get_meshes():
            objs = [plotter.clicked_object]
        else:
            objs = plotter.get_meshes()
        for ia in objs:
            try:
                ev = ia.properties.GetEdgeVisibility()
                ia.properties.SetEdgeVisibility(not ev)
                ia.properties.SetRepresentationToSurface()
                ia.properties.SetLineWidth(0.1)
            except AttributeError:
                pass

    elif key == "k":  # lightings
        if plotter.clicked_object in plotter.get_meshes():
            objs = [plotter.clicked_object]
        else:
            objs = plotter.get_meshes()
        shds = ("default", "metallic", "plastic", "shiny", "glossy", "off")
        for ia in objs:
            try:
                lnr = (ia._ligthingnr + 1) % 6
                ia.lighting(shds[lnr])
                ia._ligthingnr = lnr
            except AttributeError:
                pass

    elif key == "K":  # shading
        if plotter.clicked_object in plotter.get_meshes():
            objs = [plotter.clicked_object]
        else:
            objs = plotter.get_meshes()
        for ia in objs:
            if isinstance(ia, vedo.Mesh):
                ia.compute_normals(cells=False)
                intrp = ia.properties.GetInterpolation()
                if intrp > 0:
                    ia.properties.SetInterpolation(0)  # flat
                else:
                    ia.properties.SetInterpolation(2)  # phong

    elif key == "n":  # show normals to an actor
        plotter.remove("added_auto_normals")
        if plotter.clicked_object in plotter.get_meshes():
            if plotter.clicked_actor.GetPickable():
                norml = vedo.shapes.NormalLines(plotter.clicked_object)
                norml.name = "added_auto_normals"
                plotter.add(norml)

    elif key == "x":
        if plotter.justremoved is None:
            if plotter.clicked_object in plotter.get_meshes() or isinstance(
                plotter.clicked_object, vtki.vtkAssembly
            ):
                plotter.justremoved = plotter.clicked_actor
                plotter.renderer.RemoveActor(plotter.clicked_actor)
        else:
            plotter.renderer.AddActor(plotter.justremoved)
            plotter.justremoved = None

    elif key == "X":
        if plotter.clicked_object:
            if not plotter.cutter_widget:
                plotter.cutter_widget = addons.BoxCutter(plotter.clicked_object)
                plotter.add(plotter.cutter_widget)
                vedo.printc("Press i to toggle the cutter on/off", c='g', dim=1)
                vedo.printc("      u to flip selection", c='g', dim=1)
                vedo.printc("      r to reset cutting planes", c='g', dim=1)
                vedo.printc("      Shift+X to close the cutter box widget", c='g', dim=1)
                vedo.printc("      Ctrl+S to save the cut section to file.", c='g', dim=1)
            else:
                plotter.remove(plotter.cutter_widget)
                plotter.cutter_widget = None
            vedo.printc("Click object and press X to open the cutter box widget.", c='g')

    if iren:
        iren.Render()
