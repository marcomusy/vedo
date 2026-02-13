"""Lifecycle/runtime operations delegated from Plotter."""

from typing import Any

import vedo
import vedo.vtkclasses as vtki


__docformat__ = "google"


def initialize_interactor(plotter) -> Any:
    """Initialize the interactor if not already initialized."""
    if plotter.offscreen:
        return plotter
    if plotter.interactor:
        if not plotter.interactor.GetInitialized():
            plotter.interactor.Initialize()
            plotter.interactor.RemoveObservers("CharEvent")
    return plotter

def process_events(plotter) -> Any:
    """Process all pending events."""
    plotter.initialize_interactor()
    if plotter.interactor:
        try:
            plotter.interactor.ProcessEvents()
        except AttributeError:
            pass
    return plotter

def render(plotter, resetcam=False) -> Any:
    """Render the scene. This method is typically used in loops or callback functions."""

    if vedo.settings.dry_run_mode >= 2:
        return plotter

    if not plotter.window:
        return plotter

    plotter.initialize_interactor()

    if resetcam:
        plotter.renderer.ResetCamera()

    plotter.window.Render()

    if plotter._cocoa_process_events and plotter.interactor and plotter.interactor.GetInitialized():
        if "Darwin" in vedo.sys_platform and not plotter.offscreen:
            plotter.interactor.ProcessEvents()
            plotter._cocoa_process_events = False
    return plotter

def interactive(plotter) -> Any:
    """
    Start window interaction.
    Analogous to `show(..., interactive=True)`.
    """
    if vedo.settings.dry_run_mode >= 1:
        return plotter
    plotter.initialize_interactor()
    if plotter.interactor:
        # print("plotter.interactor.Start()")
        plotter.interactor.Start()
        # print("plotter.interactor.Start() done")
        if plotter._must_close_now:
            # print("plotter.interactor.TerminateApp()")
            if plotter.interactor:
                plotter.interactor.GetRenderWindow().Finalize()
                plotter.interactor.TerminateApp()
            plotter.interactor = None
            plotter.window = None
            plotter.renderer = None
            plotter.renderers = []
            plotter.camera = None
    return plotter

def use_depth_peeling(plotter, at=None, value=True) -> Any:
    """
    Specify whether use depth peeling algorithm at this specific renderer
    Call this method before the first rendering.
    """
    ren = plotter.renderer if at is None else plotter.renderers[at]
    ren.SetUseDepthPeeling(value)
    return plotter

def clear(plotter, at=None, deep=False) -> Any:
    """Clear the scene from all meshes and volumes."""
    renderer = plotter.renderer if at is None else plotter.renderers[at]
    if not renderer:
        return plotter

    if deep:
        renderer.RemoveAllViewProps()
    else:
        for ob in set(
            plotter.get_meshes()
            + plotter.get_volumes()
            + plotter.objects
            + plotter.axes_instances
        ):
            if isinstance(ob, vedo.shapes.Text2D):
                continue
            plotter.remove(ob)
            try:
                if ob.scalarbar:
                    plotter.remove(ob.scalarbar)
            except AttributeError:
                pass
    return plotter

def break_interaction(plotter) -> Any:
    """Break window interaction and return to the python execution flow"""
    if plotter.interactor:
        plotter.check_actors_trasform()
        plotter.interactor.ExitCallback()
    return plotter

def freeze(plotter, value=True) -> Any:
    """Freeze the current renderer. Use this with `sharecam=False`."""
    if not plotter.interactor:
        return plotter
    if not plotter.renderer:
        return plotter
    plotter.renderer.SetInteractive(not value)
    return plotter

def user_mode(plotter, mode) -> Any:
    """
    Modify the user interaction mode.

    Examples:
        ```python
        from vedo import *
        mode = interactor_modes.MousePan()
        mesh = Mesh(dataurl+"cow.vtk")
        plt = Plotter().user_mode(mode)
        plt.show(mesh, axes=1)
       ```
    See also:
    [VTK interactor styles](https://vtk.org/doc/nightly/html/classvtkInteractorStyle.html)
    """
    if not plotter.interactor:
        return plotter

    curr_style = plotter.interactor.GetInteractorStyle().GetClassName()
    # print("Current style:", curr_style)
    if curr_style.endswith("Actor"):
        plotter.check_actors_trasform()

    if isinstance(mode, (str, int)):
        # Set the style of interaction
        # see https://vtk.org/doc/nightly/html/classvtkInteractorStyle.html
        if   mode in (0, "TrackballCamera"):
            plotter.interactor.SetInteractorStyle(vtki.new("InteractorStyleTrackballCamera"))
            plotter.interactor.RemoveObservers("CharEvent")
        elif mode in (1, "TrackballActor"):
            plotter.interactor.SetInteractorStyle(vtki.new("InteractorStyleTrackballActor"))
        elif mode in (2, "JoystickCamera"):
            plotter.interactor.SetInteractorStyle(vtki.new("InteractorStyleJoystickCamera"))
        elif mode in (3, "JoystickActor"):
            plotter.interactor.SetInteractorStyle(vtki.new("InteractorStyleJoystickActor"))
        elif mode in (4, "Flight"):
            plotter.interactor.SetInteractorStyle(vtki.new("InteractorStyleFlight"))
        elif mode in (5, "RubberBand2D"):
            plotter.interactor.SetInteractorStyle(vtki.new("InteractorStyleRubberBand2D"))
        elif mode in (6, "RubberBand3D"):
            plotter.interactor.SetInteractorStyle(vtki.new("InteractorStyleRubberBand3D"))
        elif mode in (7, "RubberBandZoom"):
            plotter.interactor.SetInteractorStyle(vtki.new("InteractorStyleRubberBandZoom"))
        elif mode in (8, "Terrain"):
            plotter.interactor.SetInteractorStyle(vtki.new("InteractorStyleTerrain"))
        elif mode in (9, "Unicam"):
            plotter.interactor.SetInteractorStyle(vtki.new("InteractorStyleUnicam"))
        elif mode in (10, "Image", "image", "2d"):
            astyle = vtki.new("InteractorStyleImage")
            astyle.SetInteractionModeToImage3D()
            plotter.interactor.SetInteractorStyle(astyle)
        else:
            vedo.logger.warning(f"Unknown interaction mode: {mode}")

    elif isinstance(mode, vtki.vtkInteractorStyleUser):
        # set a custom interactor style
        if hasattr(mode, "interactor"):
            mode.interactor = plotter.interactor
            mode.renderer = plotter.renderer  # type: ignore
        mode.SetInteractor(plotter.interactor)
        mode.SetDefaultRenderer(plotter.renderer)
        plotter.interactor.SetInteractorStyle(mode)

    return plotter

def close(plotter) -> Any:
    """Close the plotter."""
    # https://examples.vtk.org/site/Cxx/Visualization/CloseWindow/
    vedo.set_last_figure(None)
    plotter.last_event = None
    plotter.sliders = []
    plotter.buttons = []
    plotter.widgets = []
    plotter.hover_legends = []
    plotter.background_renderer = None
    plotter._extralight = None

    plotter.hint_widget = None
    plotter.cutter_widget = None
    if vedo.current_plotter() == plotter:
        vedo.set_current_plotter(None)

    if vedo.settings.dry_run_mode >= 2:
        return plotter

    if not hasattr(plotter, "window"):
        return plotter
    if not plotter.window:
        return plotter
    if not hasattr(plotter, "interactor"):
        return plotter
    if not plotter.interactor:
        return plotter

    ###################################################

    plotter._must_close_now = True

    if plotter.interactor:
        if plotter._interactive:
            plotter.break_interaction()
        plotter.interactor.GetRenderWindow().Finalize()
        try:
            if "Darwin" in vedo.sys_platform:
                plotter.interactor.ProcessEvents()
        except:
            pass
        plotter.interactor.TerminateApp()
        plotter.camera = None
        plotter.renderer = None
        plotter.renderers = []
        plotter.window = None
        plotter.interactor = None

    return plotter # must return plotter for consistency
