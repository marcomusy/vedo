"""Interaction operations delegated from Plotter."""

import time
from typing import Any, Callable, Union
from typing import MutableSequence

import numpy as np

import vedo
import vedo.vtkclasses as vtki
from vedo import addons, utils
from vedo.plotter.events import Event


__docformat__ = "google"


def fill_event(plotter, ename="", pos=(), enable_picking=True) -> "Event":
    """
    Create an Event object with information of what was clicked.

    If `enable_picking` is False, no picking will be performed.
    This can be useful to avoid double picking when using buttons.
    """
    if not plotter.interactor:
        return Event()

    if len(pos) > 0:
        x, y = pos
        plotter.interactor.SetEventPosition(pos)
    else:
        x, y = plotter.interactor.GetEventPosition()
    plotter.renderer = plotter.interactor.FindPokedRenderer(x, y)

    plotter.picked2d = (x, y)

    key = plotter.interactor.GetKeySym()

    if key:
        if "_L" in key or "_R" in key:
            # skip things like Shift_R
            key = ""  # better than None
        else:
            if plotter.interactor.GetShiftKey():
                key = key.upper()

            if key == "MINUS":  # fix: vtk9 is ignoring shift chars..
                key = "underscore"
            elif key == "EQUAL":  # fix: vtk9 is ignoring shift chars..
                key = "plus"
            elif key == "SLASH":  # fix: vtk9 is ignoring shift chars..
                key = "?"

            if plotter.interactor.GetControlKey():
                key = "Ctrl+" + key

            if plotter.interactor.GetAltKey():
                key = "Alt+" + key

    if enable_picking:
        if not plotter.picker:
            plotter.picker = vtki.vtkPropPicker()

        plotter.picker.PickProp(x, y, plotter.renderer)
        actor = plotter.picker.GetProp3D()
        # Note that GetProp3D already picks Assembly

        xp, yp = plotter.interactor.GetLastEventPosition()
        dx, dy = x - xp, y - yp

        delta3d = np.array([0, 0, 0])

        if actor:
            picked3d = np.array(plotter.picker.GetPickPosition())

            try:
                vobj = actor.retrieve_object()
                old_pt = np.asarray(vobj.picked3d)
                vobj.picked3d = picked3d
                delta3d = picked3d - old_pt
            except (AttributeError, TypeError):
                pass

        else:
            picked3d = None

        if not actor:  # try 2D
            actor = plotter.picker.GetActor2D()

    event = Event()
    event.name = ename
    event.title = plotter.title
    event.id = -1  # will be set by the timer wrapper function
    event.timerid = -1  # will be set by the timer wrapper function
    event.priority = -1  # will be set by the timer wrapper function
    event.time = time.time()
    event.at = plotter.renderers.index(plotter.renderer)
    event.keypress = key
    if enable_picking:
        try:
            event.object = actor.retrieve_object()
        except AttributeError:
            event.object = actor
        try:
            event.actor = actor.retrieve_object()  # obsolete use object instead
        except AttributeError:
            event.actor = actor
        event.picked3d = picked3d
        event.picked2d = (x, y)
        event.delta2d = (dx, dy)
        event.angle2d = np.arctan2(dy, dx)
        event.speed2d = np.sqrt(dx * dx + dy * dy)
        event.delta3d = delta3d
        event.speed3d = np.sqrt(np.dot(delta3d, delta3d))
        event.isPoints = isinstance(event.object, vedo.Points)
        event.isMesh = isinstance(event.object, vedo.Mesh)
        event.isAssembly = isinstance(event.object, vedo.Assembly)
        event.isVolume = isinstance(event.object, vedo.Volume)
        event.isImage = isinstance(event.object, vedo.Image)
        event.isActor2D = isinstance(event.object, vtki.vtkActor2D)
    return event

def add_callback(plotter, event_name: str, func: Callable, priority=0.0, enable_picking=True) -> int:
    """
    Add a function to be executed while show() is active.

    Return a unique id for the callback.

    The callback function (see example below) exposes a dictionary
    with the following information:
    - `name`: event name,
    - `id`: event unique identifier,
    - `priority`: event priority (float),
    - `interactor`: the interactor object,
    - `at`: renderer nr. where the event occurred
    - `keypress`: key pressed as string
    - `actor`: object picked by the mouse
    - `picked3d`: point picked in world coordinates
    - `picked2d`: screen coords of the mouse pointer
    - `delta2d`: shift wrt previous position (to calculate speed, direction)
    - `delta3d`: ...same but in 3D world coords
    - `angle2d`: angle of mouse movement on screen
    - `speed2d`: speed of mouse movement on screen
    - `speed3d`: speed of picked point in world coordinates
    - `isPoints`: True if of class
    - `isMesh`: True if of class
    - `isAssembly`: True if of class
    - `isVolume`: True if of class Volume
    - `isImage`: True if of class

    If `enable_picking` is False, no picking will be performed.
    This can be useful to avoid double picking when using buttons.

    Frequently used events are:
    - `KeyPress`, `KeyRelease`: listen to keyboard events
    - `LeftButtonPress`, `LeftButtonRelease`: listen to mouse clicks
    - `MiddleButtonPress`, `MiddleButtonRelease`
    - `RightButtonPress`, `RightButtonRelease`
    - `MouseMove`: listen to mouse pointer changing position
    - `MouseWheelForward`, `MouseWheelBackward`
    - `Enter`, `Leave`: listen to mouse entering or leaving the window
    - `Pick`, `StartPick`, `EndPick`: listen to object picking
    - `ResetCamera`, `ResetCameraClippingRange`
    - `Error`, `Warning`
    - `Char`
    - `Timer`

    Check the complete list of events [here](https://vtk.org/doc/nightly/html/classvtkCommand.html).

    Example:
        ```python
        from vedo import *

        def func(evt):
            # this function is called every time the mouse moves
            # (evt is a dotted dictionary)
            if not evt.object:
                return  # no hit, return
            print("point coords =", evt.picked3d)
            # print(evt) # full event dump

        elli = Ellipsoid()
        plt = Plotter(axes=1)
        plt.add_callback('mouse hovering', func)
        plt.show(elli).close()
        ```

    Examples:
        - [spline_draw1.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/spline_draw1.py)
        - [colorlines.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/colorlines.py)

            ![](https://vedo.embl.es/images/advanced/spline_draw.png)

        - ..and many others!
    """
    from vtkmodules.util.misc import calldata_type # noqa

    if not plotter.interactor:
        return 0

    if vedo.settings.dry_run_mode >= 1:
        return 0

    #########################################
    @calldata_type(vtki.VTK_INT)
    def _func_wrap(iren, ename, timerid=None):
        event = plotter.fill_event(ename=ename, enable_picking=enable_picking)
        event.timerid = timerid
        event.id = cid
        event.priority = priority
        plotter.last_event = event
        func(event)

    #########################################

    event_name = utils.get_vtk_name_event(event_name)

    cid = plotter.interactor.AddObserver(event_name, _func_wrap, priority)
    # print(f"Registering event: {event_name} with id={cid}")
    return cid

def remove_callback(plotter, cid: Union[int, str]) -> Any:
    """
    Remove a callback function by its id
    or a whole category of callbacks by their name.

    Arguments:
        cid : (int, str)
            Unique id of the callback.
            If an event name is passed all callbacks of that type are removed.
    """
    if plotter.interactor:
        if isinstance(cid, str):
            cid = utils.get_vtk_name_event(cid)
            plotter.interactor.RemoveObservers(cid)
        else:
            plotter.interactor.RemoveObserver(cid)
    return plotter

def remove_all_observers(plotter) -> Any:
    """
    Remove all observers.

    Example:
    ```python
    from vedo import *

    def kfunc(event):
        print("Key pressed:", event.keypress)
        if event.keypress == 'q':
            plt.close()

    def rfunc(event):
        if event.isImage:
            printc("Right-clicked!", event)
            plt.render()

    img = Image(dataurl+"images/embryo.jpg")

    plt = Plotter(size=(1050, 600))
    plt.parallel_projection(True)
    plt.remove_all_observers()
    plt.add_callback("key press", kfunc)
    plt.add_callback("mouse right click", rfunc)
    plt.show("Right-Click Me! Press q to exit.", img)
    plt.close()
    ```
    """
    if plotter.interactor:
        plotter.interactor.RemoveAllObservers()
    return plotter

def timer_callback(plotter, action: str, timer_id=None, dt=1, one_shot=False) -> int:
    """
    Start or stop an existing timer.

    Arguments:
        action : (str)
            Either "create"/"start" or "destroy"/"stop"
        timer_id : (int)
            When stopping the timer, the ID of the timer as returned when created
        dt : (int)
            time in milliseconds between each repeated call
        one_shot : (bool)
            create a one shot timer of prescribed duration instead of a repeating one

    Examples:
        - [timer_callback1.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/timer_callback1.py)
        - [timer_callback2.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/timer_callback2.py)

        ![](https://vedo.embl.es/images/advanced/timer_callback1.jpg)
    """
    if action in ("create", "start"):

        if "Windows" in vedo.sys_platform:
            # otherwise on windows it gets stuck
            plotter.initialize_interactor()

        if timer_id is not None:
            vedo.logger.warning("you set a timer_id but it will be ignored.")
        if one_shot:
            timer_id = plotter.interactor.CreateOneShotTimer(dt)
        else:
            timer_id = plotter.interactor.CreateRepeatingTimer(dt)
        return timer_id

    elif action in ("destroy", "stop"):
        if timer_id is not None:
            plotter.interactor.DestroyTimer(timer_id)
        else:
            vedo.logger.warning("please set a timer_id. Cannot stop timer.")
    else:
        e = f"in timer_callback(). Cannot understand action: {action}\n"
        e += " allowed actions are: ['start', 'stop']. Skipped."
        vedo.logger.error(e)
    return timer_id

def add_observer(plotter, event_name: str, func: Callable, priority=0.0) -> int:
    """
    Add a callback function that will be called when an event occurs.
    Consider using `add_callback()` instead.
    """
    if not plotter.interactor:
        return -1
    event_name = utils.get_vtk_name_event(event_name)
    idd = plotter.interactor.AddObserver(event_name, func, priority)
    return idd

def compute_world_coordinate(
    plotter,
    pos2d: MutableSequence[float],
    at=None,
    objs=(),
    bounds=(),
    offset=None,
    pixeltol=None,
    worldtol=None,
) -> np.ndarray:
    """
    Transform a 2D point on the screen into a 3D point inside the rendering scene.
    If a set of meshes is passed then points are placed onto these.

    Arguments:
        pos2d : (list)
            2D screen coordinates point.
        at : (int)
            renderer number.
        objs : (list)
            list of Mesh objects to project the point onto.
        bounds : (list)
            specify a bounding box as [xmin,xmax, ymin,ymax, zmin,zmax].
        offset : (float)
            specify an offset value.
        pixeltol : (int)
            screen tolerance in pixels.
        worldtol : (float)
            world coordinates tolerance.

    Returns:
        numpy array, the point in 3D world coordinates.

    Examples:
        - [cut_freehand.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/cut_freehand.py)
        - [mousehover3.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mousehover3.py)

        ![](https://vedo.embl.es/images/basic/mousehover3.jpg)
    """
    renderer = plotter.renderer if at is None else plotter.renderers[at]

    if not objs:
        pp = vtki.vtkFocalPlanePointPlacer()
    else:
        pps = vtki.vtkPolygonalSurfacePointPlacer()
        for ob in objs:
            pps.AddProp(ob.actor)
        pp = pps  # type: ignore

    if len(bounds) == 6:
        pp.SetPointBounds(bounds)
    if pixeltol:
        pp.SetPixelTolerance(pixeltol)
    if worldtol:
        pp.SetWorldTolerance(worldtol)
    if offset:
        pp.SetOffset(offset)

    worldPos: MutableSequence[float] = [0, 0, 0]
    worldOrient: MutableSequence[float] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    pp.ComputeWorldPosition(renderer, pos2d, worldPos, worldOrient)
    # validw = pp.ValidateWorldPosition(worldPos, worldOrient)
    # validd = pp.ValidateDisplayPosition(renderer, pos2d)
    return np.array(worldPos)

def compute_screen_coordinates(plotter, obj, full_window=False) -> np.ndarray:
    """
    Given a 3D points in the current renderer (or full window),
    find the screen pixel coordinates.

    Example:
        ```python
        from vedo import *

        elli = Ellipsoid().point_size(5)

        plt = Plotter()
        plt.show(elli, "Press q to continue and print the info")

        xyscreen = plt.compute_screen_coordinates(elli)
        print('xyscreen coords:', xyscreen)

        # simulate an event happening at one point
        event = plt.fill_event(pos=xyscreen[123])
        print(event)
        ```
    """
    try:
        obj = obj.coordinates
    except AttributeError:
        pass

    if utils.is_sequence(obj):
        pts = obj
    p2d = []
    cs = vtki.vtkCoordinate()
    cs.SetCoordinateSystemToWorld()
    cs.SetViewport(plotter.renderer)
    for p in pts:
        cs.SetValue(p)
        if full_window:
            p2d.append(cs.GetComputedDisplayValue(plotter.renderer))
        else:
            p2d.append(cs.GetComputedViewportValue(plotter.renderer))
    return np.array(p2d, dtype=int)

def pick_area(plotter, pos1, pos2, at=None) -> "vedo.Mesh":
    """
    Pick all objects within a box defined by two corner points in 2D screen coordinates.

    Returns a frustum Mesh that contains the visible field of view.
    This can be used to select objects in a scene or select vertices.

    Example:
        ```python
        from vedo import *

        settings.enable_default_mouse_callbacks = False

        def mode_select(objs):
            print("Selected objects:", objs)
            d0 = mode.start_x, mode.start_y # display coords
            d1 = mode.end_x, mode.end_y

            frustum = plt.pick_area(d0, d1)
            col = np.random.randint(0, 10)
            infru = frustum.inside_points(mesh)
            infru.point_size(10).color(col)
            plt.add(frustum, infru).render()

        mesh = Mesh(dataurl+"cow.vtk")
        mesh.color("k5").linewidth(1)

        from vedo.plotter.modes import BlenderStyle
        mode = BlenderStyle()
        mode.callback_select = mode_select

        plt = Plotter().user_mode(mode)
        plt.show(mesh, axes=1)
        ```
    """
    ren = plotter.renderer if at is None else plotter.renderers[at]
    area_picker = vtki.vtkAreaPicker()
    area_picker.AreaPick(pos1[0], pos1[1], pos2[0], pos2[1], ren)
    planes = area_picker.GetFrustum()

    fru = vtki.new("FrustumSource")
    fru.SetPlanes(planes)
    fru.ShowLinesOff()
    fru.Update()

    afru = vedo.Mesh(fru.GetOutput())
    afru.alpha(0.1).lw(1).pickable(False)
    afru.name = "Frustum"
    return afru

def default_mouseleftclick(plotter, iren, event) -> None:
    x, y = iren.GetEventPosition()
    renderer = iren.FindPokedRenderer(x, y)
    picker = vtki.vtkPropPicker()
    picker.PickProp(x, y, renderer)

    plotter.renderer = renderer

    clicked_actor = picker.GetActor()
    # clicked_actor2D = picker.GetActor2D()

    # print('_default_mouseleftclick mouse at', x, y)
    # print("picked Volume:",   [picker.GetVolume()])
    # print("picked Actor2D:",  [picker.GetActor2D()])
    # print("picked Assembly:", [picker.GetAssembly()])
    # print("picked Prop3D:",   [picker.GetProp3D()])

    if not clicked_actor:
        clicked_actor = picker.GetAssembly()

    if not clicked_actor:
        clicked_actor = picker.GetProp3D()

    if not hasattr(clicked_actor, "GetPickable") or not clicked_actor.GetPickable():
        return

    plotter.picked3d = picker.GetPickPosition()
    plotter.picked2d = np.array([x, y])

    if not clicked_actor:
        return

    plotter.justremoved = None
    plotter.clicked_actor = clicked_actor

    try:  # might not be a vedo obj
        plotter.clicked_object = clicked_actor.retrieve_object()
        # save this info in the object itself
        plotter.clicked_object.picked3d = plotter.picked3d
        plotter.clicked_object.picked2d = plotter.picked2d
    except AttributeError:
        pass
