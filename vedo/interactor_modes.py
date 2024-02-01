#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass
import numpy as np

import vedo.vtkclasses as vtki

__docformat__ = "google"

__doc__ = """Submodule to customize interaction modes."""


class MousePan(vtki.vtkInteractorStyleUser):
    """
    Interaction mode to pan the scene by dragging the mouse.

    Controls:
    - Left mouse button will pan the scene.
    - Mouse middle button up/down is elevation, and left and right is azimuth.
    - Right mouse button is rotate (left/right movement) and zoom in/out
      (up/down movement)
    - Mouse scroll wheel is zoom in/out
    """

    def __init__(self):

        super().__init__()

        self.left = False
        self.middle = False
        self.right = False

        self.interactor = None
        self.renderer = None
        self.camera = None

        self.oldpickD = []
        self.newpickD = []
        self.oldpickW = np.array([0, 0, 0, 0], dtype=float)
        self.newpickW = np.array([0, 0, 0, 0], dtype=float)
        self.fpD = np.array([0, 0, 0], dtype=float)
        self.fpW = np.array([0, 0, 0], dtype=float)
        self.posW = np.array([0, 0, 0], dtype=float)
        self.motionD = np.array([0, 0], dtype=float)
        self.motionW = np.array([0, 0, 0], dtype=float)

        # print("MousePan: Left mouse button to pan the scene.")
        self.AddObserver("LeftButtonPressEvent", self._left_down)
        self.AddObserver("LeftButtonReleaseEvent", self._left_up)
        self.AddObserver("MiddleButtonPressEvent", self._middle_down)
        self.AddObserver("MiddleButtonReleaseEvent", self._middle_up)
        self.AddObserver("RightButtonPressEvent", self._right_down)
        self.AddObserver("RightButtonReleaseEvent", self._right_up)
        self.AddObserver("MouseWheelForwardEvent", self._wheel_forward)
        self.AddObserver("MouseWheelBackwardEvent", self._wheel_backward)
        self.AddObserver("MouseMoveEvent", self._mouse_move)

    def _get_motion(self):
        self.oldpickD = np.array(self.interactor.GetLastEventPosition())
        self.newpickD = np.array(self.interactor.GetEventPosition())
        self.motionD = (self.newpickD - self.oldpickD) / 4
        self.camera = self.renderer.GetActiveCamera()
        self.fpW = self.camera.GetFocalPoint()
        self.posW = self.camera.GetPosition()
        self.ComputeWorldToDisplay(self.renderer, self.fpW[0], self.fpW[1], self.fpW[2], self.fpD)
        focaldepth = self.fpD[2]
        self.ComputeDisplayToWorld(
            self.renderer, self.oldpickD[0], self.oldpickD[1], focaldepth, self.oldpickW
        )
        self.ComputeDisplayToWorld(
            self.renderer, self.newpickD[0], self.newpickD[1], focaldepth, self.newpickW
        )
        self.motionW[:3] = self.oldpickW[:3] - self.newpickW[:3]

    def _mouse_left_move(self):
        self._get_motion()
        self.camera.SetFocalPoint(self.fpW[:3] + self.motionW[:3])
        self.camera.SetPosition(self.posW[:3] + self.motionW[:3])
        self.interactor.Render()

    def _mouse_middle_move(self):
        self._get_motion()
        if abs(self.motionD[0]) > abs(self.motionD[1]):
            self.camera.Azimuth(-2 * self.motionD[0])
        else:
            self.camera.Elevation(-self.motionD[1])
        self.interactor.Render()

    def _mouse_right_move(self):
        self._get_motion()
        if abs(self.motionD[0]) > abs(self.motionD[1]):
            self.camera.Azimuth(-2.0 * self.motionD[0])
        else:
            self.camera.Zoom(1 + self.motionD[1] / 100)
        self.interactor.Render()

    def _mouse_wheel_forward(self):
        self.camera = self.renderer.GetActiveCamera()
        self.camera.Zoom(1.1)
        self.interactor.Render()

    def _mouse_wheel_backward(self):
        self.camera = self.renderer.GetActiveCamera()
        self.camera.Zoom(0.9)
        self.interactor.Render()

    def _left_down(self, w, e):
        self.left = True

    def _left_up(self, w, e):
        self.left = False

    def _middle_down(self, w, e):
        self.middle = True

    def _middle_up(self, w, e):
        self.middle = False

    def _right_down(self, w, e):
        self.right = True

    def _right_up(self, w, e):
        self.right = False

    def _wheel_forward(self, w, e):
        self._mouse_wheel_forward()

    def _wheel_backward(self, w, e):
        self._mouse_wheel_backward()

    def _mouse_move(self, w, e):
        if self.left:
            self._mouse_left_move()
        if self.middle:
            self._mouse_middle_move()
        if self.right:
            self._mouse_right_move()


###################################################################################
@dataclass
class _BlenderStyleDragInfo:
    """Data structure containing the data required to execute dragging a node"""

    # Scene related
    dragged_node = None  # Node

    # VTK related
    actors_dragging: list
    dragged_actors_original_positions: list  # original VTK positions
    start_position_3d = np.array((0, 0, 0))  # start position of the cursor

    delta = np.array((0, 0, 0))

    def __init__(self):
        self.actors_dragging = []
        self.dragged_actors_original_positions = []


###############################################
class BlenderStyle(vtki.vtkInteractorStyleUser):
    """
    Create an interaction style using the Blender default key-bindings.

    Camera action code is largely a translation of
    [this](https://github.com/Kitware/VTK/blob/master/Interaction/Style/vtkInteractorStyleTrackballCamera.cxx)
    Rubber band code
    [here](https://gitlab.kitware.com/updega2/vtk/-/blob/d324b2e898b0da080edee76159c2f92e6f71abe2/Rendering/vtkInteractorStyleRubberBandZoom.cxx)

    Interaction:

    Left button: Sections
    ----------------------
    Left button: select

    Left button drag: rubber band select or line select, depends on the dragged distance

    Middle button: Navigation
    --------------------------
    Middle button: rotate

    Middle button + shift : pan

    Middle button + ctrl  : zoom

    Middle button + alt : center view on picked point

    OR

    Middle button + alt   : zoom rubber band

    Mouse wheel : zoom

    Right button : context
    -----------------------
    Right key click: reserved for context-menu


    Keys
    ----

    2 or 3 : toggle perspective view

    a      : zoom all

    x,y,z  : view direction (toggles positive and negative)

    left/right arrows: rotate 45 deg clockwise/ccw about z-axis, snaps to nearest 45 deg
    b      : box zoom

    m      : mouse middle lock (toggles)

    space  : same as middle mouse button

    g      : grab (move actors)

    enter  : accept drag

    esc    : cancel drag, call callbackEscape


    LAPTOP MODE
    -----------
    Use space or `m` as replacement for middle button
    (`m` is sticky, space is not)

    callbacks / overriding keys:

    if `callback_any_key` is assigned then this function is called on every key press.
    If this function returns True then further processing of events is stopped.


    Moving actors
    --------------
    Actors can be moved interactively by the user.
    To support custom groups of actors to be moved as a whole the following system
    is implemented:

    When 'g' is pressed (grab) then a `_BlenderStyleDragInfo` dataclass object is assigned
    to style to `style.draginfo`.

    `_BlenderStyleDragInfo` includes a list of all the actors that are being dragged.
    By default this is the selection, but this may be altered.
    Drag is accepted using enter, click, or g. Drag is cancelled by esc

    Events
    ------
    `callback_start_drag` is called when initializing the drag.
    This is when to assign actors and other data to draginfo.

    `callback_end_drag` is called when the drag is accepted.

    Responding to other events
    --------------------------
    `callback_camera_direction_changed` : executed when camera has rotated but before re-rendering

    .. note::
        This class is based on R. de Bruin's
        [DAVE](https://github.com/RubendeBruin/DAVE/blob/master/src/DAVE/visual_helpers/vtkBlenderLikeInteractionStyle.py)
        implementation as discussed [in this issue](https://github.com/marcomusy/vedo/discussions/788).

    Example:
        [interaction_modes2.py](https://github.com/marcomusy/vedo/blob/master/examples/basic/interaction_modes2.py)
    """

    def __init__(self):

        super().__init__()

        self.interactor = None
        self.renderer = None

        # callback_select is called whenever one or mode props are selected.
        # callback will be called with a list of props of which the first entry
        # is prop closest to the camera.
        self.callback_select = None
        self.callback_start_drag = None
        self.callback_end_drag = None
        self.callback_escape_key = None
        self.callback_delete_key = None
        self.callback_focus_key = None
        self.callback_any_key = None
        self.callback_measure = None  # callback with argument float (meters)
        self.callback_camera_direction_changed = None

        # active drag
        # assigned to a _BlenderStyleDragInfo object when dragging is active
        self.draginfo: _BlenderStyleDragInfo or None = None

        # picking
        self.picked_props = []  # will be filled by latest pick

        # settings
        self.mouse_motion_factor = 20
        self.mouse_wheel_motion_factor = 0.1
        self.zoom_motion_factor = 0.25

        # internals
        self.start_x = 0  # start of a drag
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0

        self.middle_mouse_lock = False
        self.middle_mouse_lock_actor = None  # will be created when required

        # Special Modes
        self._is_box_zooming = False

        # holds an image of the renderer output at the start of a drawing event
        self._pixel_array = vtki.vtkUnsignedCharArray()

        self._upside_down = False

        self._left_button_down = False
        self._middle_button_down = False

        self.AddObserver("RightButtonPressEvent", self.right_button_press)
        self.AddObserver("RightButtonReleaseEvent", self.right_button_release)
        self.AddObserver("MiddleButtonPressEvent", self.middle_button_press)
        self.AddObserver("MiddleButtonReleaseEvent", self.middle_button_release)
        self.AddObserver("MouseWheelForwardEvent", self.mouse_wheel_forward)
        self.AddObserver("MouseWheelBackwardEvent", self.mouse_wheel_backward)
        self.AddObserver("LeftButtonPressEvent", self.left_button_press)
        self.AddObserver("LeftButtonReleaseEvent", self.left_button_release)
        self.AddObserver("MouseMoveEvent", self.mouse_move)
        self.AddObserver("WindowResizeEvent", self.window_resized)
        # ^does not seem to fire!
        self.AddObserver("KeyPressEvent", self.key_press)
        self.AddObserver("KeyReleaseEvent", self.key_release)

    def right_button_press(self, obj, event):
        pass

    def right_button_release(self, obj, event):
        pass

    def middle_button_press(self, obj, event):
        self._middle_button_down = True

    def middle_button_release(self, obj, event):
        self._middle_button_down = False

        # perform middle button focus event if ALT is down
        if self.GetInteractor().GetAltKey():
            # print("Middle button released while ALT is down")
            # try to pick an object at the current mouse position
            rwi = self.GetInteractor()
            self.start_x, self.start_y = rwi.GetEventPosition()
            props = self.perform_picking_on_selection()

            if props:
                self.focus_on(props[0])

    def mouse_wheel_backward(self, obj, event):
        self.move_mouse_wheel(-1)

    def mouse_wheel_forward(self, obj, event):
        self.move_mouse_wheel(1)

    def mouse_move(self, obj, event):
        interactor = self.GetInteractor()

        # Find the renderer that is active below the current mouse position
        x, y = interactor.GetEventPosition()
        self.FindPokedRenderer(x, y)
        # sets the current renderer
        # [this->SetCurrentRenderer(this->Interactor->FindPokedRenderer(x, y));]

        Shift = interactor.GetShiftKey()
        Ctrl = interactor.GetControlKey()
        Alt = interactor.GetAltKey()

        MiddleButton = self._middle_button_down or self.middle_mouse_lock

        # start with the special modes
        if self._is_box_zooming:
            self.draw_dragged_selection()
        elif MiddleButton and not Shift and not Ctrl and not Alt:
            self.rotate()
        elif MiddleButton and Shift and not Ctrl and not Alt:
            self.pan()
        elif MiddleButton and Ctrl and not Shift and not Alt:
            self.zoom()  # Dolly
        elif self.draginfo is not None:
            self.execute_drag()
        elif self._left_button_down and Ctrl and Shift:
            self.draw_measurement()
        elif self._left_button_down:
            self.draw_dragged_selection()

        self.InvokeEvent("InteractionEvent", None)

    def move_mouse_wheel(self, direction):
        rwi = self.GetInteractor()

        # Find the renderer that is active below the current mouse position
        x, y = rwi.GetEventPosition()
        self.FindPokedRenderer(x, y)
        # sets the current renderer
        # [this->SetCurrentRenderer(this->Interactor->FindPokedRenderer(x, y));]

        # The movement
        ren = self.GetCurrentRenderer()

        #   // Calculate the focal depth since we'll be using it a lot
        camera = ren.GetActiveCamera()
        viewFocus = camera.GetFocalPoint()

        temp_out = [0, 0, 0]
        self.ComputeWorldToDisplay(
            ren, viewFocus[0], viewFocus[1], viewFocus[2], temp_out
        )
        focalDepth = temp_out[2]

        newPickPoint = [0, 0, 0, 0]
        x, y = rwi.GetEventPosition()
        self.ComputeDisplayToWorld(ren, x, y, focalDepth, newPickPoint)

        # Has to recalc old mouse point since the viewport has moved,
        # so can't move it outside the loop
        oldPickPoint = [0, 0, 0, 0]
        # xp, yp = rwi.GetLastEventPosition()

        # find the center of the window
        size = rwi.GetRenderWindow().GetSize()
        xp = size[0] / 2
        yp = size[1] / 2

        self.ComputeDisplayToWorld(ren, xp, yp, focalDepth, oldPickPoint)
        # Camera motion is reversed
        move_factor = -1 * self.zoom_motion_factor * direction

        motionVector = (
            move_factor * (oldPickPoint[0] - newPickPoint[0]),
            move_factor * (oldPickPoint[1] - newPickPoint[1]),
            move_factor * (oldPickPoint[2] - newPickPoint[2]),
        )

        viewFocus = camera.GetFocalPoint()  # do we need to do this again? Already did this
        viewPoint = camera.GetPosition()

        camera.SetFocalPoint(
            motionVector[0] + viewFocus[0],
            motionVector[1] + viewFocus[1],
            motionVector[2] + viewFocus[2],
        )
        camera.SetPosition(
            motionVector[0] + viewPoint[0],
            motionVector[1] + viewPoint[1],
            motionVector[2] + viewPoint[2],
        )

        # the zooming
        factor = self.mouse_motion_factor * self.mouse_wheel_motion_factor
        self.zoom_by_step(direction * factor)

    def zoom_by_step(self, step):
        if self.GetCurrentRenderer():
            self.StartDolly()
            self.dolly(pow(1.1, step))
            self.EndDolly()

    def left_button_press(self, obj, event):
        if self._is_box_zooming:
            return
        if self.draginfo:
            return

        self._left_button_down = True

        interactor = self.GetInteractor()
        Shift = interactor.GetShiftKey()
        Ctrl = interactor.GetControlKey()

        if Shift and Ctrl:
            if not self.GetCurrentRenderer().GetActiveCamera().GetParallelProjection():
                self.toggle_parallel_projection()

        rwi = self.GetInteractor()
        self.start_x, self.start_y = rwi.GetEventPosition()
        self.end_x = self.start_x
        self.end_y = self.start_y

        self.initialize_screen_drawing()

    def left_button_release(self, obj, event):
        # print("LeftButtonRelease")
        if self._is_box_zooming:
            self._is_box_zooming = False
            self.zoom_box(self.start_x, self.start_y, self.end_x, self.end_y)
            return

        if self.draginfo:
            self.finish_drag()
            return

        self._left_button_down = False

        interactor = self.GetInteractor()

        Shift = interactor.GetShiftKey()
        Ctrl = interactor.GetControlKey()
        # Alt = interactor.GetAltKey()

        if Ctrl and Shift:
            pass  # we were drawing the measurement

        else:
            if self.callback_select:
                props = self.perform_picking_on_selection()
                if props:  # only call back if anything was selected
                    self.picked_props = tuple(props)
                    self.callback_select(props)

        # remove the selection rubber band / line
        self.GetInteractor().Render()

    def key_press(self, obj, event):

        key = obj.GetKeySym()
        KEY = key.upper()

        # logging.info(f"Key Press: {key}")
        if self.callback_any_key:
            if self.callback_any_key(key):
                return

        if KEY == "M":
            self.middle_mouse_lock = not self.middle_mouse_lock
            self._update_middle_mouse_button_lock_actor()
        elif KEY == "G":
            if self.draginfo is not None:
                self.finish_drag()
            else:
                if self.callback_start_drag:
                    self.callback_start_drag()
                else:
                    self.start_drag()
                    # internally calls end-drag if drag is already active
        elif KEY == "ESCAPE":
            if self.callback_escape_key:
                self.callback_escape_key()
            if self.draginfo is not None:
                self.cancel_drag()
        elif KEY == "DELETE":
            if self.callback_delete_key:
                self.callback_delete_key()
        elif KEY == "RETURN":
            if self.draginfo:
                self.finish_drag()
        elif KEY == "SPACE":
            self.middle_mouse_lock = True
            # self._update_middle_mouse_button_lock_actor()
            # self.GrabFocus("MouseMoveEvent", self)
            # # TODO: grab and release focus; possible from python?
        elif KEY == "B":
            self._is_box_zooming = True
            rwi = self.GetInteractor()
            self.start_x, self.start_y = rwi.GetEventPosition()
            self.end_x = self.start_x
            self.end_y = self.start_y
            self.initialize_screen_drawing()
        elif KEY in ('2', '3'):
            self.toggle_parallel_projection()
        elif KEY == "A":
            self.zoom_fit()
        elif KEY == "X":
            self.set_view_x()
        elif KEY == "Y":
            self.set_view_y()
        elif KEY == "Z":
            self.set_view_z()
        elif KEY == "LEFT":
            self.rotate_discrete_step(1)
        elif KEY == "RIGHT":
            self.rotate_discrete_step(-1)
        elif KEY == "UP":
            self.rotate_turtable_by(0, 10)
        elif KEY == "DOWN":
            self.rotate_turtable_by(0, -10)
        elif KEY == "PLUS":
            self.zoom_by_step(2)
        elif KEY == "MINUS":
            self.zoom_by_step(-2)
        elif KEY == "F":
            if self.callback_focus_key:
                self.callback_focus_key()

        self.InvokeEvent("InteractionEvent", None)

    def key_release(self, obj, event):
        key = obj.GetKeySym()
        KEY = key.upper()
        # print(f"Key release: {key}")
        if KEY == "SPACE":
            if self.middle_mouse_lock:
                self.middle_mouse_lock = False
                self._update_middle_mouse_button_lock_actor()

    def window_resized(self):
        # print("window resized")
        self.initialize_screen_drawing()

    def rotate_discrete_step(self, movement_direction, step=22.5):
        """
        Rotates CW or CCW to the nearest 45 deg angle
        - includes some fuzzyness to determine about which axis
        """
        camera = self.GetCurrentRenderer().GetActiveCamera()

        step = np.deg2rad(step)

        direction = -np.array(camera.GetViewPlaneNormal())  # current camera direction

        if abs(direction[2]) < 0.7:
            # horizontal view, rotate camera position about Z-axis
            angle = np.arctan2(direction[1], direction[0])

            # find the nearest angle that is an integer number of steps
            if movement_direction > 0:
                angle = step * np.floor((angle + 0.1 * step) / step) + step
            else:
                angle = -step * np.floor(-(angle - 0.1 * step) / step) - step

            dist = np.linalg.norm(direction[:2])
            direction[0] = np.cos(angle) * dist
            direction[1] = np.sin(angle) * dist

            self.set_camera_direction(direction)

        else:  # Top or bottom like view - rotate camera "up" direction

            up = np.array(camera.GetViewUp())
            angle = np.arctan2(up[1], up[0])

            # find the nearest angle that is an integer number of steps
            if movement_direction > 0:
                angle = step * np.floor((angle + 0.1 * step) / step) + step
            else:
                angle = -step * np.floor(-(angle - 0.1 * step) / step) - step

            dist = np.linalg.norm(up[:2])
            up[0] = np.cos(angle) * dist
            up[1] = np.sin(angle) * dist

            camera.SetViewUp(up)
            camera.OrthogonalizeViewUp()
            self.GetInteractor().Render()

    def toggle_parallel_projection(self):
        renderer = self.GetCurrentRenderer()
        camera = renderer.GetActiveCamera()
        camera.SetParallelProjection(not bool(camera.GetParallelProjection()))
        self.GetInteractor().Render()

    def set_view_x(self):
        self.set_camera_plane_direction((1, 0, 0))

    def set_view_y(self):
        self.set_camera_plane_direction((0, 1, 0))

    def set_view_z(self):
        self.set_camera_plane_direction((0, 0, 1))

    def zoom_fit(self):
        self.GetCurrentRenderer().ResetCamera()
        self.GetInteractor().Render()

    def set_camera_plane_direction(self, direction):
        """
        Sets the camera to display a plane of which direction is the normal
        - includes logic to reverse the direction if benificial
        """
        camera = self.GetCurrentRenderer().GetActiveCamera()

        direction = np.array(direction)
        normal = camera.GetViewPlaneNormal()
        # can not set the normal, need to change the position to do that

        current_alignment = np.dot(normal, -direction)
        if abs(current_alignment) > 0.9999:
            # print("toggling")
            direction = -np.array(normal)
        elif current_alignment > 0:  # find the nearest plane
            # print("reversing to find nearest")
            direction = -direction

        self.set_camera_direction(-direction)

    def set_camera_direction(self, direction):
        """Sets the camera to this direction, sets view up if horizontal enough"""
        direction = np.array(direction)

        ren = self.GetCurrentRenderer()
        camera = ren.GetActiveCamera()
        rwi = self.GetInteractor()

        pos = np.array(camera.GetPosition())
        focal = np.array(camera.GetFocalPoint())
        dist = np.linalg.norm(pos - focal)

        pos = focal - dist * direction
        camera.SetPosition(pos)

        if abs(direction[2]) < 0.9:
            camera.SetViewUp(0, 0, 1)
        elif direction[2] > 0.9:
            camera.SetViewUp(0, -1, 0)
        else:
            camera.SetViewUp(0, 1, 0)

        camera.OrthogonalizeViewUp()

        if self.GetAutoAdjustCameraClippingRange():
            ren.ResetCameraClippingRange()

        if rwi.GetLightFollowCamera():
            ren.UpdateLightsGeometryToFollowCamera()

        if self.callback_camera_direction_changed:
            self.callback_camera_direction_changed()

        self.GetInteractor().Render()

    def perform_picking_on_selection(self):
        """
        Performs 3d picking on the current dragged selection

        If the distance between the start and endpoints is less than the threshold
        then a SINGLE 3d prop is picked along the line.

        The selection area is drawn by the rubber band and is defined by
        `self.start_x, self.start_y, self.end_x, self.end_y`
        """
        renderer = self.GetCurrentRenderer()
        if not renderer:
            return []

        assemblyPath = renderer.PickProp(self.start_x, self.start_y, self.end_x, self.end_y)

        # re-pick in larger area if nothing is returned
        if not assemblyPath:
            self.start_x -= 2
            self.end_x += 2
            self.start_y -= 2
            self.end_y += 2
            assemblyPath = renderer.PickProp(self.start_x, self.start_y, self.end_x, self.end_y)

        # The nearest prop (by Z-value)
        if assemblyPath:
            assert (
                assemblyPath.GetNumberOfItems() == 1
            ), "Wrong assumption on number of returned nodes when picking"
            nearest_prop = assemblyPath.GetItemAsObject(0).GetViewProp()

            # all props
            collection = renderer.GetPickResultProps()
            props = [collection.GetItemAsObject(i) for i in range(collection.GetNumberOfItems())]

            props.remove(nearest_prop)
            props.insert(0, nearest_prop)
            return props

        else:
            return []

    # ----------- actor dragging ------------
    def start_drag(self):
        if self.callback_start_drag:
            # print("Calling callback_start_drag")
            self.callback_start_drag()
            return
        else:  # grab the current selection
            if self.picked_props:
                self.start_drag_on_props(self.picked_props)
            else:
                pass
                # print('Can not start drag, 
                # nothing selected and callback_start_drag not assigned')

    def finish_drag(self):
        # print('Finished drag')
        if self.callback_end_drag:
            # reset actor positions as actors positions will be controlled
            # by called functions
            for pos0, actor in zip(
                self.draginfo.dragged_actors_original_positions,
                self.draginfo.actors_dragging
            ):
                actor.SetPosition(pos0)
            self.callback_end_drag(self.draginfo)

        self.draginfo = None

    def start_drag_on_props(self, props):
        """
        Starts drag on the provided props (actors) by filling self.draginfo"""
        if self.draginfo is not None:
            self.finish_drag()
            return

        # create and fill drag-info
        draginfo = _BlenderStyleDragInfo()
        draginfo.actors_dragging = props  # [*actors, *outlines]

        for a in draginfo.actors_dragging:
            draginfo.dragged_actors_original_positions.append(a.GetPosition())  # numpy ndarray

        # Get the start position of the drag in 3d
        rwi = self.GetInteractor()
        ren = self.GetCurrentRenderer()
        camera = ren.GetActiveCamera()
        viewFocus = camera.GetFocalPoint()

        temp_out = [0, 0, 0]
        self.ComputeWorldToDisplay(
            ren, viewFocus[0], viewFocus[1], viewFocus[2],
            temp_out
        )
        focalDepth = temp_out[2]

        newPickPoint = [0, 0, 0, 0]
        x, y = rwi.GetEventPosition()
        self.ComputeDisplayToWorld(
            ren, x, y, focalDepth, newPickPoint)

        mouse_pos_3d = np.array(newPickPoint[:3])
        draginfo.start_position_3d = mouse_pos_3d
        self.draginfo = draginfo

    def execute_drag(self):

        rwi = self.GetInteractor()
        ren = self.GetCurrentRenderer()

        camera = ren.GetActiveCamera()
        viewFocus = camera.GetFocalPoint()

        # Get the picked point in 3d
        temp_out = [0, 0, 0]
        self.ComputeWorldToDisplay(
            ren, viewFocus[0], viewFocus[1], viewFocus[2], temp_out
        )
        focalDepth = temp_out[2]

        newPickPoint = [0, 0, 0, 0]
        x, y = rwi.GetEventPosition()
        self.ComputeDisplayToWorld(ren, x, y, focalDepth, newPickPoint)

        mouse_pos_3d = np.array(newPickPoint[:3])

        # compute the delta and execute

        delta = np.array(mouse_pos_3d) - self.draginfo.start_position_3d
        # print(f'Delta = {delta}')
        view_normal = np.array(ren.GetActiveCamera().GetViewPlaneNormal())

        delta_inplane = delta - view_normal * np.dot(delta, view_normal)
        # print(f'delta_inplane = {delta_inplane}')

        for pos0, actor in zip(
            self.draginfo.dragged_actors_original_positions, 
            self.draginfo.actors_dragging
        ):
            m = actor.GetUserMatrix()
            if m:
                print("UserMatrices/transforms not supported")
                # m.Invert() #inplace
                # rotated = m.MultiplyFloatPoint([*delta_inplane, 1])
                # actor.SetPosition(pos0 + np.array(rotated[:3]))
            actor.SetPosition(pos0 + delta_inplane)

        # print(f'Set position to {pos0 + delta_inplane}')
        self.draginfo.delta = delta_inplane  # store the current delta

        self.GetInteractor().Render()

    def cancel_drag(self):
        """Cancels the drag and restored the original positions of all dragged actors"""
        for pos0, actor in zip(
            self.draginfo.dragged_actors_original_positions, 
            self.draginfo.actors_dragging
        ):
            actor.SetPosition(pos0)
        self.draginfo = None
        self.GetInteractor().Render()

    # ----------- end dragging --------------

    def zoom(self):
        rwi = self.GetInteractor()
        x, y = rwi.GetEventPosition()
        xp, yp = rwi.GetLastEventPosition()

        direction = y - yp
        self.move_mouse_wheel(direction / 10)

    def pan(self):

        ren = self.GetCurrentRenderer()

        if ren:
            rwi = self.GetInteractor()

            #   // Calculate the focal depth since we'll be using it a lot
            camera = ren.GetActiveCamera()
            viewFocus = camera.GetFocalPoint()

            temp_out = [0, 0, 0]
            self.ComputeWorldToDisplay(
                ren, viewFocus[0], viewFocus[1], viewFocus[2], temp_out
            )
            focalDepth = temp_out[2]

            newPickPoint = [0, 0, 0, 0]
            x, y = rwi.GetEventPosition()
            self.ComputeDisplayToWorld(ren, x, y, focalDepth, newPickPoint)

            #   // Has to recalc old mouse point since the viewport has moved,
            #   // so can't move it outside the loop

            oldPickPoint = [0, 0, 0, 0]
            xp, yp = rwi.GetLastEventPosition()
            self.ComputeDisplayToWorld(ren, xp, yp, focalDepth, oldPickPoint)
            #
            #   // Camera motion is reversed
            #
            motionVector = (
                oldPickPoint[0] - newPickPoint[0],
                oldPickPoint[1] - newPickPoint[1],
                oldPickPoint[2] - newPickPoint[2],
            )

            viewFocus = camera.GetFocalPoint()  # do we need to do this again? Already did this
            viewPoint = camera.GetPosition()

            camera.SetFocalPoint(
                motionVector[0] + viewFocus[0],
                motionVector[1] + viewFocus[1],
                motionVector[2] + viewFocus[2],
            )
            camera.SetPosition(
                motionVector[0] + viewPoint[0],
                motionVector[1] + viewPoint[1],
                motionVector[2] + viewPoint[2],
            )

            if rwi.GetLightFollowCamera():
                ren.UpdateLightsGeometryToFollowCamera()

            self.GetInteractor().Render()

    def rotate(self):

        ren = self.GetCurrentRenderer()

        if ren:

            rwi = self.GetInteractor()
            dx = rwi.GetEventPosition()[0] - rwi.GetLastEventPosition()[0]
            dy = rwi.GetEventPosition()[1] - rwi.GetLastEventPosition()[1]

            size = ren.GetRenderWindow().GetSize()
            delta_elevation = -20.0 / size[1]
            delta_azimuth = -20.0 / size[0]

            rxf = dx * delta_azimuth * self.mouse_motion_factor
            ryf = dy * delta_elevation * self.mouse_motion_factor

            self.rotate_turtable_by(rxf, ryf)

    def rotate_turtable_by(self, rxf, ryf):

        ren = self.GetCurrentRenderer()
        rwi = self.GetInteractor()

        # rfx is rotation about the global Z vector (turn-table mode)
        # rfy is rotation about the side vector

        camera = ren.GetActiveCamera()
        campos = np.array(camera.GetPosition())
        focal = np.array(camera.GetFocalPoint())
        up = camera.GetViewUp()
        upside_down_factor = -1 if up[2] < 0 else 1

        # rotate about focal point

        P = campos - focal  # camera position

        # Rotate left/right about the global Z axis
        H = np.linalg.norm(P[:2])  # horizontal distance of camera to focal point
        elev = np.arctan2(P[2], H)  # elevation

        # if the camera is near the poles, then derive the azimuth from the up-vector
        sin_elev = np.sin(elev)
        if abs(sin_elev) < 0.8:
            azi = np.arctan2(P[1], P[0])  # azimuth from camera position
        else:
            if sin_elev < -0.8:
                azi = np.arctan2(upside_down_factor * up[1], upside_down_factor * up[0])
            else:
                azi = np.arctan2(-upside_down_factor * up[1], -upside_down_factor * up[0])

        D = np.linalg.norm(P)  # distance from focal point to camera

        # apply the change in azimuth and elevation
        azi_new = azi + rxf / 60

        elev_new = elev + upside_down_factor * ryf / 60

        # the changed elevation changes H (D stays the same)
        Hnew = D * np.cos(elev_new)

        # calculate new camera position relative to focal point
        Pnew = np.array((Hnew * np.cos(azi_new), Hnew * np.sin(azi_new), D * np.sin(elev_new)))

        # calculate the up-direction of the camera
        up_z = upside_down_factor * np.cos(elev_new)  # z follows directly from elevation
        up_h = upside_down_factor * np.sin(elev_new)  # horizontal component
        #
        # if upside_down:
        #     up_z = -up_z
        #     up_h = -up_h

        up = (-up_h * np.cos(azi_new), -up_h * np.sin(azi_new), up_z)

        new_pos = focal + Pnew

        camera.SetViewUp(up)
        camera.SetPosition(new_pos)

        camera.OrthogonalizeViewUp()

        # Update

        if self.GetAutoAdjustCameraClippingRange():
            ren.ResetCameraClippingRange()

        if rwi.GetLightFollowCamera():
            ren.UpdateLightsGeometryToFollowCamera()

        if self.callback_camera_direction_changed:
            self.callback_camera_direction_changed()

        self.GetInteractor().Render()

    def zoom_box(self, x1, y1, x2, y2):
        """Zooms to a box"""
        # int width, height;
        #   width = abs(this->EndPosition[0] - this->StartPosition[0]);
        #   height = abs(this->EndPosition[1] - this->StartPosition[1]);

        if x1 > x2:
            _ = x1
            x1 = x2
            x2 = _
        if y1 > y2:
            _ = y1
            y1 = y2
            y2 = _

        width = x2 - x1
        height = y2 - y1

        #   int *size = this->ren->GetSize();
        ren = self.GetCurrentRenderer()
        size = ren.GetSize()
        origin = ren.GetOrigin()
        camera = ren.GetActiveCamera()

        # Assuming we're drawing the band on the view-plane
        rbcenter = (x1 + width / 2, y1 + height / 2, 0)

        ren.SetDisplayPoint(rbcenter)
        ren.DisplayToView()
        ren.ViewToWorld()

        worldRBCenter = ren.GetWorldPoint()

        invw = 1.0 / worldRBCenter[3]
        worldRBCenter = [c * invw for c in worldRBCenter]
        winCenter = [origin[0] + 0.5 * size[0], origin[1] + 0.5 * size[1], 0]

        ren.SetDisplayPoint(winCenter)
        ren.DisplayToView()
        ren.ViewToWorld()

        worldWinCenter = ren.GetWorldPoint()
        invw = 1.0 / worldWinCenter[3]
        worldWinCenter = [c * invw for c in worldWinCenter]

        translation = [
            worldRBCenter[0] - worldWinCenter[0],
            worldRBCenter[1] - worldWinCenter[1],
            worldRBCenter[2] - worldWinCenter[2],
        ]

        pos = camera.GetPosition()
        fp = camera.GetFocalPoint()
        #
        pos = [pos[i] + translation[i] for i in range(3)]
        fp = [fp[i] + translation[i] for i in range(3)]

        #
        camera.SetPosition(pos)
        camera.SetFocalPoint(fp)

        if width > height:
            if width:
                camera.Zoom(size[0] / width)
        else:
            if height:
                camera.Zoom(size[1] / height)

        self.GetInteractor().Render()

    def focus_on(self, prop3D):
        """Move the camera to focus on this particular prop3D"""

        position = prop3D.GetPosition()

        # print(f"Focus on {position}")

        ren = self.GetCurrentRenderer()
        camera = ren.GetActiveCamera()

        fp = camera.GetFocalPoint()
        pos = camera.GetPosition()

        camera.SetFocalPoint(position)
        camera.SetPosition(
            position[0] - fp[0] + pos[0],
            position[1] - fp[1] + pos[1],
            position[2] - fp[2] + pos[2],
        )

        if self.GetAutoAdjustCameraClippingRange():
            ren.ResetCameraClippingRange()

        rwi = self.GetInteractor()
        if rwi.GetLightFollowCamera():
            ren.UpdateLightsGeometryToFollowCamera()

        self.GetInteractor().Render()

    def dolly(self, factor):
        ren = self.GetCurrentRenderer()

        if ren:
            camera = ren.GetActiveCamera()

            if camera.GetParallelProjection():
                camera.SetParallelScale(camera.GetParallelScale() / factor)
            else:
                camera.Dolly(factor)
                if self.GetAutoAdjustCameraClippingRange():
                    ren.ResetCameraClippingRange()

            # if not do_not_update:
            #     rwi = self.GetInteractor()
            #     if rwi.GetLightFollowCamera():
            #         ren.UpdateLightsGeometryToFollowCamera()
            #     # rwi.Render()

    def draw_measurement(self):
        rwi = self.GetInteractor()
        self.end_x, self.end_y = rwi.GetEventPosition()
        self.draw_line(self.start_x, self.end_x, self.start_y, self.end_y)

    def draw_dragged_selection(self):
        rwi = self.GetInteractor()
        self.end_x, self.end_y = rwi.GetEventPosition()
        self.draw_rubber_band(self.start_x, self.end_x, self.start_y, self.end_y)

    def initialize_screen_drawing(self):
        # make an image of the currently rendered image

        rwi = self.GetInteractor()
        rwin = rwi.GetRenderWindow()

        size = rwin.GetSize()

        self._pixel_array.Initialize()
        self._pixel_array.SetNumberOfComponents(4)
        self._pixel_array.SetNumberOfTuples(size[0] * size[1])

        front = 1  # what does this do?
        rwin.GetRGBACharPixelData(0, 0, size[0] - 1, size[1] - 1, front, self._pixel_array)

    def draw_rubber_band(self, x1, x2, y1, y2):
        rwi = self.GetInteractor()
        rwin = rwi.GetRenderWindow()

        size = rwin.GetSize()

        tempPA = vtki.vtkUnsignedCharArray()
        tempPA.DeepCopy(self._pixel_array)

        # check size, viewport may have been resized in the mean-time
        if tempPA.GetNumberOfTuples() != size[0] * size[1]:
            # print(
            #     "Starting new screen-image - viewport has resized without us knowing"
            # )
            self.initialize_screen_drawing()
            self.draw_rubber_band(x1, x2, y1, y2)
            return

        x2 = min(x2, size[0] - 1)
        y2 = min(y2, size[1] - 1)

        x2 = max(x2, 0)
        y2 = max(y2, 0)

        # Modify the pixel array
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        minx = min(x2, x1)
        miny = min(y2, y1)

        # draw top and bottom
        for i in range(width):

            # c = round((10*i % 254)/254) * 254  # find some alternating color
            c = 0

            idx = (miny * size[0]) + minx + i
            tempPA.SetTuple(idx, (c, c, c, 1))

            idx = ((miny + height) * size[0]) + minx + i
            tempPA.SetTuple(idx, (c, c, c, 1))

        # draw left and right
        for i in range(height):
            # c = round((10 * i % 254) / 254) * 254  # find some alternating color
            c = 0

            idx = ((miny + i) * size[0]) + minx
            tempPA.SetTuple(idx, (c, c, c, 1))

            idx = idx + width
            tempPA.SetTuple(idx, (c, c, c, 1))

        # and Copy back to the window
        rwin.SetRGBACharPixelData(0, 0, size[0] - 1, size[1] - 1, tempPA, 0)
        rwin.Frame()

    def line2pixels(self, x1, x2, y1, y2):
        """Returns the x and y values of the pixels on a line between x1,y1 and x2,y2.
        If start and end are identical then a single point is returned"""

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return [x1], [y1]

        if abs(dx) > abs(dy):
            dhdw = dy / dx
            r = range(0, dx, int(dx / abs(dx)))
            x = [x1 + i for i in r]
            y = [round(y1 + dhdw * i) for i in r]
        else:
            dwdh = dx / dy
            r = range(0, dy, int(dy / abs(dy)))
            y = [y1 + i for i in r]
            x = [round(x1 + i * dwdh) for i in r]

        return x, y

    def draw_line(self, x1, x2, y1, y2):
        rwi = self.GetInteractor()
        rwin = rwi.GetRenderWindow()

        size = rwin.GetSize()

        x1 = min(max(x1, 0), size[0])
        x2 = min(max(x2, 0), size[0])
        y1 = min(max(y1, 0), size[1])
        y2 = min(max(y2, 0), size[1])

        tempPA = vtki.vtkUnsignedCharArray()
        tempPA.DeepCopy(self._pixel_array)

        xs, ys = self.line2pixels(x1, x2, y1, y2)
        for x, y in zip(xs, ys):
            idx = (y * size[0]) + x
            tempPA.SetTuple(idx, (0, 0, 0, 1))

        # and Copy back to the window
        rwin.SetRGBACharPixelData(0, 0, size[0] - 1, size[1] - 1, tempPA, 0)

        camera = self.GetCurrentRenderer().GetActiveCamera()
        scale = camera.GetParallelScale()

        # Set/Get the scaling used for a parallel projection, i.e.
        #
        # the half of the height of the viewport in world-coordinate distances.
        # The default is 1. Note that the "scale" parameter works as an "inverse scale"
        #  larger numbers produce smaller images.
        # This method has no effect in perspective projection mode

        half_height = size[1] / 2
        # half_height [px] = scale [world-coordinates]

        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        meters_per_pixel = scale / half_height
        meters = length * meters_per_pixel

        if camera.GetParallelProjection():
            print(f"Line length = {length} px = {meters} m")
        else:
            print("Need to be in non-perspective mode to measure. Press 2 or 3 to get there")

        if self.callback_measure:
            self.callback_measure(meters)

        # # can we add something to the window here?
        # freeType = vtk.FreeTypeTools.GetInstance()
        # textProperty = vtki.vtkTextProperty()
        # textProperty.SetJustificationToLeft()
        # textProperty.SetFontSize(24)
        # textProperty.SetOrientation(25)
        #
        # textImage = vtki.vtkImageData()
        # freeType.RenderString(textProperty, "a somewhat longer text", 72, textImage)
        # # this does not give an error, assume it works
        # #
        # textImage.GetDimensions()
        # textImage.GetExtent()
        #
        # # # Now put the textImage in the RenderWindow
        # rwin.SetRGBACharPixelData(0, 0, size[0] - 1, size[1] - 1, textImage, 0)

        rwin.Frame()

    def _update_middle_mouse_button_lock_actor(self):

        if self.middle_mouse_lock_actor is None:
            # create the actor
            # Create a text on the top-rightcenter
            textMapper = vtki.new("TextMapper")
            textMapper.SetInput("Middle mouse lock [m or space] active")
            textProp = textMapper.GetTextProperty()
            textProp.SetFontSize(12)
            textProp.SetFontFamilyToTimes()
            textProp.BoldOff()
            textProp.ItalicOff()
            textProp.ShadowOff()
            textProp.SetVerticalJustificationToTop()
            textProp.SetJustificationToCentered()
            textProp.SetColor((0, 0, 0))

            self.middle_mouse_lock_actor = vtki.vtkActor2D()
            self.middle_mouse_lock_actor.SetMapper(textMapper)
            self.middle_mouse_lock_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
            self.middle_mouse_lock_actor.GetPositionCoordinate().SetValue(0.5, 0.98)

            self.GetCurrentRenderer().AddActor(self.middle_mouse_lock_actor)

        self.middle_mouse_lock_actor.SetVisibility(self.middle_mouse_lock)
        self.GetInteractor().Render()
