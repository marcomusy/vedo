#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

try:
    import vedo.vtkclasses as vtk
except ImportError:
    import vtkmodules.all as vtk

__docformat__ = "google"

__doc__ = """Submodule to customize interaction modes."""


class MousePan(vtk.vtkInteractorStyleUser):
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

        self.camera = None
        self.interactor = None
        self.renderer = None

        self.oldpickD = []
        self.newpickD = []
        self.oldpickW = np.array([0, 0, 0, 0], dtype=float)
        self.newpickW = np.array([0, 0, 0, 0], dtype=float)
        self.fpD      = np.array([0, 0, 0], dtype=float)
        self.fpW      = np.array([0, 0, 0], dtype=float)
        self.motionD  = np.array([0, 0], dtype=float)
        self.motionW  = np.array([0, 0, 0], dtype=float)

        self.AddObserver("LeftButtonPressEvent",    self.left_down)
        self.AddObserver("LeftButtonReleaseEvent",  self.left_up)
        self.AddObserver("MiddleButtonPressEvent",  self.middle_down)
        self.AddObserver("MiddleButtonReleaseEvent",self.middle_up)
        self.AddObserver("RightButtonPressEvent",   self.right_down)
        self.AddObserver("RightButtonReleaseEvent", self.right_up)
        self.AddObserver("MouseWheelForwardEvent",  self.wheel_forward)
        self.AddObserver("MouseWheelBackwardEvent", self.wheel_backward)
        self.AddObserver("MouseMoveEvent",          self._mouse_move)

    def get_motion(self):
        self.oldpickD = np.array(self.interactor.GetLastEventPosition())
        self.newpickD = np.array(self.interactor.GetEventPosition())
        self.motionD = (self.newpickD - self.oldpickD) / 4
        self.camera = self.renderer.GetActiveCamera()
        self.fpW = self.camera.GetFocalPoint()
        self.posW = self.camera.GetPosition()
        self.ComputeWorldToDisplay(
            self.renderer, self.fpW[0], self.fpW[1], self.fpW[2], self.fpD
        )
        focaldepth = self.fpD[2]
        self.ComputeDisplayToWorld(
            self.renderer, self.oldpickD[0], self.oldpickD[1], focaldepth, self.oldpickW
        )
        self.ComputeDisplayToWorld(
            self.renderer, self.newpickD[0], self.newpickD[1], focaldepth, self.newpickW
        )
        self.motionW[:3] = self.oldpickW[:3] - self.newpickW[:3]


    def mouse_left_move(self):
        self.get_motion()
        self.camera.SetFocalPoint(self.fpW[:3] + self.motionW[:3])
        self.camera.SetPosition(self.posW[:3] + self.motionW[:3])
        self.interactor.Render()

    def mouse_middle_move(self):
        self.get_motion()
        if abs(self.motionD[0]) > abs(self.motionD[1]):
            self.camera.Azimuth(-2 * self.motionD[0])
        else:
            self.camera.Elevation(-self.motionD[1])
        self.interactor.Render()

    def mouse_right_move(self):
        self.get_motion()
        if abs(self.motionD[0]) > abs(self.motionD[1]):
            self.camera.Azimuth(-2.0 * self.motionD[0])
        else:
            self.camera.Zoom(1 + self.motionD[1] / 100)
        self.interactor.Render()

    def mouse_wheel_forward(self):
        self.camera = self.renderer.GetActiveCamera()
        self.camera.Zoom(1.1)
        self.interactor.Render()

    def mouse_wheel_backward(self):
        self.camera = self.renderer.GetActiveCamera()
        self.camera.Zoom(0.9)
        self.interactor.Render()

    def left_down(self, w, e):
        self.left = True

    def left_up(self, w, e):
        self.left = False

    def middle_down(self, w, e):
        self.middle = True

    def middle_up(self, w, e):
        self.middle = False

    def right_down(self, w, e):
        self.right = True

    def right_up(self, w, e):
        self.right = False

    def wheel_forward(self, w, e):
        self.mouse_wheel_forward()

    def wheel_backward(self, w, e):
        self.mouse_wheel_backward()

    def _mouse_move(self, w, e):
        if self.left:
            self.mouse_left_move()
        if self.middle:
            self.mouse_middle_move()
        if self.right:
            self.mouse_right_move()

