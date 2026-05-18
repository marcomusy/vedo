#!/usr/bin/env python3
from __future__ import annotations

import vedo.vtkclasses as vtki
from vedo.plotter.modes import BlenderStyle, FlyOverSurface, MousePan


class _RenderCountingInteractor:
    def __init__(self, render_window=None):
        self.render_count = 0
        self.render_window = render_window

    def Render(self):
        self.render_count += 1

    def GetRenderWindow(self):
        return self.render_window


class _KeyEvent:
    def __init__(self, key: str):
        self.key = key

    def GetKeySym(self):
        return self.key

    def GetShiftKey(self):
        return False


def test_mouse_pan_can_disable_left_drag_pan() -> None:
    mode = MousePan(enable_pan=False, enable_zoom=True)
    calls = []
    mode.left = True
    mode._mouse_left_move = lambda: calls.append("pan")

    mode._mouse_move(None, None)

    assert calls == []


class _FlyCamera:
    def __init__(self):
        self.position = None
        self.focal_point = None
        self.view_up = None

    def SetPosition(self, position):
        self.position = tuple(position)

    def SetFocalPoint(self, focal_point):
        self.focal_point = tuple(focal_point)

    def SetViewUp(self, *view_up):
        self.view_up = tuple(view_up)


class _FlyRenderer:
    def __init__(self, bounds):
        self.bounds = bounds
        self.camera = _FlyCamera()
        self.reset_count = 0

    def GetActiveCamera(self):
        return self.camera

    def ComputeVisiblePropBounds(self):
        return self.bounds

    def ResetCameraClippingRange(self):
        self.reset_count += 1


def test_fly_over_surface_x_reset_uses_bounds_z_origin() -> None:
    mode = FlyOverSurface()
    mode.renderer = _FlyRenderer((0, 1000, 0, 0, 100, 110))
    interactor = vtki.vtkRenderWindowInteractor()
    mode.SetInteractor(interactor)

    mode._key(_KeyEvent("X"), "KeyPressEvent")

    assert mode.renderer.camera.position == (2000, 0, 350)
    assert mode.renderer.camera.focal_point == (-500, 0, 350)


def test_blender_style_escape_clears_box_zoom_mode() -> None:
    mode = BlenderStyle()
    interactor = _RenderCountingInteractor()
    mode.GetInteractor = lambda: interactor
    mode._is_box_zooming = True
    mode._left_button_down = True

    mode.key_press(_KeyEvent("Escape"), None)

    assert mode._is_box_zooming is False
    assert mode._left_button_down is False
    assert interactor.render_count == 1


class _RubberBandWindow:
    def __init__(self):
        self.frame_count = 0
        self.pixel_data_calls = []

    def GetSize(self):
        return 10, 10

    def SetRGBACharPixelData(self, *args):
        self.pixel_data_calls.append(args)

    def Frame(self):
        self.frame_count += 1


def test_blender_style_rubber_band_clamps_both_corners() -> None:
    mode = BlenderStyle()
    window = _RubberBandWindow()
    mode.GetInteractor = lambda: _RenderCountingInteractor(window)
    mode._pixel_array = vtki.vtkUnsignedCharArray()
    mode._pixel_array.SetNumberOfComponents(4)
    mode._pixel_array.SetNumberOfTuples(100)

    mode.draw_rubber_band(-5, 12, -3, 12)

    assert window.frame_count == 1
    assert len(window.pixel_data_calls) == 1
