#!/usr/bin/env python3
from __future__ import annotations

"""Regression checks for global axes type 6 (corner outline)."""

import numpy as np

import vedo.vtkclasses as vtki
from vedo import Actor2D, Box, Plotter, Text2D, Volume, addons
from vedo.plotter import keymap


class _FakeAxesInteractor:
    def __init__(self, key: str, renderer):
        self.key = key
        self.renderer = renderer
        self.rendered = False

    def GetKeySym(self):
        return self.key

    def GetShiftKey(self):
        return False

    def GetControlKey(self):
        return False

    def GetAltKey(self):
        return False

    def GetEventPosition(self):
        return 0, 0

    def FindPokedRenderer(self, _x, _y):
        return self.renderer

    def Render(self):
        self.rendered = True


def test_axes6_handles_raw_vtk_props_and_actor2d() -> None:
    plt = Plotter(offscreen=True)
    try:
        scalarbar = vtki.vtkScalarBarActor()
        xyplot = vtki.vtkXYPlotActor()
        xyplot.PickableOn()

        actor2d = Actor2D()
        actor2d.pickable(False)

        text2d = Text2D("corner-outline-test").pickable(False)
        vol = Volume(np.ones((6, 6, 6), dtype=np.float32))

        plt.objects = [scalarbar, xyplot, actor2d, text2d, vol]
        plt.renderer = plt.renderers[0]
        plt.axes_instances[0] = None

        addons.add_global_axes(axtype=6, c=None, bounds=vol.bounds())

        assert plt.axes == 6
        assert isinstance(plt.axes_instances[0], vtki.vtkActor)
        assert not actor2d.GetPickable()
        assert not text2d.actor.GetPickable()
    finally:
        plt.close()


def test_axes6_kp6_switch_with_scalarbar_and_xyplot() -> None:
    plt = Plotter(axes=7, offscreen=True)
    try:
        scalarbar = vtki.vtkScalarBarActor()
        xyplot = vtki.vtkXYPlotActor()
        xyplot.PickableOn()
        box = Box(size=(2, 3, 4))

        plt.add(box)
        plt.objects = [scalarbar, xyplot, box]
        plt.renderer = plt.renderers[0]
        plt.axes_instances[0] = vtki.vtkActor()

        iren = _FakeAxesInteractor("KP_6", plt.renderer)
        orig_interactor = plt.interactor
        plt.interactor = iren
        try:
            keymap.handle_default_keypress(plt, iren, None)
        finally:
            plt.interactor = orig_interactor

        assert iren.rendered
        assert plt.axes == 6
        assert isinstance(plt.axes_instances[0], vtki.vtkActor)
    finally:
        plt.close()


def test_axes6_no_pickable_3d_objects_returns_cleanly() -> None:
    plt = Plotter(offscreen=True)
    try:
        actor2d = Actor2D()
        box = Box().pickable(False)
        plt.objects = [actor2d, box]
        plt.renderer = plt.renderers[0]
        plt.axes_instances[0] = None

        addons.add_global_axes(axtype=6, c=None, bounds=box.bounds())

        assert plt.axes_instances[0] is None
    finally:
        plt.close()
