#!/usr/bin/env python3
from __future__ import annotations

"""Regression checks for Slicer3DPlotter configuration."""

import numpy as np

from vedo import Volume
from vedo.addons import Slider2D
from vedo.applications import Slicer3DPlotter


def test_slicer3d_plotter_keeps_2d_sliders_with_explicit_shape() -> None:
    vol = Volume(np.arange(27, dtype=np.float32).reshape(3, 3, 3))

    plt = Slicer3DPlotter(vol, shape=[2, 1], offscreen=True)

    assert len(plt.renderers) == 2
    assert plt._use_slider3d is False
    assert isinstance(plt.xslider, Slider2D)
    assert isinstance(plt.yslider, Slider2D)
    assert isinstance(plt.zslider, Slider2D)

    plt.close()


def test_slicer3d_plotter_keeps_slices_in_target_renderer() -> None:
    vol = Volume(np.arange(27, dtype=np.float32).reshape(3, 3, 3))

    plt = Slicer3DPlotter(vol, N=2, offscreen=True)

    assert plt.renderers[0].HasViewProp(plt._box.actor) == 1
    assert plt.renderers[1].HasViewProp(plt._box.actor) == 0
    assert plt.renderers[0].HasViewProp(plt.zslice.actor) == 1
    assert plt.renderers[1].HasViewProp(plt.zslice.actor) == 0

    plt.close()


def test_slicer3d_plotter_replaces_previous_slice_actor() -> None:
    vol = Volume(np.arange(64, dtype=np.float32).reshape(4, 4, 4))

    plt = Slicer3DPlotter(vol, N=2, offscreen=True)
    ren0 = plt.renderers[0]

    first = plt.zslice
    assert first is not None

    plt._update_slice("z", 2, render=False)
    second = plt.zslice
    assert second is not None
    assert second is not first
    assert ren0.HasViewProp(first.actor) == 0
    assert ren0.HasViewProp(second.actor) == 1

    plt._update_slice("z", 1, render=False)
    third = plt.zslice
    assert third is not None
    assert third is not second
    assert ren0.HasViewProp(second.actor) == 0
    assert ren0.HasViewProp(third.actor) == 1

    plt.close()


def test_slicer3d_plotter_resets_camera_for_target_renderer_on_init() -> None:
    vol = Volume(np.arange(27, dtype=np.float32).reshape(3, 3, 3))

    plt = Slicer3DPlotter(vol, N=2, offscreen=True)
    cam = plt.renderers[0].GetActiveCamera()
    bounds = plt._box.bounds()
    center = (
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2,
    )

    assert np.allclose(cam.GetFocalPoint(), center)

    plt.close()
