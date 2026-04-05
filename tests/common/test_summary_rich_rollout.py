#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from rich.console import Console

import vedo
from vedo.applications.chemistry import PeriodicTable
from vedo.plotter.events import Event


def capture_text(obj) -> str:
    console = Console(record=True, force_terminal=False, width=100)
    console.print(obj)
    txt = console.export_text()
    return txt


def test_summary_rich_rollout() -> None:
    vol = vedo.Volume(np.zeros((2, 2, 2), dtype=np.float32))
    assert "dimensions" in capture_text(vol)
    assert "scalar range" in capture_text(vol)

    img = vedo.Image(np.zeros((2, 3, 3), dtype=np.uint8))
    assert "dimensions" in capture_text(img)
    assert "level/window" in capture_text(img)

    x, y, z = np.meshgrid(np.arange(2), np.arange(2), np.arange(2), indexing="ij")
    sgrid = vedo.StructuredGrid([x, y, z])
    assert "dimensions" in capture_text(sgrid)
    assert "memory size" in capture_text(sgrid)

    rgrid = vedo.RectilinearGrid([np.arange(2), np.arange(2), np.arange(2)])
    assert "dimensions" in capture_text(rgrid)
    assert "bounds" in capture_text(rgrid)

    egrid = vedo.ExplicitStructuredGrid([x, y, z])
    egrid_txt = capture_text(egrid)
    assert "cell dimensions" in egrid_txt
    assert "data dimension" in egrid_txt

    group = vedo.Group([vedo.Cube(), vedo.Sphere()])
    assert "n. of objects" in capture_text(group)

    assembly = vedo.Assembly(vedo.Cube(), vedo.Sphere())
    assert "position" in capture_text(assembly)
    assert "bounds" in capture_text(assembly)

    plt = vedo.Plotter(offscreen=True, axes=1, title="summary test")
    try:
        plt_txt = capture_text(plt)
        assert "n. of objects" in plt_txt
        assert "axes style" in plt_txt
    finally:
        plt.close()

    evt = Event()
    evt.name = "LeftButtonPress"
    evt.isMesh = True
    evt_txt = capture_text(evt)
    assert "LeftButtonPress" in evt_txt
    assert "isMesh" in evt_txt

    periodic = PeriodicTable()
    periodic_txt = capture_text(periodic)
    assert "Number of elements" in periodic_txt
    assert "Element symbol" in periodic_txt

    minimizer = vedo.utils.Minimizer(lambda vals: (vals[0] - 1.0) ** 2)
    minimizer.set_parameters({"x": (0.0, 0.1)})
    minimizer.minimize()
    minimizer_txt = capture_text(minimizer)
    assert "Function name" in minimizer_txt
    assert "Hessian Matrix" in minimizer_txt

    settings_txt = capture_text(vedo.settings)
    assert "default_font" in settings_txt
