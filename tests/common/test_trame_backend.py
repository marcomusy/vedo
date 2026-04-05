#!/usr/bin/env python3
from __future__ import annotations

"""Smoke check for trame backend wiring."""

import contextlib
import io

import pytest

import vedo


@pytest.mark.optional_dependency
@pytest.mark.rendering
def test_trame_backend_wiring() -> None:
    pytest.importorskip("trame")
    pytest.importorskip("trame_vtk")
    pytest.importorskip("trame_vuetify")

    previous_backend = vedo.settings.default_backend
    vedo.settings.default_backend = "trame"

    plt = vedo.Plotter(offscreen=True)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = plt.show(vedo.Sphere(), interactive=False)
        server_ok = hasattr(plt, "server") and plt.server is not None
        controller_ok = hasattr(plt, "controller") and plt.controller is not None
        state_ok = hasattr(plt, "state") and plt.state is not None
    plt.close()
    vedo.settings.default_backend = previous_backend

    message = output.getvalue()
    assert result is not None
    assert server_ok
    assert controller_ok
    assert state_ok
    assert "trame-vtk" not in message
    assert "trame-vuetify" not in message
