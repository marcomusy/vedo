#!/usr/bin/env python3
from __future__ import annotations

"""Regression checks for Plotter in dry-run mode."""

import vedo


def test_plotter_dry_run_mode_uses_fallback_screen_size() -> None:
    previous_dry_run_mode = vedo.settings.dry_run_mode
    vedo.settings.dry_run_mode = 2

    try:
        plt = vedo.Plotter(size=(1200, 600))
        result = plt.show(vedo.Sphere(), interactive=False)
        plt.close()
    finally:
        vedo.settings.dry_run_mode = previous_dry_run_mode

    assert result is plt
    assert plt.window is not None
    assert plt.renderers
