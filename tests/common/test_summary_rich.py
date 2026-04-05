#!/usr/bin/env python3
from __future__ import annotations

from rich.console import Console

import vedo


def test_summary_rich_output() -> None:
    pts = vedo.Points([[0, 0, 0], [1, 0, 0], [0, 1, 0]], c="red")
    pts.metadata["tag"] = ["demo"]
    plain_text = str(pts)
    assert "pointcloud.core.Points at (" in plain_text
    assert "elements" in plain_text
    assert "metadata" in plain_text

    pts_console = Console(record=True, force_terminal=False, width=100)
    pts_console.print(pts)
    pts_rich_text = pts_console.export_text()
    assert "elements" in pts_rich_text
    assert "position" in pts_rich_text
    assert "metadata" in pts_rich_text

    mesh = vedo.Cube()
    mesh_console = Console(record=True, force_terminal=False, width=100)
    mesh_console.print(mesh)
    mesh_rich_text = mesh_console.export_text()
    assert "Cube at (" in mesh_rich_text
    assert "elements" in mesh_rich_text
    assert "bounds" in mesh_rich_text
