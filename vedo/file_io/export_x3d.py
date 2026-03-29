from __future__ import annotations
"""X3D scene export helpers."""

import vedo.vtkclasses as vtki

from .constants import _x3d_html_template

__docformat__ = "google"


def _export_x3d(plt, fileoutput="scene.x3d", binary=False) -> None:
    """Export the current scene to X3D plus a small HTML loader page."""
    plt.render()

    exporter = vtki.new("X3DExporter")
    exporter.SetBinary(binary)
    exporter.FastestOff()
    exporter.SetInput(plt.window)
    exporter.SetFileName(fileoutput)
    exporter.Update()
    exporter.Write()

    wsize = plt.window.GetSize()
    x3d_html = _x3d_html_template.replace("~fileoutput", fileoutput)
    x3d_html = x3d_html.replace("~width", str(wsize[0]))
    x3d_html = x3d_html.replace("~height", str(wsize[1]))
    with open(fileoutput.replace(".x3d", ".html"), "w", encoding="UTF-8") as outF:
        outF.write(x3d_html)
