from __future__ import annotations
"""I/O operations delegated from Plotter."""

from typing import Any

import vedo
import vedo.vtkclasses as vtki


__docformat__ = "google"


def screenshot(plotter, filename="screenshot.png", scale=1, asarray=False) -> Any:
    """
    Take a screenshot of the Plotter window.

    Arguments:
        scale : (int)
            set image magnification as an integer multiplicating factor
        asarray : (bool)
            return a numpy array of the image instead of writing a file

    Warning:
        If you get black screenshots try to set `interactive=False` in `show()`
        then call `screenshot()` and `plt.interactive()` afterwards.

    Example:
        ```py
        from vedo import *
        sphere = Sphere().linewidth(1)
        plt = show(sphere, interactive=False)
        plt.screenshot('image.png')
        plt.interactive()
        plt.close()
        ```

    Example:
        ```py
        from vedo import *
        sphere = Sphere().linewidth(1)
        plt = show(sphere, interactive=False)
        plt.screenshot('anotherimage.png')
        plt.interactive()
        plt.close()
        ```
    """
    return vedo.file_io.screenshot(filename, scale, asarray)

def toimage(plotter, scale=1) -> "vedo.image.Image":
    """
    Generate a `Image` object from the current rendering window.

    Arguments:
        scale : (int)
            set image magnification as an integer multiplicating factor
    """
    if vedo.settings.screeshot_large_image:
        w2if = vtki.new("RenderLargeImage")
        w2if.SetInput(plotter.renderer)
        w2if.SetMagnification(scale)
    else:
        w2if = vtki.new("WindowToImageFilter")
        w2if.SetInput(plotter.window)
        if hasattr(w2if, "SetScale"):
            w2if.SetScale(scale, scale)
        if vedo.settings.screenshot_transparent_background:
            w2if.SetInputBufferTypeToRGBA()
        w2if.ReadFrontBufferOff()  # read from the back buffer
    w2if.Update()
    return vedo.image.Image(w2if.GetOutput())

def export(plotter, filename="scene.npz", binary=False) -> Any:
    """
    Export scene to file to HTML, X3D or Numpy file.

    Examples:
        - [export_x3d.py](https://github.com/marcomusy/vedo/tree/master/examples/other/export_x3d.py)
        - [export_numpy.py](https://github.com/marcomusy/vedo/tree/master/examples/other/export_numpy.py)
    """
    vedo.file_io.export_window(filename, binary=binary)
    return plotter
