"""Camera/view operations delegated from Plotter."""

from typing import Any

import numpy as np

import vedo
import vedo.vtkclasses as vtki
from vedo import addons
from vedo import utils


__docformat__ = "google"


def reset_camera(plotter, tight=None) -> Any:
    """
    Reset the camera position and zooming.
    If tight (float) is specified the zooming reserves a padding space
    in the xy-plane expressed in percent of the average size.
    """
    if tight is None:
        plotter.renderer.ResetCamera()
    else:
        x0, x1, y0, y1, z0, z1 = plotter.renderer.ComputeVisiblePropBounds()
        cam = plotter.camera

        plotter.renderer.ComputeAspect()
        aspect = plotter.renderer.GetAspect()
        angle = np.pi * cam.GetViewAngle() / 180.0
        dx = x1 - x0
        dy = y1 - y0
        dist = max(dx / aspect[0], dy) / np.tan(angle / 2) / 2

        cam.SetViewUp(0, 1, 0)
        cam.SetPosition(x0 + dx / 2, y0 + dy / 2, dist * (1 + tight))
        cam.SetFocalPoint(x0 + dx / 2, y0 + dy / 2, 0)
        if cam.GetParallelProjection():
            ps = max(dx / aspect[0], dy) / 2
            cam.SetParallelScale(ps * (1 + tight))
        plotter.renderer.ResetCameraClippingRange(x0, x1, y0, y1, z0, z1)
    return plotter

def reset_clipping_range(plotter, bounds=None) -> Any:
    """
    Reset the camera clipping range to include all visible actors.
    If bounds is given, it will be used instead of computing it.
    """
    if bounds is None:
        plotter.renderer.ResetCameraClippingRange()
    else:
        plotter.renderer.ResetCameraClippingRange(bounds)
    return plotter

def reset_viewup(plotter, smooth=True) -> Any:
    """
    Reset the orientation of the camera to the closest orthogonal direction and view-up.
    """
    vbb = addons.compute_visible_bounds()[0]
    x0, x1, y0, y1, z0, z1 = vbb
    mx, my, mz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    d = plotter.camera.GetDistance()

    viewups = np.array(
        [(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0)]
    )
    positions = np.array(
        [
            (mx, my, mz + d),
            (mx, my, mz - d),
            (mx, my + d, mz),
            (mx, my - d, mz),
            (mx + d, my, mz),
            (mx - d, my, mz),
        ]
    )

    vu = np.array(plotter.camera.GetViewUp())
    vui = np.argmin(np.linalg.norm(viewups - vu, axis=1))

    poc = np.array(plotter.camera.GetPosition())
    foc = np.array(plotter.camera.GetFocalPoint())
    a = poc - foc
    b = positions - foc
    a = a / np.linalg.norm(a)
    b = b.T * (1 / np.linalg.norm(b, axis=1))
    pui = np.argmin(np.linalg.norm(b.T - a, axis=1))

    if smooth:
        outtimes = np.linspace(0, 1, num=11, endpoint=True)
        for t in outtimes:
            vv = vu * (1 - t) + viewups[vui] * t
            pp = poc * (1 - t) + positions[pui] * t
            ff = foc * (1 - t) + np.array([mx, my, mz]) * t
            plotter.camera.SetViewUp(vv)
            plotter.camera.SetPosition(pp)
            plotter.camera.SetFocalPoint(ff)
            plotter.render()

        # interpolator does not respect parallel view...:
        # cam1 = dict(
        #     pos=poc,
        #     viewup=vu,
        #     focal_point=(mx,my,mz),
        #     clipping_range=plotter.camera.GetClippingRange()
        # )
        # # cam1 = plotter.camera
        # cam2 = dict(
        #     pos=positions[pui],
        #     viewup=viewups[vui],
        #     focal_point=(mx,my,mz),
        #     clipping_range=plotter.camera.GetClippingRange()
        # )
        # vcams = plotter.move_camera([cam1, cam2], output_times=outtimes, smooth=0)
        # for c in vcams:
        #     plotter.renderer.SetActiveCamera(c)
        #     plotter.render()
    else:

        plotter.camera.SetViewUp(viewups[vui])
        plotter.camera.SetPosition(positions[pui])
        plotter.camera.SetFocalPoint(mx, my, mz)

    plotter.renderer.ResetCameraClippingRange()

    # vbb, _, _, _ = addons.compute_visible_bounds()
    # x0,x1, y0,y1, z0,z1 = vbb
    # plotter.renderer.ResetCameraClippingRange(x0, x1, y0, y1, z0, z1)
    plotter.render()
    return plotter

def move_camera(plotter, cameras, t=0, times=(), smooth=True, output_times=()) -> list:
    """
    Takes as input two cameras set camera at an interpolated position:

    Cameras can be vtkCamera or dictionaries in format:

        `dict(pos=..., focal_point=..., viewup=..., distance=..., clipping_range=...)`

    Press `shift-C` key in interactive mode to dump a python snipplet
    of parameters for the current camera view.
    """
    nc = len(cameras)
    if len(times) == 0:
        times = np.linspace(0, 1, num=nc, endpoint=True)

    assert len(times) == nc

    cin = vtki.new("CameraInterpolator")

    # cin.SetInterpolationTypeToLinear() # buggy?
    if nc > 2 and smooth:
        cin.SetInterpolationTypeToSpline()

    for i, cam in enumerate(cameras):
        vcam = cam
        if isinstance(cam, dict):
            vcam = utils.camera_from_dict(cam)
        cin.AddCamera(times[i], vcam)

    mint, maxt = cin.GetMinimumT(), cin.GetMaximumT()
    rng = maxt - mint

    if len(output_times) == 0:
        cin.InterpolateCamera(t * rng, plotter.camera)
        return [plotter.camera]
    else:
        vcams = []
        for tt in output_times:
            c = vtki.vtkCamera()
            cin.InterpolateCamera(tt * rng, c)
            vcams.append(c)
        return vcams

def fly_to(plotter, point) -> Any:
    """
    Fly camera to the specified point.

    Arguments:
        point : (list)
            point in space to place camera.

    Example:
        ```python
        from vedo import *
        cone = Cone()
        plt = Plotter(axes=1)
        plt.show(cone)
        plt.fly_to([1,0,0])
        plt.interactive().close()
        ```
    """
    if plotter.interactor:
        plotter.resetcam = False
        plotter.interactor.FlyTo(plotter.renderer, point)
    return plotter

def look_at(plotter, plane="xy") -> Any:
    """Move the camera so that it looks at the specified cartesian plane"""
    cam = plotter.renderer.GetActiveCamera()
    fp = np.array(cam.GetFocalPoint())
    p = np.array(cam.GetPosition())
    dist = np.linalg.norm(fp - p)
    plane = plane.lower()
    if "x" in plane and "y" in plane:
        cam.SetPosition(fp[0], fp[1], fp[2] + dist)
        cam.SetViewUp(0.0, 1.0, 0.0)
    elif "x" in plane and "z" in plane:
        cam.SetPosition(fp[0], fp[1] - dist, fp[2])
        cam.SetViewUp(0.0, 0.0, 1.0)
    elif "y" in plane and "z" in plane:
        cam.SetPosition(fp[0] + dist, fp[1], fp[2])
        cam.SetViewUp(0.0, 0.0, 1.0)
    else:
        vedo.logger.error(f"in plotter.look() cannot understand argument {plane}")
    return plotter

def parallel_projection(plotter, value=True, at=None) -> Any:
    """
    Use parallel projection `at` a specified renderer.
    Object is seen from "infinite" distance, e.i. remove any perspective effects.
    An input value equal to -1 will toggle it on/off.
    """
    r = plotter.renderer if at is None else plotter.renderers[at]

    if value == -1:
        val = r.GetActiveCamera().GetParallelProjection()
        value = not val
    r.GetActiveCamera().SetParallelProjection(value)
    r.Modified()
    return plotter

def render_hidden_lines(plotter, value=True) -> Any:
    """Remove hidden lines when in wireframe mode."""
    plotter.renderer.SetUseHiddenLineRemoval(not value)
    return plotter

def fov(plotter, angle: float) -> Any:
    """
    Set the field of view angle for the camera.
    This is the angle of the camera frustum in the horizontal direction.
    High values will result in a wide-angle lens (fish-eye effect),
    and low values will result in a telephoto lens.

    Default value is 30 degrees.
    """
    plotter.renderer.GetActiveCamera().UseHorizontalViewAngleOn()
    plotter.renderer.GetActiveCamera().SetViewAngle(angle)
    return plotter

def zoom(plotter, zoom: float) -> Any:
    """Apply a zooming factor for the current camera view"""
    plotter.renderer.GetActiveCamera().Zoom(zoom)
    return plotter

def azimuth(plotter, angle: float) -> Any:
    """Rotate camera around the view up vector."""
    plotter.renderer.GetActiveCamera().Azimuth(angle)
    return plotter

def elevation(plotter, angle: float) -> Any:
    """Rotate the camera around the cross product of the negative
    of the direction of projection and the view up vector."""
    plotter.renderer.GetActiveCamera().Elevation(angle)
    return plotter

def roll(plotter, angle: float) -> Any:
    """Roll the camera about the direction of projection."""
    plotter.renderer.GetActiveCamera().Roll(angle)
    return plotter

def dolly(plotter, value: float) -> Any:
    """Move the camera towards (value>0) or away from (value<0) the focal point."""
    plotter.renderer.GetActiveCamera().Dolly(value)
    return plotter
