from __future__ import annotations
"""Bootstrap helpers used by Plotter constructor."""

import vedo
import vedo.vtkclasses as vtki


__docformat__ = "google"


def configure_renderer_common(renderer, bg, bg2=None, two_sided=False):
    """Apply common renderer/window flags used at Plotter startup."""
    renderer.SetLightFollowCamera(vedo.settings.light_follows_camera)
    if two_sided:
        renderer.SetTwoSidedLighting(vedo.settings.two_sided_lighting)

    renderer.SetUseDepthPeeling(vedo.settings.use_depth_peeling)
    if vedo.settings.use_depth_peeling:
        renderer.SetMaximumNumberOfPeels(vedo.settings.max_number_of_peels)
        renderer.SetOcclusionRatio(vedo.settings.occlusion_ratio)
    renderer.SetUseFXAA(vedo.settings.use_fxaa)
    renderer.SetPreserveDepthBuffer(vedo.settings.preserve_depth_buffer)

    renderer.SetBackground(vedo.get_color(bg))
    if bg2:
        renderer.GradientBackgroundOn()
        renderer.SetBackground2(vedo.get_color(bg2))


def apply_gradient_mode(renderer):
    """Apply configured background gradient mode when supported by VTK build."""
    if vedo.settings.background_gradient_orientation <= 0:
        return
    try:
        modes = [
            vtki.vtkViewport.GradientModes.VTK_GRADIENT_VERTICAL,
            vtki.vtkViewport.GradientModes.VTK_GRADIENT_HORIZONTAL,
            vtki.vtkViewport.GradientModes.VTK_GRADIENT_RADIAL_VIEWPORT_FARTHEST_SIDE,
            vtki.vtkViewport.GradientModes.VTK_GRADIENT_RADIAL_VIEWPORT_FARTHEST_CORNER,
        ]
        renderer.SetGradientMode(modes[vedo.settings.background_gradient_orientation])
        renderer.GradientBackgroundOn()
    except AttributeError:
        pass
