#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Branding helpers extracted from vedo.shapes."""

import vedo
from vedo.colors import get_color


def vedo_logo(distance=0.0, c=None, bc="t", version=False, frame=True):
    """Create the 3D vedo logo."""
    from vedo.shapes import Text3D

    if c is None:
        c = (0, 0, 0)
        plt = vedo.current_plotter()
        if plt:
            if sum(get_color(plt.backgrcol)) > 1.5:
                c = [0, 0, 0]
            else:
                c = "linen"

    font = "Comae"
    vlogo = Text3D("vэdo", font=font, s=1350, depth=0.2, c=c, hspacing=0.8)
    vlogo.scale([1, 0.95, 1]).x(-2525).pickable(False).bc(bc)
    vlogo.properties.LightingOn()

    vr, rul = None, None
    if version:
        vr = Text3D(
            vedo.__version__, font=font, s=165, depth=0.2, c=c, hspacing=1
        ).scale([1, 0.7, 1])
        vr.rotate_z(90).pos(2450, 50, 80)
        vr.bc(bc).pickable(False)
    elif frame:
        rul = vedo.RulerAxes(
            (-2600, 2110, 0, 1650, 0, 0),
            xlabel="European Molecular Biology Laboratory",
            ylabel=vedo.__version__,
            font=font,
            xpadding=0.09,
            ypadding=0.04,
        )
    fakept = vedo.Point((0, 500, distance * 1725), alpha=0, c=c, r=1).pickable(0)
    return vedo.Assembly([vlogo, vr, fakept, rul]).scale(1 / 1725)
