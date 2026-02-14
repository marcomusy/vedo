#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Orientation icon widget extracted from vedo.addons."""

import vedo.vtkclasses as vtki
from vedo import utils


class Icon(vtki.vtkOrientationMarkerWidget):
    """
    Add an inset icon mesh into the renderer.
    """

    def __init__(self, mesh, pos=3, size=0.08):
        """
        Add an inset icon mesh into the renderer.

        Arguments:
            pos : (list, int)
                icon position in the range [1-4] indicating one of the 4 corners,
                or it can be a tuple (x,y) as a fraction of the renderer size.
            size : (float)
                size of the icon space as fraction of the window size.
        """
        super().__init__()
        self.name = "Icon"

        try:
            self.SetOrientationMarker(mesh.actor)
        except AttributeError:
            self.SetOrientationMarker(mesh)

        if utils.is_sequence(pos):
            self.SetViewport(pos[0] - size, pos[1] - size, pos[0] + size, pos[1] + size)
        else:
            if pos < 2:
                self.SetViewport(0, 1 - 2 * size, size * 2, 1)
            elif pos == 2:
                self.SetViewport(1 - 2 * size, 1 - 2 * size, 1, 1)
            elif pos == 3:
                self.SetViewport(0, 0, size * 2, size * 2)
            elif pos == 4:
                self.SetViewport(1 - 2 * size, 0, 1, size * 2)
