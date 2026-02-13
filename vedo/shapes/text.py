#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Text-related helpers and shapes.

Compatibility facade that re-exports text symbols from focused modules.
"""

from vedo.shapes.text_utils import _reps
from vedo.shapes.text3d import Text3D
from vedo.shapes.text2d import Text2D
from vedo.shapes.latex import Latex

__all__ = ["Text3D", "Text2D", "Latex", "_reps"]
