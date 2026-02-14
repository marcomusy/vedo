"""Launch the interactive image editor application."""
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# A simple image editor that allows the user to apply various filters
import sys
from vedo import dataurl
from vedo.applications import ImageEditor


if len(sys.argv) > 1:
    filename = sys.argv[1]
else:  # use a default image
    filename = dataurl + "e3_EGFP.jpg"
    print(f"Using default image: {filename}")

editor = ImageEditor(filename)
editor.start()
editor.close()
