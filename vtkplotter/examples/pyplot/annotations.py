"""Show some text as a corner annotation.
Fonts: arial, courier, times.
"""
from vtkplotter import show, Text2D, Cube

with open("annotations.py") as fname:
    t = fname.read()

txt2d = Text2D(t, pos=3, s=1.2, c='k', bg="lb", font="courier")

show(txt2d, Cube(), verbose=0, axes=0)
