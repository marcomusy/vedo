"""
Show some text as a corner annotation.
Fonts: arial, courier, times.
"""
from vtkplotter import show, Text, Cube

with open("annotations.py") as fname:
    t = fname.read()

actor2d = Text(t, pos=3, s=1.2, c='k', bg="lb", font="courier")

show(actor2d, Cube(), verbose=0, axes=0)
