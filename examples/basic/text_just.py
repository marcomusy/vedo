"""
Text jutification and positioning:
 top-left, top-right, bottom-left, bottom-right,
 center-left, center-right, center-left, center-right
"""
print(__doc__)
from vtkplotter import show, Text, Point

txt = "Text\njustification\n& position"

pos = [1, 2, 0]

tx = Text(txt, pos, s=1, depth=0.1, justify="bottom-left")

t0 = Point(pos, c="r")  # mark text position
ax = Point(c="blue")    # mark axes origin

show(tx,
     tx.box().c("y"),
     t0,
     ax,
     axes=8, bg="lb", size=(500,500)
     )
