"""
Mark a specific point on a mesh with some text.
"""
from vtkplotter import Sphere, Point, show, Text

sp = Sphere().wire(True)

pcoords = sp.getPoint(144)

pt = Point(pcoords, r=12, c="white")

tx = Text("my fave\npoint", pcoords, s=0.1, c="lightblue", bc="green", followcam=False)

show(sp, pt, tx, Text(__doc__), axes=1)
