"""
Show a cube for each available color name
"""
print(__doc__)
from vedo import Cube, Text2D, show
from vedo.colors import colors, getColor

from operator import itemgetter

# sorting by hex color code:
sorted_colors = sorted(colors.items(), key=itemgetter(1))
# or by name:
# sorted_colors = sorted(colors.items(), key=itemgetter(0))

cbs=[]
for i, sc in enumerate(sorted_colors):
    cname = sc[0]
    rgb = getColor(cname)
    cb = Cube(c=rgb)
    tname = Text2D(cname, pos=3)
    cbs.append([tname, cb])

print("click on any cube and press i")
show(cbs, N=len(sorted_colors), azimuth=.2, size='fullscreen')

