"""Show a cube for each available color name"""
print(__doc__)
from vedo import Cube, Text2D, show
from vedo.colors import colors, getColor
from operator import itemgetter

# sorting by hex color code (matplotlib colors):
sorted_colors1 = sorted(colors.items(), key=itemgetter(1))
cbs=[]
for sc in sorted_colors1:
    cname = sc[0]
    if cname[-1] in "123456789": continue
    cb = Cube().lw(1).color(cname)
    tname = Text2D(cname, pos=3)
    cbs.append([tname, cb])
print("click on any cube and press i or I")
show(cbs, N=len(cbs), azimuth=.2, size='full', title="matplotlib colors").close()

# sort by name (bootstrap5 colors):
sorted_colors2 = sorted(colors.items(), key=itemgetter(0))
cbs = []
for sc in sorted_colors2:
    cname = sc[0]
    if cname[-1] not in "123456789": continue
    cb = Cube().lw(1).lighting('off').color(cname)
    tname = Text2D(cname, pos=3)
    cbs.append([tname, cb])
show(cbs, shape=(11,9), azimuth=.2, size=(800,1000), title="bootstrap5 colors")


