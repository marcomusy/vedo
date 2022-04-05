"""Show a cube for each available color name"""
print(__doc__)
from vedo import Cube, Text2D, show, settings
from vedo.colors import colors
from operator import itemgetter

settings.immediateRendering = False  # faster for multi-renderers

# sorting by hex color code (matplotlib colors):
sorted_colors1 = sorted(colors.items(), key=itemgetter(1))
cbs=[]
for sc in sorted_colors1:
    cname = sc[0]
    if cname[-1] in "123456789": continue
    cb = Cube().lw(1).color(cname)
    tname = Text2D(cname, s=0.9)
    cbs.append([tname, cb])
print("click on any cube and press i or I")
plt1= show(cbs, N=len(cbs), azimuth=.2, size=(2100,1300),
           title="matplotlib colors", interactive=0)
plt1.render()  # because of immediateRendering=False

# sort by name (bootstrap5 colors):
sorted_colors2 = sorted(colors.items(), key=itemgetter(0))
cbs = []
for sc in sorted_colors2:
    cname = sc[0]
    if cname[-1] not in "123456789": continue
    cb = Cube().lw(1).lighting('off').color(cname)
    cbs.append([cname, cb])
plt2= show(cbs, shape=(11,9), azimuth=.2, size=(800,1000),
           title="bootstrap5 colors", new=True)

plt2.close()
plt1.close()
