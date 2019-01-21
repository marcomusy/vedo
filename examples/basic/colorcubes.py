# Show a cube for each available color name
#
from vtkplotter import Plotter, cube, text
from vtkplotter.colors import colors, getColor
from operator import itemgetter

# sorting by hex color code:
sorted_colors = sorted(colors.items(), key=itemgetter(1))
# or by name:
# sorted_colors = sorted(colors.items(), key=itemgetter(0))


vp = Plotter(N=len(sorted_colors), axes=0, size='fullscreen')

for i, sc in enumerate(sorted_colors):
    cname = sc[0]
    rgb = getColor(cname)
    cb = cube(c=rgb)
    tname = text(cname, pos=3)
    vp.show([cb, tname], at=i)

print('click on any cube and press i')
vp.show(zoom=1.1, interactive=1)

