"""Embed a 3D scene
in a webpage with x3d"""
from vedo import dataurl, Plotter, Volume, Text3D

plt = Plotter(size=(800,600), bg='GhostWhite')

embryo = Volume(dataurl+'embryo.tif').isosurface().decimate(0.5)
coords = embryo.points()
embryo.cmap('PRGn', coords[:,1]) # add dummy colors along y

txt = Text3D(__doc__, font='Bongas', s=350, c='red2', depth=0.05)
txt.pos(2500, 300, 500)

plt.show(embryo, txt, txt.box(padding=250), axes=1, viewup='z', zoom=1.2)

# This exports the scene and generates 2 files:
# embryo.x3d and an example embryo.html to inspect in the browser
plt.export('embryo.x3d', binary=False)

print("Type: \n firefox embryo.html")
