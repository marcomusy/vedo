"""Draw a continuous line on an image with the DrawingWidget."""
from vedo import DrawingWidget, Plotter, Image, dataurl

img = Image(dataurl + "embryo.jpg").resize(0.5)

plt = Plotter(axes=1)

drw = DrawingWidget(img)
drw.on()
cid = drw.add_observer("end interaction", lambda w, e: print(drw.line))

plt.show(img, __doc__, zoom=1.2)

drw.remove()
plt.close()
