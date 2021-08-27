"""
Example of drawing objects on different windows
and/or subwindows within the same window.
We split the main window in many subwindows and draw
somethingon specific windows numbers.
Then open an independent window and draw a shape on it.
"""
print(__doc__)
from vedo import *

##########################################################################
# this is one instance of the class Plotter with 5 raws and 5 columns
vp1 = Plotter(shape=(5,5), axes=0)

# set a different background color for a specific subwindow (the last one)
vp1.renderers[24].SetBackground(0.8, 0.9, 0.9)  # use vtk method SetBackground()

# load the meshes and give them a name
a = vp1.load(dataurl+"shuttle.obj")
b = vp1.load(dataurl+"cessna.vtk").c("red")
c = vp1.load(dataurl+"porsche.ply")

# show a Text2D in each renderer
for i in range(25):
    vp1.show("renderer\nnr."+str(i), at=i)

vp1.show(a, at= 6)
vp1.show(b, at=23)
vp1.show(c, at=24)


##########################################################################
# declare a second independent instance of the class Plotter
# shape can also be given as a string, e.g.:
# shape="2/6" means 2 renderers above and 6 below
# shape="3|1" means 3 renderers on the left and one on the right

s = load(dataurl+'mug.ply')

# Set the position of the horizontal of vertical splitting [0,1]:
#settings.windowSplittingPosition = 0.5

vp2 = Plotter(pos=(500, 250), shape='2/6')

for i in range(len(vp2.renderers)):
    s2 = s.clone(deep=False).color(i)
    vp2.show(s2, 'renderer #'+str(i), at=i)

interactive()
vp2.close()
vp1.close()
