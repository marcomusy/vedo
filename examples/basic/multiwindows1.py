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
plt1 = Plotter(shape=(5,5), axes=0)

# set a different background color for a specific subwindow (the last one)
plt1.renderers[24].SetBackground(0.8, 0.9, 0.9)  # use vtk method SetBackground()

# load the meshes and give them a name
a = plt1.load(dataurl+"shuttle.obj")
b = plt1.load(dataurl+"cessna.vtk").c("red")
c = plt1.load(dataurl+"porsche.ply")

# show a Text2D in each renderer
for i in range(25):
    plt1.at(i).show(f"renderer\nnr.{i}")

plt1.at( 6).show(a)
plt1.at(23).show(b)
plt1.at(24).show(c)


##########################################################################
# declare a second independent instance of the class Plotter
# shape can also be given as a string, e.g.:
# shape="2/6" means 2 renderers above and 6 below
# shape="3|1" means 3 renderers on the left and one on the right

s = Mesh(dataurl+'mug.ply')

# Set the position of the horizontal of vertical splitting [0,1]:
#settings.windowSplittingPosition = 0.5

plt2 = Plotter(pos=(500, 250), shape='2/6')

for i in range(len(plt2.renderers)):
    s2 = s.clone(deep=False).color(i)
    plt2.at(i).show(s2, f'renderer #{i}')

plt2.interactive()
plt2.close()
plt1.close()
