"""Sliders and buttons controlling objects"""
from vedo import *


def slider0(widget, event):
    value = widget.GetRepresentation().GetValue()+0.5
    sphere.color(value)

def slider1(widget, event):
    rep = widget.GetRepresentation()
    value = rep.GetValue()+0.5
    rep.SetTitleText(getColorName(value))
    cube.color(value)

def buttonfunc():
    cube.alpha(1 - cube.alpha()) # toggle mesh transparency
    sphere.alpha(1 - sphere.alpha())
    button.switch()              # change to next status

######
plt = Plotter(N=2, axes=True)

######
sphere = Sphere(r=0.6).alpha(0.9).color(0)
plt.show(sphere, __doc__, at=0)  # show the sphere on the first renderer
plt.addSlider2D(slider0,
               -9, 9,           # slider range
               value=0,         # initial value
               pos=([0.1,0.1],  # first point of slider in the renderer
                    [0.4,0.1]), # 0.4 = 40% of the window size width
               title="slider 0, color number")

######
cube = Cube().alpha(0.9).color(0)
plt.show(cube, at=1)
plt.addSlider2D(slider1,
               -9, 9,
               value=0,
               pos=([0.1,0.1],
                    [0.4,0.1]),
               title="slider 1, color number")

######
button = plt.addButton(buttonfunc,
    pos=(0.5, 0.9),       # x,y fraction from bottom left corner
    states=["HIGH alpha (click here!)", "LOW alpha (click here!)"],
    c = ["w", "k"],       # colors of states (foreground)
    bc= ["k", "grey"],    # colors of states (background)
    font="Quikhand",
    size=35,
)

plt.show(interactive=1).close()
