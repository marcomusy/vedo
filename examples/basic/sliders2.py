"""Sliders and buttons controlling objects"""
from vedo import *

settings.use_depth_peeling = True

def slider0(widget, event):
    sphere.color(widget.value)

def slider1(widget, event):
    val = widget.value
    widget.title = get_color_name(val)
    cube.color(val)

def button_func(obj, event):
    cube.alpha(1 - cube.alpha()) # toggle mesh transparency
    sphere.alpha(1 - sphere.alpha())
    button.switch()              # change to next status

######
sphere = Sphere(r=0.6).lw(1).color(0).alpha(0.8)
cube = Cube().lw(1).color(0).alpha(0.8)

plt = Plotter(N=2, axes=True)

######
plt.at(0).show(sphere, __doc__)  # show the sphere on the first renderer
plt.add_slider(
    slider0,
    -9, 9,           # slider range
    value=0,         # initial value
    pos=([0.1,0.1],  # first point of slider in the renderer
         [0.4,0.1]), # 0.4 = 40% of the window size width
    title="slider nr.0, color number",
)

######
plt.at(1).show(cube)
plt.add_slider(
    slider1,
    -9, 9,
    value=0,
    pos=([0.1,0.1], [0.4,0.1]),
    title="slider nr.1, color number",
)

######
button = plt.at(1).add_button(
    button_func,
    pos=(0.5, 0.95),      # x,y fraction from bottom left corner
    states=["HIGH alpha (click here!)", "LOW alpha (click here!)"],
    c = ["w", "k"],       # colors of states (foreground)
    bc= ["k", "grey"],    # colors of states (background)
    font="Quikhand",
    size=35,
)

plt.show().interactive().close()
