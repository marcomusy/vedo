"""Create a button using an Image icon to show its state"""
from vedo import Cone, Image, dataurl, ButtonWidget, Plotter

def button_func(widget, evtname):
    print("button_func called")
    # The button state is 0/1, used here as color index and theme switch.
    cone.color(button.state)
    if button.state:
        plt.background("black")
    else:
        plt.background("white")

def on_mouse_click(event):
    # Update cone color only when something is clicked in the scene.
    if event.object:
        print("on_mouse_click", event)
        cone.color(button.state)

cone = Cone().color(0)

plt = Plotter(bg="w", axes=1)
plt.add_callback("mouse click", on_mouse_click)

plt.add(cone, __doc__)

# Button states use image icons.
img0 = Image(dataurl + "images/play-button.png")
img1 = Image(dataurl + "images/power-on.png")

button = ButtonWidget(
    button_func,
    states=[img0, img1],
    c=["red4", "blue4"],
    bc=("k9", "k5"),
    size=100,
    plotter=plt,
)
button.pos([0, 0]).enable()

plt.show(elevation=-40)
