"""Create a range slider to scale two spheres"""
from vedo import *


def slider1(w, e):
    if slid1.value > slid2.value:
        slid1.value = slid2.value
    s1.scale(slid1.value, reset=True)


def slider2(w, e):
    if slid2.value < slid1.value:
        slid2.value = slid1.value
    s2.scale(slid2.value, reset=True)


s1 = Sphere().c("red5").alpha(0.5).scale(0.8)
s2 = Sphere().c("green4").alpha(0.5).scale(1.2)

plt = Plotter()

slid2 = plt.add_slider(
    slider2,
    xmin=0.1,
    xmax=2,
    value=1.2,
    slider_length=0.02,
    slider_width=0.06,
    alpha=0.5,
    c="green4",
    show_value=True,
    font="Calco",
)

slid1 = plt.add_slider(
    slider1,
    xmin=0.1,
    xmax=2.0,
    value=0.8,
    slider_length=0.01,
    slider_width=0.05,
    alpha=0.5,
    tube_width=0.0015,
    c="red5",
    show_value=True,
    font="Calco",
)

plt.show(s1, s2, __doc__, axes=1).close()
