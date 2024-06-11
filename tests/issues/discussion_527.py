from vedo import *

settings.immediate_rendering = False  # faster for multi-renderers

# (0,0) is the bottom-left corner of the window, (1,1) the top-right
# the order in the list defines the priority when overlapping
custom_shape = [
    dict(bottomleft=(0.0, 0.0), topright=(0.5, 1), bg="wheat", bg2="w"),  # ren0
    dict(bottomleft=(0.5, 0.0), topright=(1, 1), bg="blue3", bg2="lb"),  # ren1
    dict(bottomleft=(0.2, 0.05), topright=(0.8, 0.1), bg="white"),  # ren2
]
plt = Plotter(shape=custom_shape, size=(1600, 900))

s0 = ParametricShape(0)
s1 = ParametricShape(1)

plt.at(0).add(s0, "renderer0")
plt.at(1).add(s1, "renderer1")


def slider1(widget, event):
    value = widget.value
    s0.rotate_y(value)
    s1.rotate_y(-value)

opts = dict(
    slider_length=0.06,
    slider_width=0.6,
    end_cap_length=0.02,
    end_cap_width=0.5,
    tube_width=0.1,
    title_height=0.15,
)
plt.at(2).add_slider(slider1, -5, 5, value=0, pos=([0.05, 0.02], [0.55, 0.02]), **opts)

plt.show(interactive=True).close()