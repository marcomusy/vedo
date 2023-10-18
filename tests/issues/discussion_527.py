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

plt.show(s0, "renderer0", at=0)
plt.show(s1, "renderer1", at=1)


def slider1(widget, event):
    value = slider_rep.GetValue()
    s0.rotate_y(value)
    s1.rotate_y(-value)


plt.renderer = plt.renderers[2]  # make it the current renderer
slider = plt.add_slider(slider1, -5, 5, value=0, pos=([0.05, 0.02], [0.55, 0.02]))
slider_rep = slider.GetRepresentation()
vscale = 20
slider_rep.SetSliderLength(0.003 * vscale)  # make it thicker
slider_rep.SetSliderWidth(0.025 * vscale)
slider_rep.SetEndCapLength(0.001 * vscale)
slider_rep.SetEndCapWidth(0.025 * vscale)
slider_rep.SetTubeWidth(0.0075 * vscale)

plt.interactive().close()