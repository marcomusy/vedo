"""Mouse hind limb growth from day 10 9h to day 15 21h"""
from vedo import settings, dataurl, Assembly
from vedo import Text2D, Plotter, Image, Axes, Line


def sliderfunc(widget, event):
    i = int(widget.value)
    days = int((i * 2 + 249) / 24)
    widget.title = f"{days}d {i*2+249-days*24}h"
    # remove the old and add the new shape
    # (no need to render as the slider makes a call to rendering)
    plt.pop().add(objs[i])


objs = Assembly(dataurl+"timecourse1d.npy")  # load a list of shapes

settings.default_font = "Glasgo"

plt = Plotter(bg="blackboard")
plt += Text2D(__doc__, pos="top-center", s=1.2).color("w")
plt += Image(dataurl + "images/limbs_tc.jpg").scale(0.0154).y(10)
plt += Line([(0, 8), (0, 10), (28.6, 10), (4.5, 8)]).color("gray")
plt += Axes(objs[-1])
plt += objs[0]
plt.add_slider(
    sliderfunc,
    0,
    len(objs) - 1,
    pos=[(0.4, 0.1), (0.9, 0.1)],
    show_value=False,
    title_size=1.5,
)
plt.show(zoom=1.2, mode="image")
plt.close()
