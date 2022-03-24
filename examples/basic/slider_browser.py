"""Mouse hind limb growth from day 10 9h to day 15 21h"""
from vedo import settings, dataurl, load
from vedo import Text2D, Plotter, Picture, Axes, Line

def sliderfunc(widget, event):
    i = int(widget.GetRepresentation().GetValue())
    days = int((i*2+249)/24)
    time = f"{days}d {i*2+249-days*24}h"
    widget.GetRepresentation().SetTitleText(time)
    # remove the old and add the new shape
    # (no need to render as the slider makes a call to rendering)
    plt.pop().add(objs[i], render=False)

objs = load(dataurl+'timecourse1d.npy') # load a list of shapes

settings.defaultFont = "Glasgo"

plt = Plotter(bg='blackboard')
plt += Text2D(__doc__, pos="top-center", s=1.2, c='w')
plt += Picture(dataurl+'images/limbs_tc.jpg').scale(0.0154).y(10)
plt += Line([(0,8), (0,10), (28.6,10), (4.5,8)], c='gray')
plt += Axes(objs[-1])
plt += objs[0]
plt.addSlider2D(
    sliderfunc,
    0, len(objs)-1,
    pos=[(0.4,0.1), (0.9,0.1)], showValue=False, titleSize=1.5,
)
plt.show(zoom=1.2, mode='image')
plt.close()
