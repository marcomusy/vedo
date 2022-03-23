"""Mouse hind limb growth from day 10 9h to day 15 21h"""
from vedo import settings, dataurl, load
from vedo import Text2D, Plotter, Picture, Axes, Line

def sliderfunc(widget, event):
    global k
    knew = int(widget.GetRepresentation().GetValue())
    if k==knew:
        return
    objs[k].off()    #switch off old
    objs[knew].on()  #switch on new
    k = knew
    days = int((k*2+249)/24)
    t = f"{days}d {k*2+249 -days*24}h"
    widget.GetRepresentation().SetTitleText(t)

settings.defaultFont = "Glasgo"

objs = load(dataurl+'timecourse1d.npy') # load a list of shapes
# switch off them all except the first
for i in range(1, len(objs)):
    objs[i].lineWidth(3).off()
k = 0 # mesh index

# show the biggest and continue (return a Plotter instance)
plt = Plotter(bg='bb')
plt += objs
plt += Axes(objs[-1])
plt += Text2D(__doc__, pos="top-center", s=1.2, c='w')
plt += Picture(dataurl+'images/limbs_tc.jpg').scale(0.0154).y(10)
plt += Line([(0,8), (0,10), (28.6,10), (4.5,8)], c='gray')
plt.addSlider2D(
    sliderfunc,
    0, len(objs)-1,
    pos=[(0.4,0.1), (0.9,0.1)], showValue=False, titleSize=1.5,
)
plt.show(zoom=1.2, mode='image')
plt.close()
