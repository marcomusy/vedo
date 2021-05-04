"""Mouse hind limb growth from day 10 9h to day 15 21h"""
from vedo import *

settings.defaultFont = "SmartCouric"

objs = load(dataurl+'timecourse1d.npy') # list of meshes

# show the biggest and continue (return a Plotter instance)
plt = show(objs[-1], axes=1, interactive=False, bg='bb')
plt.actors = objs # set Plotter internal list of objs to be shown

# switch off all but the first
for i in range(1, len(objs)):
    objs[i].c('gold').lineWidth(2.0).off()

k = 0 # visible mesh index
def sliderfunc(widget, event):
    global k
    knew = int(widget.GetRepresentation().GetValue())
    if k==knew: return
    plt.actors[k].off()    #switch off old
    plt.actors[knew].on()  #switch on new
    k = knew
    days = int((k+249)/24)
    hours = ' %2sh (' % (k+249 -days*24)
    limbage = str(days)+ "d"+ hours + str(k+249)+"h)"
    widget.GetRepresentation().SetTitleText(limbage)

plt.addSlider2D(sliderfunc, k, len(objs)-1,
                pos=[(0.4,0.1), (0.9,0.1)], showValue=False, titleSize=1.5)

plt += Text2D(__doc__, pos="top-center", s=1., c='w')
plt += load(dataurl+'images/limbs_tc.jpg').scale(0.0154).y(10.0)
plt += Line([(0,8), (0,10), (28.6,10), (4.5,8)], c='gray')

plt.show(zoom=1.2, interactive=True).close()
