"""
 Mouse hind limb growth from day 10 9h to day 15 21h
"""
from vtkplotter import *

objs = load(datadir+'timecourse1d.npy') # list of meshes

# show the biggest and continue (return a Plotter instance)
vp = show(objs[-1], axes=1, interactive=False, bg='bb')
vp.actors = objs # set Plotter internal list of objs to be shown

# switch off all the others
for i in range(1, len(objs)): objs[i].c('gold').lw(2.0).off()

k = 0 # visible mesh index
def sliderfunc(widget, event):
    global k
    vp.actors[k].off() #switch off
    k = int(widget.GetRepresentation().GetValue())
    vp.actors[k].on()  #switch on
    days = int((k+249)/24)
    hours = ' %2sh (' % (k+249 -days*24)
    limbage = str(days)+ "d"+ hours + str(k+249)+"h)"
    widget.GetRepresentation().SetTitleText(limbage)
    vp.renderer.ResetCamera()

vp.addSlider2D(sliderfunc, 0, len(objs)-1,
               pos=[(0.4,0.1), (0.9,0.1)], showValue=False)

vp += Text2D(__doc__, font='SpecialElite', s=1.2, c='w')
vp += load(datadir+'images/limbs_tc.jpg').scale(0.0154).y(10.0)
vp += Line([(0,8), (0,10), (28.6,10), (4.5,8)], c='gray')

vp.show(zoom=1.2, interactive=True)
