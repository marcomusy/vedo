#!/usr/bin/python3
#
"""Build 2 windows that can interact and share functions"""
from vedo import *


plt1 = Plotter(title='Window 1', sharecam=False, shape=(8,2))
plt2 = Plotter(title='Window 2', size=(700,700), pos=(400,0))

####################################################################################
def keyfunc(evt):
    printc('keyfunc called, pressed key:', evt.keyPressed)

    if evt.keyPressed=='c':
        i = plt1.renderers.index(plt1.renderer)
        if i>= len(shapes): return
        shapes[i].color('red')
        plt1.render()
        plt2.render()
        onLeftClick(None)
    elif evt.keyPressed=='h':
        plt2.add(Text2D(instr, pos='bottom-right', c='dg', bg='g', font='Quikhand'))

####################################################################################
def onLeftClick(evt):
    if not evt: return
    mesh = evt.actor
    i = plt1.renderers.index(plt1.renderer)
    if i>= len(shapes): return
    printc('onLeftClick called!', c='y')

    ishape = shapes[i]
    sname = Text2D('This is called: '+ishape.name,
                   pos='top-center', c='r', bg='y', font='Calco')
    sprop = Text2D('color = '+getColorName(ishape.color()),
                   pos='bottom-left', c=ishape.color(), bg='k', font='Calco')
    instt = Text2D(instr, pos='bottom-right', c='dg', bg='g', font='Quikhand')

    axes = ishape.buildAxes(yzGrid=False)
    acts = [ishape, axes, sname, sprop, instt]

    plt1.background('silver').render()
    plt2.clear()
    plt2.add(acts, resetcam=True)

####################################################################################
plt1.addCallback("KeyPress", keyfunc)
plt1.addCallback("LeftButtonPress", onLeftClick)
plt1.at(len(plt1.renderers)-1).show(
    Picture(dataurl+'images/embl_logo.jpg').rotateX(-20),
    Text2D('Some footnote', pos='bottom-right', font='Quikhand', c='grey', s=0.6),
)

shapes = []
for i in range(15):
    ps = ParametricShape(i).color(i-9)
    pname = Text2D(ps.name, bg='b', s=0.7, font='Calco')
    plt1.at(i).show(ps, pname)
    shapes.append(ps)

####################################################################################
instr  = "Click on the left panel to select a shape\n"
instr += "Press c to make the shape red\n"
plt2.addCallback('KeyPress', keyfunc)
plt2.show(
    __doc__,
    VedoLogo(distance=10),
    Text2D("My Multi Viewer 1.0", pos=(.5,.8), s=2.5, c='dg', font='Kanopus', justify='center'),
    Text2D(instr, bg='g', pos=(0.5,0.1), s=1.2, font='Quikhand', justify='center'),
)
plt2.close()
plt1.close()
