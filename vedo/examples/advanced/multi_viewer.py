"""Build 2 windows that can interact and share functions"""
from vedo import *


plt1 = Plotter(title='Window 1', sharecam=False, shape=(8,2))
plt2 = Plotter(title='Window 2', size=(700,700), pos=(250,0))

####################################################################################
def keyfunc(key):
    printc('keyfunc called, pressed key:', key)

    if key=='c':
        i = plt1.renderers.index(plt1.renderer)
        if i>= len(shapes): return
        shapes[i].color('red')
        plt1.interactor.Render()
        plt2.interactor.Render()
        onLeftClick(None)
    elif key=='x':
        for r in plt1.renderers: r.SetBackground(1,1,1) # white
        plt1.interactor.Render()
    elif key=='h':
        plt2.add(Text2D(instr, pos='bottom-right', c='dg', bg='g', font='Godsway'))
    elif key=='r':
        plt2.renderer.ResetCamera()
        plt2.interactor.Render()

####################################################################################
def onLeftClick(mesh):
    i = plt1.renderers.index(plt1.renderer)
    if i>= len(shapes): return
    printc('onLeftClick called!', c='y')

    ishape = shapes[i]
    sname = Text2D('This is called: '+ishape.name,
                   pos='top-center', c='r', bg='y', font='CallingCode')
    sprop = Text2D('color = '+getColorName(ishape.color()),
                   pos='bottom-left', c=ishape.color(), bg='k', font='ChineseRuler')
    instt = Text2D(instr, pos='bottom-right', c='dg', bg='g', font='Godsway')

    axes = addons.buildAxes(ishape, yzGrid=False)
    acts = [ishape, axes, sname, sprop, instt]

    plt1.renderer.SetBackground(0.7,0.7,0.7) # grey
    plt1.interactor.Render()
    plt2.clear()
    plt2.add(acts)
    plt2.resetcam = False

####################################################################################
plt1.keyPressFunction = keyfunc
plt1.mouseLeftClickFunction = onLeftClick
plt1.show(Picture(datadir+'images/embl_logo.jpg').rotateX(-20),
          Text2D('Some footnote', pos='bottom-right', font='Godsway', c='grey', s=0.6),
          at=len(plt1.renderers)-1)

shapes = []
for i in range(15):
    ps = ParametricShape(i).color(i-9)
    pname = Text2D(ps.name, pos=1, bg='b', s=0.9, font='CallingCode')
    plt1.show(ps, pname, at=i)
    shapes.append(ps)

####################################################################################
instr  = "Click on the left panel to select a shape\n"
instr += "Press c to make the shape red\n"
instr += "Press x to reset to white the panel background"
plt2.keyPressFunction = keyfunc
plt2.show(__doc__,
          Picture(datadir+'images/vedo_small.png'), Point([175,50,1200], alpha=0),
          Text2D("My Multi Viewer 1.0",
                 pos=(.5,.8), s=2, c='dg', font='ImpactLabel', justify='center'),
          Text2D(instr, bg='g', pos=(0.5,0.2), s=1.2, font='Godsway', justify='center'),
          resetcam=True)

