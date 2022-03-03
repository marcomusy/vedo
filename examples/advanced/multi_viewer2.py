from vedo import settings, Plotter, ParametricShape, VedoLogo, Text2D

settings.rendererFrameWidth = 1

##############################################################################
def onLeftClick(evt):
    if not evt.actor: return
    shapename.text(f'This is called: {evt.actor.name}, on renderer nr.{evt.at}')
    plt.at(1).remove(actsonshow).add(evt.actor, resetcam=True)
    actsonshow.clear()
    actsonshow.append(evt.actor)

##############################################################################
sy, sx, dx = 0.12, 0.12, 0.01
# Define the renderers rectangle areas
# to help finding bottomleft&topright corners check out utils.gridcorners()
shape = [
    dict(bottomleft=(0,0), topright=(1,1), bg='k7'), # the full empty window
    dict(bottomleft=(dx*2+sx,0.01), topright=(1-dx,1-dx), bg='w'), # the display window
    dict(bottomleft=(dx,sy*1), topright=(dx+sx,sy*2), bg='k8', bg2='lb'), # CrossCap
    dict(bottomleft=(dx,sy*2), topright=(dx+sx,sy*3), bg='k8', bg2='lb'),
    dict(bottomleft=(dx,sy*3), topright=(dx+sx,sy*4), bg='k8', bg2='lb'),
    dict(bottomleft=(dx,sy*4), topright=(dx+sx,sy*5), bg='k8', bg2='lb'),
    dict(bottomleft=(dx,sy*5), topright=(dx+sx,sy*6), bg='k8', bg2='lb'),
    dict(bottomleft=(dx,sy*6), topright=(dx+sx,sy*7), bg='k8', bg2='lb'),
    dict(bottomleft=(dx,sy*7), topright=(dx+sx,sy*8), bg='k8', bg2='lb'), # RandomHills
]

plt = Plotter(shape=shape, sharecam=False, size=(1050, 980))
plt.addCallback("when i click my mouse button please call", onLeftClick)

for i in range(2,9):
    ps = ParametricShape(i).color(i)
    pname = Text2D(ps.name, c='k', bg='blue', s=0.7, font='Calco')
    plt.at(i).show(ps, pname)

shapename = Text2D(pos='top-center', c='r', bg='y', font='Calco') # empty text

vlogo = VedoLogo(distance=5)
actsonshow = [vlogo]

title = "My Multi Viewer 1.0"
instr = "Click on the left panel to select a shape\n"
instr+= "Press h to print the full list of options"

plt.at(1).show(
    vlogo, shapename,
    Text2D(title, pos=(0.5,0.85), s=2.5, c='dg', font='Kanopus', justify='center'),
    Text2D(instr, bg='g', pos=(0.5,0.05), s=1.2, font='Quikhand', justify='center'),
)
plt.interactive().close()
