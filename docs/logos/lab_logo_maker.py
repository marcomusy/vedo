from vedo import Volume, Text3D, show, dataurl

vol = Volume(dataurl+"embryo.slc").mode(0).c('b2').alphaUnit(5)

t = Text3D("Sharpe\n~~~Lab", s=40, font="Spears", vspacing=1.4, depth=.04)
t.c('k1').rotateX(90).pos(200,150,70)

cam = dict(pos=(363, -247, 121),
           focalPoint=(240, 137, 116),
           viewup=(4.45e-3, 0.0135, 1.00),
           distance=403)

show(vol, t, size=(700,400), camera=cam)

