from vedo import settings, Volume, Text, show, datadir

settings.fontIsMono=False
settings.fontLSpacing=0.2
settings.fontHSpacing=0.5

vol = Volume(datadir+"embryo.slc")
vol.mode(1).c('b2')
# vol.mode(1).c(['dr','dr','dr','dr', 'dg', 'dg', 'db', 'db', 'db'])
# vol.mode(0).c('b2')

t = Text("Sharpe\n~~~Lab", s=40, font="./Spears.npz", vspacing=1.4, depth=.04)
t.c('k1').rotateX(90).pos(200,150,70)

cam = dict(pos=(363, -247, 121),
           focalPoint=(240, 137, 116),
           viewup=(4.45e-3, 0.0135, 1.00),
           distance=403,
           clippingRange=(36.0, 874))

show(vol, t, size=(700,400),
     camera=cam,
     # elevation=-89, zoom=2,
)

