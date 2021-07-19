#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from vedo import printc, Text2D, Text3D, show, settings, Line, Plotter, shapes
from vedo.settings import fonts
import numpy as np
import os


################################################################################## 2D
acts2d = []
for i, f in enumerate(fonts):
    t = Text2D(f+': The quick fox jumps over the lazy dog. 1234567890 αβγδεθλμνπστφψω',
               pos=(.015, 1-(i+3)*.06), font=f, s=1.3, c='k')
    acts2d.append(t)

acts2d.append(Text2D("List of Available Fonts", pos='top-center', bg='k', s=1.1))
plt0 = show(acts2d, bg2='cornsilk', axes=False, zoom=1.2, size=(1200,700), interactive=False)


################################################################################## 3D
# Symbols ~ ^ _ are reserved modifiers:
#  use ~ to add a short space, 1/4 of the default size,
#  use ^ and _ to start up/sub scripting, a space terminates them.
txt = """The quick fox jumps over the lazy dog.
Symbols: !@#$%&*()+=-{}[]:;|<>?/\euro1234567890\~
Units:  \delta=0.25E-03 ~μm, T_sea ~=~5.3~±0.7~\circC
LaTeX: \nabla\dotE=~4\pi~\rho, \nabla\timesE=~-1/c~~\partialB/\partialt
       ih~\partial/\partialt~\Psi = [-h^2 /2m\nabla^2  + V(r,t)]~\Psi(r,t)
       \DeltaE~=~h\nu, y = \Sigma_n ~A_n cos(\omega_n t+\delta_n ) sin(k_n x)
       \intx\dot~dx = \onehalf x\^2 + const.
       d^2 x^\mu  + \Gamma^\mu_\alpha\beta ~dx^\alpha ~dx^\beta  = 0
       -∇\^2u(x) = f(x) in Ω, u(x)~=~u_D (x) in \partial\Omega
Protect underscore \\\_ and \\\^ with a backslash.
"""

plt = Plotter(N=4, pos=(300,0), size=(1600,950))

cam = dict(pos=(3.99e+5, 8.51e+3, 6.47e+5),
           focalPoint=(2.46e+5, 1.16e+5, -9.24e+3),
           viewup=(-0.0591, 0.983, 0.175),
           distance=6.82e+5,
           clippingRange=(5.26e+5, 8.92e+5))

for i,fnt in enumerate(["Kanopus", "Normografo", "Theemim", "VictorMono"]):
    t = Text3D(txt, font=fnt, italic=0).c('darkblue').scale(12300)
    plt.show(t,
             Text2D("Font: "+fnt, font=fnt, bg='r'),
             axes=dict(xtitle='my units for L_x  (\mum)',
                       ytitle='my Y-axis with\na long description',
                       titleFont=fnt,
                       labelFont=fnt,
                       digits=2,
                      ),
             at=i,
             camera=cam,
             resetcam=not bool(i),
    )

################################################################################ printout
for font in fonts:
    printc(font + " - available characters are:", " "*25, bold=1, invert=1)
    fontfile = os.path.join(settings.fonts_path, font + '.npz')
    font_meshes = np.load(fontfile, allow_pickle=True)['font'][0]
    for k in font_meshes.keys():
        printc(k, end=' ')
    print()

printc('\n(use the above to copy&paste any char into your python script!)', italic=1)
printc('Symbols ~ ^ _ are reserved modifiers:', italic=1)
printc(' use ~ to add a short space, 1/4 of the default size,', italic=1)
printc(' use ^ and _ to start up/sub scripting, space terminates them.\n', italic=1)
printc('Supported LaTeX tags:', box='-', c='y')

for r in shapes._reps:
    print('\\'+repr(r[0]).replace("'","").replace("\\",""),' = ', r[1])
printc('Font Summary', c='g', box='-')
for i, f in enumerate(fonts):
    printc('Font: ', f, c='g')

################################################################################## 3D
cam = dict(pos=(55.8, -4.27, 107),
           focalPoint=(27.1, -29.2, -0.0532),
           viewup=(-0.0642, 0.976, -0.210),
           distance=113,
           clippingRange=(87.1, 147))

ln1 = Line([-1,-2],[52,-2], lw=0.1, c='grey')
fn3d=[ln1]
gap = 0
for i, font in enumerate(fonts):
    txt = font+": abcdefghijklmnopqrtuvwxyz 1234567890"
    if font in ["Theemim", "Kanopus", "Normografo", "VictorMono",
                "Galax", "LogoType", "Comae", "LionelOfParis"]:
         txt += "\n         αβγδεζηθκλμνξπρστφχψω ΔΘΛΞΠΣΦΨΩ"
         gap -= 2
    if font in ["VictorMono", "Kanopus", "LogoType", "Comae","LionelOfParis"]:
        txt+= " БГДЖЗИЙКЛ"
    gap -= 4
    t2 = Text3D(txt, font=font, italic=0).c(i).y(gap)
    ln = Line([-1,gap-1],[52,gap-1], lw=0.5, c='grey')
    fn3d.extend([t2,ln])

show(fn3d,
     new=True,
     pos=(400,100), size=(900,900), azimuth=20,
     axes=9,
     camera=cam,
     bg2='bb',
     bg='k',
    ).close()

plt.close()
plt0.close()
