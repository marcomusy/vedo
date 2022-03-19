#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from vedo import printc, Text2D, Text3D, show, Plotter, shapes
from vedo import fonts, fonts_path, settings
import numpy as np
import os

settings.allowInteraction = True

################################################################################## 2D
inred = Text2D("°monospaced fonts are marked in red", c='r5', pos='bottom-center', font='VictorMono')
acts2d = [inred]

txt = 'The quick fox jumps over the lazy dog. 1234567890 αβγδεθλμνπστφψω'
for i, f in enumerate(fonts):
    bg = None
    if f in ['Calco', 'Glasgo', 'SmartCouric', 'VictorMono']:
        bg = 'red5'
    t = Text2D(f'{f}: {txt}', pos=(.015, 1-(i+3)*.06), font=f, s=1.3, c='k', bg=bg)
    acts2d.append(t)

acts2d.append(Text2D("List of built-in fonts", pos='top-center', bg='k', s=1.3))
plt0a = show(acts2d, bg2='cornsilk', size=(1300,800), interactive=False)

## online fonts:
acts2d = []
i = 0
for key, props in sorted(settings.font_parameters.items()):
    if props['islocal']:
        continue
    if key=='Justino2' or key=='Justino3':
        continue
    bg = None
    if props['mono']:
        bg = 'red5'
    t = Text2D(f'{key}: {txt}', pos=(.015, 1-(i+2)*.06), font=key, s=1.3, c='k', bg=bg)
    acts2d.append(t)
    i+=1
plt0b = show(acts2d,
             Text2D("Additional fonts (https://vedo.embl.es/fonts)", pos='top-center', bg='k', s=1.3),
             bg2='lb', size=(1300,900), pos=(1200,200), new=True)

################################################################################ printout
for font in fonts:
    printc(font + " - available characters:", " "*25, bold=1, invert=1)
    fontfile = os.path.join(fonts_path, font + '.npz')
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
    plt.at(i)
    plt.show(t,
             Text2D("Font: "+fnt, font=fnt, bg='r'),
             axes=dict(xtitle='my units for L_x  (\mum)',
                       ytitle='my Y-axis with\na long description',
                       titleFont=fnt,
                       labelFont=fnt,
                       digits=2,
                      ),
             camera=cam,
             resetcam=not bool(i),
    )
plt.interactive().close()

plt0b.close()
plt0a.close()
