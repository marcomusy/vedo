from vedo import *
from vedo.settings import fonts
import numpy as np

################################################################################ printout
for font in fonts:
    printc(font + " - available characters are:", " "*25, bold=1, invert=1)
    try:
        fontfile = settings.fonts_path + font + '.npz'
        font_meshes = np.load(fontfile, allow_pickle=True)['font'][0]
    except:
        pass
    for k in font_meshes.keys(): printc(k, end=' ')
    printc('\n(use the above to copy&paste any char into your python script!)', italic=1)
    printc('Symbols ~ ^ _ are reserved modifiers:', italic=1)
    printc(' use ~ to add a short space, 1/4 of the default size,', italic=1)
    printc(' use ^ and _ to start up/sub scripting, space terminates them.\n', italic=1)

################################################################################## 2D
fonts = ['Arial', 'Courier'] + fonts + ['Times']

for i, f in enumerate(fonts):
    printc('Font: ', f)
    Text2D(f+':  The quick fox jumps over the lazy dog. 1234567890',
           pos=(.01, 1-(i+2)*.055), font=f, s=1.5, c='k')

printc('More fonts available at:')
printc('\thttps://www.1001freefonts.com', underline=1)
printc('\t(use them by setting font=filename.ttf)')

Text2D("List of 2D available fonts", pos='top-center', bg='k', s=1.1)
show(..., bg2='cornsilk', axes=False, zoom=1.2, size=(1200,900), interactive=False)

################################################################################## 3D
#font = 'VTK' # the default
#font = 'BPmonoBold'
font = 'BPmonoItalics'
#font = 'BPmonoRegular'
#font = 'CallingCode'
#font = 'ChineseRuler'
#font = 'ClassCoder'
#font = 'MonospaceTypewriter'
#font = 'Montserrat'
#font = 'Quikhand'

# Symbols ~ ^ _ are reserved modifiers:
#  use ~ to add a short space, 1/4 of the default size,
#  use ^ and _ to start up/sub scripting, a space terminates them.

txt = """The quick fox jumps over the lazy dog.
A second line of text with line break\nNumbers: 1234567890
Symbols: !@#$%&*()+=-{}[]:;|<>?/|
Physics/Maths: ψ=0.25E-03 ~mμ, T_sea ~=~5.3~±0.7~°C
a~=~g^μ_ξi ~G_J^norm F^J_lmn  N_A =6.023e+23  f(θ,φ) → θ·e^cos~φ"""

t = Text(txt, font=font).c('darkblue').bc('tomato').scale(12300)

show(t, "3D polygonal Text demo",
     newPlotter=True,
     pos=(900,0), size=(1000,500), zoom=2, azimuth=20,
     axes=dict(xtitle='my units in μm·10^-3',
               ytitle='my Y-axis with\na long description',
               titleFont=font,
               yTitleSize=0.02,
               xTitleOffset=0.1,
               labelFont='Quikhand',
               digits=2,
               )
     )
