"""Setting illumination properties:
ambient, diffuse
specular, specularPower, specularColor.
"""
#https://lorensen.github.io/VTKExamples/site/Python/Rendering/SpecularSpheres
from vedo import Plotter, Arrow, Light, dataurl


vp = Plotter(axes=1)

ambient, diffuse, specular = 0.1, 0., 0.
specularPower, specularColor= 20, 'white'

for i in range(8):
    s = vp.load(dataurl+'apple.ply').c('gold')
    s.normalize().pos((i%4)*2.2, int(i<4)*3, 0)

    #s.phong()
    s.flat()

    # modify the default with specific values
    s.lighting('default', ambient, diffuse, specular, specularPower, specularColor)
    #ambient += 0.125
    diffuse += 0.125
    specular += 0.125

vp += __doc__
vp.show()

print('Adding a light source..')
p = (3, 1.5, 3)
f = (3, 1.5, 0)
vp += [Arrow(p,f, s=0.01, c='gray', alpha=0.2), Light(pos=p, focalPoint=f)]
vp.show().close()
