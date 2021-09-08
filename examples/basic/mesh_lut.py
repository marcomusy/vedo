"""Build a custom colormap, including
out-of-range and NaN colors and labels"""
from vedo import buildLUT, Sphere, show, settings

settings.useDepthPeeling = True # might help with transparencies

# generate a sphere and stretch it, so it sits between z=-2 and z=+2
mesh = Sphere(quads=True).scale([1,1,2]).lineWidth(0.1)

# create some dummy data array to be associated to points
data = mesh.points()[:,2]  # pick z-coords, use them as scalar data
data[10:70] = float('nan') # make some values invalid by setting to NaN
data[300:600] = 100        # send some values very far above-scale

# build a custom LookUp Table of colors:
#               value, color, alpha
lut = buildLUT([
                #(-2, 'pink'      ),  # up to -2 is pink
                (0.0, 'pink'      ),  # up to 0 is pink
                (0.4, 'green', 0.5),  # up to 0.4 is green with alpha=0.5
                (0.7, 'darkblue'  ),
                #( 2, 'darkblue'  ),
               ],
               vmin=-1.2, belowColor='lightblue',
               vmax= 0.7, aboveColor='grey',
               nanColor='red',
               interpolate=False,
              )
# 3D scalarbar:
mesh.cmap(lut, data).addScalarBar3D(title='My 3D scalarbar', c='white')
mesh.scalarbar.scale(1.5).rotateX(90).y(1) # make it bigger and place it

# 2D scalarbar:
# mesh.cmap(lut, data).addScalarBar()

show(mesh, __doc__,
     axes=dict(zLabelSize=.04, numberOfDivisions=10),
     elevation=-80, bg='blackboard',
).close()
