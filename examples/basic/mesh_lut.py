"""Build a custom colormap, including
out-of-range and NaN colors and labels"""
from vedo import build_lut, Sphere, show

# Generate a sphere and stretch it, so it sits between z=-2 and z=+2
mesh = Sphere(quads=True).scale([1,1,1.8]).linewidth(1)

# Create some dummy data array to be associated to points
data = mesh.vertices[:,2].copy()  # pick z-coords, use them as scalar data
data[10:70] = float('nan') # make some values invalid by setting to NaN
data[300:600] = 100        # send some values very far above-scale

# Build a custom Look-Up-Table of colors:
#     value, color, alpha
lut = build_lut(
    [
      #(-2, 'pink'      ),  # up to -2 is pink
      (0.0, 'pink'      ),  # up to 0 is pink
      (0.4, 'green', 0.5),  # up to 0.4 is green with alpha=0.5
      (0.7, 'darkblue'  ),
      #( 2, 'darkblue'  ),
    ],
    vmin=-1.2,
    vmax= 0.7,
    below_color='lightblue',
    above_color='grey',
    nan_color='red',
    interpolate=False,
)

# 3D scalarbar:
mesh.cmap(lut, data).add_scalarbar3d(title='My Scalarbar', c='white')
# mesh.scalarbar.scale(1.5).rotate_x(90).shift(0,2) # make it bigger and place it2)

# OR 2D scalarbar:
# mesh.cmap(lut, data).add_scalarbar()

# OR 2D scalarbar derived from the 3D one:
mesh.scalarbar = mesh.scalarbar.clone2d(pos=[0.7, -0.95], size=0.2)

show(mesh,
     __doc__,
     axes=dict(zlabel_size=.04, number_of_divisions=10),
     elevation=-80, 
     bg='blackboard',
).close()
