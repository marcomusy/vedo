from vedo import makeLUT, Sphere

mesh = Sphere().lineWidth(0.1)

# create some data array to be associated to points
data = mesh.points()[:,2]
data[10:20] = float('nan')

# Build a lookup table of colors:
#               scalar  color   alpha
lut = makeLUT( [(-0.80, 'pink'       ),
                (-0.33, 'green',  0.8),
                ( 0.67, 'red'        ),
               ],
               vmin=-1, vmax=1,
               aboveColor='grey',
               belowColor='white',
               interpolate=False,
               )

mesh.cmap(lut, data).addScalarBar()

mesh.show(axes=1, viewup='z')
