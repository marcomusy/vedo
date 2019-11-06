from vtkplotter import makeLUT, Sphere

mesh = Sphere().lineWidth(0.1)

# create some data array to be associated to points
data = mesh.getPoints()[:,2]
data[10:20] = float('nan')

# Build a lookup table of colors:
#               scalar  color   alpha
lut1 = makeLUT([(-0.80, 'pink'       ),
                (-0.33, 'green',  0.8),
                ( 0.67, 'red'        ),
               ],
               vmin=-1, vmax=1,
               aboveColor='grey',
               belowColor='white',
               interpolate=False,
               )

mesh.pointColors(data, cmap=lut1).addScalarBar()

#Avoid interpolating cell colors before mapping:
#mesh.mapper.InterpolateScalarsBeforeMappingOff()

mesh.show(bg='white', axes=1, viewup='z')
