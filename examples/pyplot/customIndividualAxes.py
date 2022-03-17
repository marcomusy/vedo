"""Create individual axes to each separate object in a scene.
Access any element to change its size and color"""
from vedo import *

settings.useDepthPeeling = True

# Create a bunch of objects
s1 = Sphere(pos=(10, 0, 0), r=1, c='r')
s2 = Sphere(pos=( 0,10, 0), r=2, c='g')
s3 = Sphere(pos=( 0, 0,10), r=3, c='b')
pt = Point([-4,-4,-4], c='k')

# Build individual axes for each object.
#  A new Assembly object is returned:
axes1 = Axes(s1, c='r')
axes2 = Axes(s2, c='g')
axes3 = Axes(s3, c='b', numberOfDivisions=10)

# axes3 is an Assembly (group of Meshes).
# Unpack it and scale the 7th label getting it by its name,
# make it 5 times bigger big and fuchsia:
axes3.unpack('xNumericLabel7').scale(5).c('fuchsia')
# Print all element names in axes3:
#for m in axes3.getMeshes(): print(m.name)

# By specifiyng axes in show(), new axes are
#  created which span the whole bounding box.
#  Options are passed through a dictionary
show(pt, s1, s2, s3, axes1, axes2, axes3, __doc__,
     viewup='z',
     axes=dict(c='black',
               numberOfDivisions=10,
               yzGrid=False,
              ),
).close()
