"""Transform a Mesh by defining how a specific
set of points (landmarks) must move"""
from vedo import *

s1 = Mesh(dataurl + "bunny.obj").c("gold").flag('bunny1')

# Make a copy of s1 (note that this mesh is not really necessary
# we only used it to click a bunch of points as landmarks,
# moreover landmark points do not need to belong to any mesh!)
s2  = s1.clone().scale([1,1.5,1]).pos(.2,0,0)
s2.color('t').wireframe().flag('bunny2')

landmarksFrom = [
    [-0.067332, 0.177376, -0.05199058],
    [-0.004541, 0.085447,  0.05713107],
    [-0.011799, 0.175825, -0.02279279],
    [-0.081910, 0.117902,  0.04889364],
]

landmarksTo = [
    [0.1287002, 0.2651531, -0.0469673],
    [0.1948514, 0.1285412,  0.0571203],
    [0.1860555, 0.2626522, -0.0202493],
    [0.1149052, 0.1731894,  0.0474256],
]

s3 = s1.clone().transformWithLandmarks(landmarksFrom, landmarksTo)
s3.flag('transformed bunny')

ars = Arrows(landmarksFrom, landmarksTo, s=0.5).c('k').alpha(0.5)

show(s1,s2,s3, ars, __doc__, axes=True).close()
