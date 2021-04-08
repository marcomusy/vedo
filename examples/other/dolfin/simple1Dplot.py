"""A simple 1D plot with axes customization"""
from dolfin import *
from vedo.dolfin import plot, screenshot

mesh = UnitIntervalMesh(100)
x = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)
u1 = project(cos(10*x[0]), V)
u2 = project(exp(x[0]), V)

class MyExpression(UserExpression):
    def __init__(self,u1,u2,**kwargs):
        self.u1 = u1
        self.u2 = u2
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = self.u1(x)/self.u2(x)
    def value_shape(self):
        return ()

f0 = MyExpression(u1, u2, degree=1)

plot( interpolate(f0,V),
      warpYfactor=0.5, # y-scaling factor to solution
      lc='navy',       # line color and width
      lw=3,
      xtitle="time [sec]",
      ytitle="velocity [a.u.]",
      axes={'xyGrid':True,
            'xyPlaneColor':'blue',
            'xyGridColor':'peru',
            'xyAlpha':0.1,
            'yHighlightZero':True,
           },
      scalarbar=False,
      zoom=1.1,
    )

#screenshot('pic.png') # uncomment to take a screenshot