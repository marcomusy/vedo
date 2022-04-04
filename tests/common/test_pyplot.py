#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from vedo import shapes, show, dataurl, settings
from vedo import Picture, Mesh, Points, Point
from vedo.pyplot import Figure, donut


settings.useParallelProjection = True

fig = Figure([-1,12], [-2,14], aspect=16/9, padding=0,
    title="Lorem Ipsum Neque porro quisquam",
    xtitle="test x-axis should always align",
    ytitle="y-axis (zeros should match)",
    grid=True,
)
print(f"yscale = {fig.yscale}")

man = Mesh(dataurl+'man.vtk').scale(1.4).pos(7,4).rotateX(-90, around='itself')
fig += man

pic = Picture("https://vedo.embl.es/examples/data/textures/bricks.jpg")
fig += pic.scale(0.005).pos(2,10)

fig += Points([[8,1],[10,3]], r=15)

fig += donut([0.1, 0.2, 0.3, 0.1, 0.3], c=[1,2,3,4,'w']).scale(1).pos(1,6,.2)

fig += Point([2,3])
fig += Point([4,5])
fig += shapes.Circle([4,5])
fig += shapes.Circle([0,0], r=3)
fig += shapes.Circle([0,12], r=3).c("r6")
fig += shapes.Circle([11,12], r=3).c("p5")
fig += shapes.Circle([11,0], r=3).c("o5")

fig += shapes.Arrow([2,3], [4,5]).z(.05)
fig += shapes.Line( [2,3], [4,5]).z(.1).lw(2)

fig += shapes.Line([2,2], [4,4], c='k', lw=6).z(.1)

fig += shapes.DashedLine([8,3],[10,5], spacing=0.5, c='r')
fig += shapes.Tube([[8,2,0],[10,4,0]], r=.1).lighting('ambient')

fig+= shapes.Marker('.').pos(5,5).scale(12)
fig+= shapes.Star3D().pos(5,7).scale(0.5)
fig+= shapes.Cross3D().pos(5,3).scale(0.5)

fig += shapes.Glyph([[5,9]], shapes.Sphere(r=0.5))

fig += shapes.Spline([[4,0],[5,2],[6,0],[7,0.5]]).c('r4')
fig += shapes.CSpline([[4,0],[5,2],[6,0],[7,0.5]]).c('r6')
fig += shapes.KSpline([[4,0],[5,2],[6,0],[7,0.5]]).c('r8')
fig += shapes.Bezier([[4,-1],[5,1],[6,-1],[7,-1.5]])

fig += shapes.Brace([2,1], [4,3],comment='Brace', pad1=0, italic=3).z(0.1) ## BUGGED

fig+= shapes.Ribbon(shapes.Spline([[4,0],[5,2],[6,0],[7,0.5]]),
                    shapes.Bezier([[4,-1],[5,1],[6,-1],[7,-1.5]]))

fig+= shapes.Star([8,6])
fig+= shapes.Sphere([8,9,0])
fig+= shapes.Spheres([[8,10,0],[9,10,0]], r=0.2, c='g')

fig += shapes.Ellipsoid().pos(9,11)
fig += shapes.Grid().scale(2).pos(7,11)

fig += shapes.Rectangle([2,6], [4,8], radius=0.1).c('b5')

fig += shapes.Cone().scale(2).pos(10,6).rotateY(90, around='itself')
fig += shapes.Text3D("MyTest3D", c='k', justify='center', font="Quikhand")\
    .pos(5,11).scale(0.5).rotateZ(20, around='itself')

fig += shapes.Latex('sin(x^2)', res=150).scale(3).pos(10,0)

fig2 = Figure([-2.5, 14],[-5,14], padding=0, title='Test Embedding Figure')
fig2.insert(fig)

# show(fig2, size=(1600, 1100), zoom='tight').close()



