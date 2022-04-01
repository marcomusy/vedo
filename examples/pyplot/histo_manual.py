"""Categories and repeats"""
# manually create a plot by adding Rectangles to a Figure object
from vedo import np, settings, Rectangle, Text3D, Line, DashedLine
from vedo.pyplot import Figure

settings.defaultFont = "Theemim"

#################################################################### First plot
groupA = np.random.randn(3)*10+50
groupB = np.random.randn(3)*10+60
groupC = np.random.randn(3)*10+70

fig = Figure(
    [-5,55], [-10,100],   # x and y ranges
    xtitle='', ytitle='', # this disables x and y axes
)

#################
x0 = 0
for i in range(3):
    x1 = x0 + 4
    val= groupA[i]
    fig += Rectangle([x0,0], [x1, val], c=f'red{3+i*2}')
    x0 = x1
fig += Text3D("Group A", justify='center', c='k').pos(6,-7).scale(4)
fig += Line([-1,0], [13, 0], lw=2)

#################
x0 = 20
for i in range(3):
    x1 = x0 + 4
    val= groupB[i]
    fig += Rectangle([x0,0], [x1, val], c=f'purple{3+i*2}')
    x0 = x1
fig += Text3D("Group B", justify='center', c='k').pos(26,-7).scale(4)
fig += Line([19,0], [33, 0], lw=2)

#################
x0 = 40
for i in range(3):
    x1 = x0 + 4
    val= groupC[i]
    fig += Rectangle([x0,0], [x1, val], c=f'orange{3+i*2}')
    x0 = x1
fig += Text3D("Group C", justify='center', c='k').pos(46,-7).scale(4)
fig += Line([39,0], [53, 0], lw=2)

#################
fig += DashedLine([-2,50], [55,50], c='k3', lw=1)
fig += Text3D("50%").pos(-7,49).scale(3).c('k')
fig.show(size=(1000,700), zoom='tight', title=__doc__).clear()


#################################################################### Second plot
fig = Figure(
    [0, 100], [-20, 80],
    aspect=3/4,             # can change the aspect ratio
    xtitle='', ytitle='', # this disables x and y axes
)
for i in range(5):
    val = np.random.randn()*10+50
    y0, y1 = 2*i, 2*i+1
    fig += Rectangle([0,y0], [100,y1], radius=0.5, c='k6')
    fig += Rectangle([0,y0], [val,y1], radius=0.5, c='r4').z(1)
fig += Text3D("Group A", justify='center', c='k').pos(50,-5).scale(2.5)


for i in range(5):
    val = np.random.randn()*10+60
    y0, y1 = 2*i + 20, 2*i+1 + 20
    fig += Rectangle([0,y0], [100,y1], radius=0.5, c='k6')
    fig += Rectangle([0,y0], [val,y1], radius=0.5, c='p5').z(1)
fig += Text3D("Group B", justify='center', c='k').pos(50,15).scale(2.5)

for i in range(5):
    val = np.random.randn()*10+70
    y0, y1 = 2*i + 40, 2*i+1 + 40
    fig += Rectangle([0,y0], [100,y1], radius=0.5, c='k6')
    fig += Rectangle([0,y0], [val,y1], radius=0.5, c='o5').z(1)
fig += Text3D("Group C", justify='center', c='k').pos(50,35).scale(2.5)

fig.show(size=(1000,700), zoom='tight').close()
