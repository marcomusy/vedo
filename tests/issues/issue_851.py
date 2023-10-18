from vedo import *

x=[5]*20
y=[24]*20
z=range(20)
c=range(20)

cols = color_map(c, "viridis", vmin=0, vmax=25)

tube1 = Tube(list(zip(x,y,z)), c=c, res=30, r=5)
tube2 = Tube(list(zip(x,y,z)), c=cols, res=30, r=5).pos(15,0,0)

show(tube1, tube2, bg='black', bg2='bb', axes=True)
