"""Markers set, analogous to matplotlib"""
from vedo import Plotter, Marker, Text3D

symbols = ['.', 'p','*','h','D','d','o','v','>','<','s','x','+','a']

plt = Plotter(size=(1500,300), axes=0)
for i,s in enumerate(symbols):
    plt += Marker(s, filled=True).x(i*0.6).color(i)
    plt += Text3D(s, pos=[i*0.6,-0.6,0], s=0.12).color('k')
plt += __doc__

plt.show(zoom=5, viewup='2d').close()
