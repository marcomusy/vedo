"""Markers set, analogous to matplotlib"""
from vedo import Plotter, Marker, Text3D

symbols = ['.','o','O', '0', 'p','*','h','D','d','v','^','>','<','s', 'x', 'a']

plt = Plotter(size=(1500,300))
plt += __doc__

for i,s in enumerate(symbols):
    plt += Marker(s, filled=True).x(i*0.6).backColor('blue5')
    plt += Text3D(s, pos=[i*0.6,-0.6,0], s=0.12, literal=True, font="Calco")

plt.show(zoom='tight').close()
