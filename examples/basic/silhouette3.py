"""Make the silhouette of an object
move along with camera position"""
from vedo import *

# Silhouette is camera-dependent, so create plotter first.
plt = Plotter(bg='blue4', bg2='white')

s = Mesh(dataurl+'shark.ply').c('gray',0.1).lw(1).lc('k')
silh = s.silhouette().c('red3',0.9).lw(3)

plt.show(s, silh, __doc__).close()
