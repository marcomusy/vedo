"""Make the silhouette of an object
move along with camera position"""
from vedo import *

s = Mesh(dataurl+'shark.ply').c('gray',0.1).lw(0.1).lc('k')

# this call creates the camera object needed by silhouette()
show(s, bg='db', bg2='lb', interactive=False)

sil = s.silhouette().c('darkred',0.9).lw(3)

show(s, sil, __doc__).interactive().close()
