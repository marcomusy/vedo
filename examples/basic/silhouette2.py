"""Make the silhouette of an object
move along with camera position"""
from vedo import *

s = load(datadir+'shark.ply')
s.alpha(0.1).c('gray').lw(0.1).lc('k')

# this call creates the camera object needed by silhouette()
show(s, bg='db', bg2='lb', interactive=False)

sil = s.silhouette().c('darkred').alpha(0.9).lw(3)

show(s, sil, __doc__, interactive=True)