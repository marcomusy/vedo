from vedo import Hyperboloid, show
from vedo.pyplot import histogram
import numpy as np
np.random.seed(1)

##################################################################
radhisto = histogram(
    np.random.rand(200)*6.28,
     mode='polar',
     title="random orientations",
     bins=10,
     c=range(10), #'orange', #uniform color
     alpha=0.8,
     labels=["label"+str(i) for i in range(10)],
)
show(radhisto, at=0, N=2, axes=0, sharecam=False)

##################################################################
hyp = Hyperboloid(res=20).cutWithPlane().rotateY(-90)
hyp.color('grey').alpha(0.3)

# select 10 random indices of points on the surface
idx = np.random.randint(0, hyp.NPoints(), size=10)

radhistos = []
for i in idx:
    #generate a random histogram
    rh = histogram(
         np.random.randn(100),
         mode='polar',
         bins=12,
         r1=0.2,     # inner radius
         phigap=1.0, # leave a space btw phi bars
         cmap='viridis_r',
         showDisc=False,
         showAngles=False,
         showErrors=False,
    )
    rh.scale(0.15)          # scale histogram to make it small
    rh.pos(hyp.points(i))   # set its position on the surface
    rh.orientation(hyp.normalAt(i)) # orient it along normal
    radhistos.append(rh)

show(hyp, radhistos, at=1).interactive().close()
