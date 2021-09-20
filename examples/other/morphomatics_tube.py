# Git repo at : https://morphomatics.github.io/tutorial_ssm/
# Install with:
#  pip install git+https://github.com/morphomatics/morphomatics.git#egg=morphomatics
#
from morphomatics.geom import Surface
from morphomatics.stats import StatisticalShapeModel
from morphomatics.manifold import FundamentalCoords
import numpy as np
import vedo

ln1 = [[1, 1, x / 2] for x in np.arange(0,15, 0.15)]
ln2 = [[np.sin(x), np.cos(x), x / 2] for x in np.arange(0,15, 0.15)]
rads= [0.4*(np.cos(6*ir/len(ln2)))**2+0.1 for ir in range(len(ln2))]

vmesh1 = vedo.Tube(ln1, r=0.08, c="tomato").triangulate().clean()
vmesh2 = vedo.Tube(ln2, r=rads, c="tomato").triangulate().clean()

verts1 = vmesh1.points()
verts2 = vmesh2.points()
faces  = np.array(vmesh1.faces())

# construct model
SSM = StatisticalShapeModel(lambda ref: FundamentalCoords(ref))
surfaces = [Surface(v, faces) for v in [verts1, verts2]]
SSM.construct(surfaces)

# sample trajectory along the main mode of variation
shapes = [vmesh1]
std = np.sqrt(SSM.variances[0])
for t in np.linspace(-1.0, 1.0, 20):
    e = SSM.space.exp(SSM.mean_coords, t * std * SSM.modes[0])
    v = SSM.space.from_coords(e)
    shapes.append(vedo.Mesh([v, faces]))
shapes.append(vmesh2.rotateY(-90).flat())

plt = vedo.applications.Browser(shapes, prefix="shape ")
plt.show(viewup='z', bg2='lb')
