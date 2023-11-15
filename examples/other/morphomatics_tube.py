"""Morphomatics example"""
try:
    from morphomatics.geom import Surface
    from morphomatics.stats import StatisticalShapeModel
    from morphomatics.manifold import FundamentalCoords
except ModuleNotFoundError:
    print("Install with:")
    print("pip install git+https://github.com/morphomatics/morphomatics.git#egg=morphomatics")
    exit(0)
import numpy as np
import vedo

ln1 = [[1, 1, x / 2] for x in np.arange(0,15, 0.15)]
ln2 = [[np.sin(x), np.cos(x), x / 2] for x in np.arange(0,15, 0.15)]
rads= [0.4*(np.cos(6*ir/len(ln2)))**2+0.1 for ir in range(len(ln2))]

vmesh1 = vedo.Tube(ln1, r=0.08).triangulate().clean()
vmesh2 = vedo.Tube(ln2, r=rads).triangulate().clean()

verts1 = vmesh1.vertices
verts2 = vmesh2.vertices
faces  = np.array(vmesh1.cells)

# construct model
SSM = StatisticalShapeModel(lambda ref: FundamentalCoords(ref))
surfaces = [Surface(v, faces) for v in [verts1, verts2]]
SSM.construct(surfaces)

# sample trajectory along the main mode of variation
shapes = []
std = np.sqrt(SSM.variances[0])
for t in np.linspace(-1.0, 1.0, 20):
    e = SSM.space.exp(SSM.mean_coords, t * std * SSM.modes[0])
    v = SSM.space.from_coords(e)
    shapes.append(vedo.Mesh([v, faces]))

plt = vedo.applications.Browser(shapes, slider_title="shape", bg2='lb')
plt.show(viewup='z').close()
