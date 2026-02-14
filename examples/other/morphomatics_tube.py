"""Morphomatics example"""
import numpy as np
import vedo

# Configure inputs and run the visualization workflow.
try:
    from morphomatics.geom import Surface
    from morphomatics.stats import StatisticalShapeModel
    from morphomatics.manifold import FundamentalCoords
except ModuleNotFoundError:
    print("Install with:")
    print("pip install -U git+https://github.com/morphomatics/morphomatics.git#egg=morphomatics")
    exit(0)

ln1 = [[1, 1, x / 2] for x in np.arange(0, 15, 0.2)]
ln2 = [[np.sin(x), np.cos(x), x / 2] for x in np.arange(0,15, 0.2)]
rads= [0.4*(np.cos(6*ir/len(ln2)))**2+0.2 for ir in range(len(ln2))]

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
    e = SSM.space.connec.exp(SSM.mean_coords, t * std * SSM.modes[0])
    v = SSM.space.from_coords(e)
    shapes.append(vedo.Mesh([v, faces]))

plt = vedo.applications.Browser(shapes, slider_title="shape", bg2='blue9')
plt.add(shapes[ 0].clone().on().wireframe().alpha(0.1).lighting('off'))
plt.add(shapes[-1].clone().on().wireframe().alpha(0.1).lighting('off'))
plt.show(viewup='z', axes=1).close()
