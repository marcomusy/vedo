from vedo import *
import matplotlib.colors as colors
import matplotlib.pyplot as plt

settings.default_font = "Antares"

man = Mesh(dataurl + "man.vtk")
h_knees = -0.5
over_limit  =  1.5
under_limit = -1.4

# let the scalar be the z coordinate of the mesh vertices
scals = man.vertices[:, 2]

# build a complicated colour map
c1 = plt.cm.viridis(np.linspace(0.0, 0.7, 128))
c2 = plt.cm.terrain(np.linspace(0.5, 0.8, 128))
c = np.vstack((c1, c2))
cmap = colors.LinearSegmentedColormap.from_list("heights", c)
cmap.set_over(color="red")
cmap.set_under(color="orange")
norm = colors.TwoSlopeNorm(h_knees, vmin=under_limit, vmax=over_limit)
mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

# build look up table
lut = build_lut(
    [(v, mapper.to_rgba(v)[:3]) for v in np.linspace(under_limit, over_limit, 128)],
    above_color=cmap.get_over()[:3],
    below_color=cmap.get_under()[:3],
    vmin=under_limit,
    vmax=over_limit,
)

man.cmap(lut, scals)
man.add_scalarbar3d(above_text="Above Eyes", below_text="Below Heels")
man.scalarbar = man.scalarbar.clone2d("center-left", size=0.3)  # make it 2D
show(man, axes=1, viewup="z")
