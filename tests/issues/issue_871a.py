import numpy as np
from vedo import settings, Plotter, Arrow, screenshot, \
    ScalarBar3D, Axes, mag, color_map, Assembly, Line

settings.default_font = 'Theemim'
settings.multi_samples=8

filename = 'data/obs_xyz.dat'
xyz = np.loadtxt(filename, dtype=np.float64)
xyz = xyz[1:, :] - np.array([0, 20, 0])
n_obs, n_col = xyz.shape

# Read in the data
num_field = np.loadtxt('data/tmp_field.txt', dtype=np.float64)
amp = mag(num_field)

# We need to create a few slices from the original arrays
# First create a mask array to mark all receiver points with x = -10, 0, 10
x_mark = [-19.5, -9.236842110, 1.026315790, 11.289473680, 19.5]
y_mark = [-4.5, 4.5]
mask = np.zeros(n_obs, dtype=bool)

# We need to create a mask array to mark all receiver points with x being
# any of x_mark and y being any of y_marko
for x_m in x_mark:
    mask_local = np.abs(xyz[:, 0] - x_m) < 1e-6
    mask = np.logical_or(mask, mask_local)

# Create an Arrows object for all the receivers where mask is True
start = xyz[mask, :]
orientation = num_field[mask, :]
orientation = orientation / mag(orientation)[:, None] # normalize
amp_mask = amp[mask]
vrange = np.array([amp_mask.min(), amp_mask.max()])

arrs = []
for i in range(start.shape[0]):
    arr = Arrow(start[i], start[i] + orientation[i] * 4)
    color = color_map(
        amp_mask[i], "jet", 
        vmin=vrange[0], vmax=vrange[1],
    )
    arr.color(color).lighting('off')
    arrs.append(arr)
arrows = Assembly(arrs)

# create a 2D scalarbar
# scalarbar = ScalarBar(
#     vrange,
#     title='E (V/m)',
#     c="jet",
#     font_size=22, 
#     pos=(0.7, 0.25),
#     size=(60,600),
# )

# create a dummy line to use as a 3D scalarbar
pos = (-10, -14, -32)
line = Line([0, 0], [1, 0]).cmap("jet", vrange * 1e6)
scalarbar = ScalarBar3D(
    line,
    # c="white",
    title="E (:muV/m)",
    title_size=3,
    label_rotation=90,
    label_offset=.5,
    label_size=2,
    pos=pos,
    size=(1, 20),
    nlabels=5,
)
scalarbar.rotate_z(-90)

size = (3920, 2160)
plt = Plotter()

axes = Axes(
    arrows, 
    xtitle='Easting (m)',
    ytitle='Northing (m)',
    ztitle='Elevation (m)',
    xtitle_position=0.60,
    xlabel_size=0.018,
    xtitle_offset=0.15,
    ytitle_position=0.85,
    ylabel_rotation=-90,
    ylabel_size=0.02,
    ytitle_rotation=180,
    y_values_and_labels=[(-5, "-5"), (0, "0"), (5, "5")],
    axes_linewidth=4,
    xrange=arrows.xbounds(),
    yrange=arrows.ybounds(),
    zxgrid2=True,
    zshift_along_y=1,
    zaxis_rotation=-70,
    ztitle_size=0.02,
    ztitle_position=0.68,
    xyframe_line=True,
    grid_linewidth=2,
)
cam = dict(
    position=(-58.8911, -54.5234, 8.63461),
    focal_point=(-5.25549, -0.0457020, -23.8989),
    viewup=(0.248841, 0.304150, 0.919549),
    distance=83.0844,
    clipping_range=(34.8493, 143.093),
)
fig_name = 'data/electric_field.png'
plt.show(arrows, axes, scalarbar, interactive=0, camera=cam)
# screenshot(fig_name)
plt.interactive().close()