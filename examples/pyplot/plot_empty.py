"""Create an empty Figure to be filled in a loop
Any 3D Mesh object can be added to the figure!"""
from vedo import *
from vedo.pyplot import plot, Figure

settings.defaultFont = "Cartoons123"
settings.palette = 2

# Create an empty Figure and plot on it
fig = Figure(
    xlim=(0,12),
    ylim=(-1.5, 1.5),
    padding=0,    # no extra space
    aspect=16/9,  # desired aspect ratio
    xtitle="speed [mph]",
    grid=True,
    axes=dict(axesLineWidth=3, xyFrameLine=3),
)

for i in range(2,11,2):
    x = np.linspace(0, 4*np.pi, 20)
    y = np.sin(x) * np.sin(x/12) * i/5

    fig += plot(x, y, '-0', lc=i, splined=True, like=fig)

fig += Arrow([5,-1], [8,-1], s=0.5, c='green3')

# Add any number of polygonal Meshes.
# Use insert() to preserve the object aspect ratio inside the Figure coord system:
mesh = Mesh(dataurl+'cessna.vtk').c('blue5').scale(0.5).pos(4, 0.5, 0.5)
circle = Circle([5,0.5,-0.1], c='orange5')
fig.insert(mesh, circle)

show(fig, __doc__, size=(800,700), zoom='tight').close()


