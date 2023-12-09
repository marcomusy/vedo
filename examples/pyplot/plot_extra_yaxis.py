"""Add a secondary y-axis for units conversion"""
from vedo import np, settings, dataurl, Mesh, show
from vedo.pyplot import plot, Figure

settings.annotated_cube_texts = ['front','back','left','right','top','bttm']

x0, x1 = [0.3, 2.0]
x = np.linspace(x0, x1, num=50)

# The main plot
fig1 = plot(
    x,
    1000*np.cos(x+1),
    xlim=[x0, x1],
    ylim=[-1000, 250],
    aspect=16/9,
    padding=0,                    # do not mess up with margins
    title="Wing pull vs position",
    xtitle="Distance from airplane axis [m]",
    ytitle="N [Kg*m/s^2 ]",
    axes=dict(
        xygrid_transparent=False,
        xygrid_color='k7',
        xyalpha=1,
        xyplane_color='w',
        yhighlight_zero=True,
    ),
)
# fig1copy = fig1.clone2d("bottom-right") # can make it 2d (on screen)

# This empty Figure just creates a new y-axis in red
fig2 = Figure(
    fig1.xlim,                    # same as fig1
    fig1.ylim * 7.236,            # units conversion factor
    aspect=fig1.aspect,           # same as fig1
    padding=fig1.padding,         # same as fig1
    xtitle='',                    # don't draw the x-axis!
    ytitle='Poundal [lb*ft/s^2 ]',
    axes=dict(                    # extra options for y-axis
        number_of_divisions=10,
        yshift_along_x=1,         # shift 100% to the right
        ylabel_offset=-1,
        ylabel_justify="center-left",
        ytitle_position=0.5,
        ytitle_justify="top-center",
        axes_linewidth=3,
        c='red3',
    ),
)

fig1.rotate_x(90).rotate_z(90).shift(-0.5, 0, 1)
fig2.rotate_x(90).rotate_z(90).shift(-0.5, 0, 1)

msh = Mesh(dataurl+"cessna.vtk")

cam = dict(  # press C to get these values
    pos=(3.899, -0.4781, 1.157),
    focal_point=(-0.1324, 0.9041, 0.3530),
    viewup=(-0.1725, 0.06857, 0.9826),
)
show(msh, fig1, fig2, __doc__,
     axes=5, camera=cam, bg2='lb').close()


