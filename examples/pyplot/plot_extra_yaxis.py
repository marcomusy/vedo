"""Add a secondary y-axis for units conversion"""
from vedo import np, settings, dataurl, Mesh, show
from vedo.pyplot import plot, Figure

settings.annotatedCubeTexts = ['front','back','left','right','top','bttm']

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
        xyGridTransparent=False,
        xyGridColor='k7',
        xyAlpha=1,
        xyPlaneColor='w',
        yHighlightZero=True,
    ),
)

# This empty Figure just creates a new y-axis in red
fig2 = Figure(
    fig1.xlim,                    # same as fig1
    fig1.ylim * 7.236,            # units conversion factor
    aspect=fig1.aspect,           # same as fig1
    padding=fig1.padding,         # same as fig1
    xtitle='',                    # don't draw the x-axis!
    ytitle='Poundal [lb*ft/s^2 ]',
    axes=dict(                    # extra options for y-axis
        yShiftAlongX=1,           # shift 100% to the right
        yLabelOffset=-1,
        yLabelJustify="center-left",
        yTitlePosition=0.5,
        yTitleJustify="top-center",
        # yTitleOffset=-0.12,
        axesLineWidth=3,
        numberOfDivisions=10,
        c='red3',
    ),
)

fig1.rotateX(90).rotateZ(90).shift(-0.5, 0, 1)
fig2.rotateX(90).rotateZ(90).shift(-0.5, 0, 1)

msh = Mesh(dataurl+"cessna.vtk")

cam = dict(  # press C to get these values
    pos=(3.899, -0.4781, 1.157),
    focalPoint=(-0.1324, 0.9041, 0.3530),
    viewup=(-0.1725, 0.06857, 0.9826),
)
show(msh, fig1, fig2, __doc__, axes=5, camera=cam, bg2='lb').close()


