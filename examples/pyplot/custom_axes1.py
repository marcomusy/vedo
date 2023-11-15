"""Customizing axes style
(40+ control parameters!)
Title font: """
from vedo import Box, Lines, Points, Spline, show, settings

settings.default_font = 'Theemim'

# an invisible box:
world = Box(pos=(2.7,0,0), size=(12,10,8), alpha=0)

# a dummy spline with its shadow on the xy plane
pts = Points([(-2,-3.2,-1.5), (3,-1.2,-2), (7,3,4)], r=12)
spl = Spline(pts, res=50)            # make a spline from points
spl.add_shadow(plane='z', point=-4)  # add its shadow at z=-4
lns = Lines(spl, spl.shadows[0])     # join spline points with its own shadow

# make a dictionary of axes options
axes_opts = dict(
    xtitle='My variable :Omega^:lowerxi_lm  in units of :mum^3', # latex-style syntax
    ytitle='This is my highly\ncustomized y-axis',
    ztitle='z in units of Å', # many unicode chars are supported (type: vedo -r fonts)
    y_values_and_labels=[(-3.2,'Mark^a_-3.2'), (-1.2,'Carmen^b_-1.2'), (3,'John^c_3')],
    text_scale=1.3,           # make all text 30% bigger
    number_of_divisions=5,    # approximate number of divisions on longest axis
    axes_linewidth= 2,
    grid_linewidth= 1,
    zxgrid2=True,             # show zx plane on opposite side of the bounding box
    yzgrid2=True,             # show yz plane on opposite side of the bounding box
    xyplane_color='green7',
    xygrid_color='green3',    # darkgreen line color
    xyalpha=0.2,              # grid opacity
    xtitle_position=0.5,      # title fractional positions along axis
    xtitle_justify="top-center", # align title wrt to its axis
    ytitle_size=0.02,
    ytitle_box=True,
    ytitle_offset=0.05,
    ylabel_offset=0.4,
    yhighlight_zero=True,     # draw a line highlighting zero position if in range
    yhighlight_zero_color='red',
    zline_color='blue5',
    ztitle_color='blue5',
    ztitle_backface_color='v',# violet color of axis title backface
    label_font="Quikhand",
    ylabel_size=0.025,        # size of the numeric labels along Y axis
    ylabel_color='green4',    # color of the numeric labels along Y axis
)

show(world, pts, spl, lns, __doc__+settings.default_font, axes=axes_opts).close()

