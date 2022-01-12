"""Customizing axes style
(40+ control parameters!)
Title font: """
from vedo import Box, Lines, Points, Spline, show, settings

settings.defaultFont = 'Theemim'

# an invisible box:
world = Box(pos=(2.7,0,0), size=(12,10,8), alpha=0)

# a dummy spline with its shadow on the xy plane
pts = Points([(-2,-3.2,-1.5), (3,-1.2,-2), (7,3,4)], r=12)
spl = Spline(pts, res=50).addShadow(plane='z', point=-4) # make spline and add its shadow at z=-4
lns = Lines(spl, spl.shadows[0])                         # join spline points with its own shadow

# make a dictionary of axes options
axes_opts = dict(
    xtitle='My variable \Omega^\lowerxi_lm  in units of \mum^3', # latex-style syntax
    ytitle='This is my highly\ncustomized y-axis',
    ztitle='z in units of Å', # many unicode chars are supported (type: vedo -r fonts)
    yValuesAndLabels=[(-3.2,'Mark^a_-3.2'), (-1.2,'Carmen^b_-1.2'), (3,'John^c_3')],
    textScale=1.3,       # make all text 30% bigger
    numberOfDivisions=5, # approximate number of divisions on longest axis
    axesLineWidth= 2,
    gridLineWidth= 1,
    zxGrid2=True,        # show zx plane on opposite side of the bounding box
    yzGrid2=True,        # show yz plane on opposite side of the bounding box
    xyPlaneColor='green7',
    xyGridColor='dg',    # darkgreen line color
    xyAlpha=0.2,         # grid opacity
    xTitlePosition=0.5,  # title fractional positions along axis
    xTitleJustify="top-center", # align title wrt to its axis
    yTitleSize=0.02,
    yTitleBox=True,
    yTitleOffset=0.05,
    yLabelOffset=0.4,
    yHighlightZero=True, # draw a line highlighting zero position if in range
    yHighlightZeroColor='red',
    zLineColor='blue',
    zTitleColor='blue',
    zTitleBackfaceColor='v', # violet color of axis title backface
    labelFont="Quikhand",
    yLabelSize=0.025,    # size of the numeric labels along Y axis
    yLabelColor='dg',    # color of the numeric labels along Y axis
)

show(world, pts, spl, lns, __doc__+settings.defaultFont, axes=axes_opts).close()

