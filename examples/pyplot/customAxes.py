"""Customizing axes style
(40+ control parameters!)

Axes font: """
from vedo import Box, show, settings

settings.defaultFont = 'Theemim'

#an invisible box:
box = Box(pos=(2.7,0,0), length=12, width=10, height=8, alpha=0)

# make a dictionary of axes options
axes_opts = dict(
    xtitle='My variable \Sigma^\lowerxi_lm  in units of \mum^3',
    ytitle='This is my \ncustomized y-axis',
    ztitle='z values go here!',
    yPositionsAndLabels= [(-3.2,'Mark'), (-1.2,'Carmen'), (3,'John')],
    textScale=1.2,       # make text 20% bigger
    numberOfDivisions=5, # approx number of divisions on longest axis
    axesLineWidth= 2,
    gridLineWidth= 1,
    xOriginMarkerSize=0.02,
    yOriginMarkerSize=None,
    titleDepth=0.1,      # extrusion fractional depth of title text
    xyGrid=True,         # show a gridded wall on plane xy
    yzGrid=True,
    zxGrid=False,
    zxGrid2=True,        # show zx plane on opposite side of the bounding box
    xyPlaneColor='green',
    xyGridColor='darkgreen', # line color
    xyAlpha=0.2,         # plane opacity
    showTicks=True,      # show major ticks
    xTitlePosition= 0.5, # title fractional positions along axis
    yTitleOffset=-0.02,  # title fractional offset distance from axis line
    xTitleJustify="top-center",
    xTitleRotation=20,
    yTitleBox=True,
    xLineColor='black',
    zLineColor='blue',
    zTitleColor='blue',
    zTitleBackfaceColor='red', # color of axis title on the backface
    zTitleSize=0.04,
    xHighlightZero=True, # draw a line highlighting zero position if in range
    xHighlightZeroColor='tomato',
    xTickLength=0.015,
    xTickThickness=0.0025,
    xTickColor='black',
    xMinorTicks=3,       # number of minor ticks btw two major ticks
    tipSize=0.01,        # size of the arrow tip cone
    xLabelOffset=0.015,  # offset of numeric labels
    yLabelSize=0.025,    # size of the numeric labels along Y axis
)

show(box, __doc__+settings.defaultFont, axes=axes_opts, viewup='z')

