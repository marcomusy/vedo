"""Customizing axes style
(30 control parameters)"""
# check out parameters usage at:
#
from vtkplotter import *

b = Box(pos=(0,0,0), length=90, width=80, height=70).alpha(0)

show(b, Text(__doc__), bg='white',
     axes={
        'xtitle':'Some long variable description [a.u.]',
        'ytitle':'This is my \ncustomized y-axis',
        'ztitle':'z values go here!',
        'numberOfDivisions':4, # number of divisions on the shortest axis
        'axesLineWidth': 2,
        'gridLineWidth': 1,
        'reorientShortTitle':True,
        'xKeepAspectRatio':True,
        'originMarkerSize':.02,
        'enableLastLabel':True, # show last numeric label on axes
        'titleDepth':0, # extrusion fractional depth of title text
        'xyGrid':True,  # show a gridded wall on plane xy
        'yzGrid':True,
        'zxGrid':True,
        'zxGrid2':False, # show zx plane on opposite side of the bounding box
        'xyPlaneColor':'green',
        'xyGridColor':'darkgreen', # line color
        'xyAlpha':0.2,   # plane opacity
        'showTicks':True, # show major ticks
        'xTitlePosition': 0.5, # title fractional positions along axis
        'xTitleOffset':0.02,   # title fractional offset distance from axis line
        'xTitleJustify':"top-center",
        'xTitleRotation':20,
        'xLineColor':'black',
        'zLineColor':'blue',
        'zTitleColor':'blue',
        'zTitleBackfaceColor':'red', # color of axis title on the backface
        'zTitleSize':0.05,
        'xHighlightZero':True, # draw a line highlighting zero position if in range
        'xHighlightZeroColor':'tomato',
        'xTickRadius':0.005,
        'xTickThickness':0.0025,
        'xTickColor':'black',
        'xMinorTicks':3, # number of minor ticks btw two major ticks
        'tipSize':0.01,  # size of the arrow tip cone
        'xTicksPrecision':2, # nr. of significative digits to be shown
        'xLabelSize':0.02, # size of the numeric labels along axis
        'xLabelOffset': -0.05, # offset of numeric labels
     }
)
