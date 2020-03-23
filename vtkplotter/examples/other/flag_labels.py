"""Hover mouse onto an object
to pop a flag-style label
"""
from vtkplotter import *

# Can modify default behaviour through settings:
#settings.flagDelay = 0          # popup delay in milliseconds
#settings.flagFont = "Courier"   # font type ("Arial", "Courier", "Times")
#settings.flagFontSize = 18
#settings.flagJustification = 0
#settings.flagAngle = 0
#settings.flagBold = False
#settings.flagItalic = True
#settings.flagShadow = False
#settings.flagColor = 'black'
#settings.flagBackgroundColor = 'white'

s = load(datadir+'/bunny.obj').flag() # picks filename by default
c = Cube(side=0.2).x(0.3).flag('my cube\nlabel')

#s.flag(False) #disable

show(s, c, Text2D(__doc__))
