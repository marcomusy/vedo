#!/usr/bin/env python
#
# Generate a list of N colors starting from color1 to color2 in RGB or HSV space

from vtkplotter.colors import makePalette, getColorName


cols = makePalette('red', 'blue', 10, HSV=True)

for c in cols:
    print('rgb =', c, ' closest color is:', getColorName(c))

