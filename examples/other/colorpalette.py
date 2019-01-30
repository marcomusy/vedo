'''
Generate a list of N colors starting from color1 to color2 in RGB or HSV space
'''
from __future__ import print_function
print(__file__, __doc__)

from vtkplotter.colors import makePalette, getColorName

cols = makePalette('red', 'blue', 10, hsv=True)

for c in cols:
    print('rgb =', c, ' closest color is:', getColorName(c))

