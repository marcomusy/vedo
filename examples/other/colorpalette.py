"""
Generate a list of N colors starting from color1 to color2 in RGB or HSV space
"""
print(__doc__)

from vedo.colors import buildPalette, getColorName

cols = buildPalette("red", "blue", 10, hsv=True)

for c in cols:
    print("rgb =", c, " closest color is:", getColorName(c))
