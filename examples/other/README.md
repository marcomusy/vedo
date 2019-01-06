# Other examples
In this directory you will find special examples that might need other external modules.
```bash
git clone https://github.com/marcomusy/vtkplotter.git
cd vtkplotter/examples/other
python example.py  # on mac OSX try 'pythonw' instead
```
(_click thumbnail image to get to the python script_)

|    |    |
|:-------------:|:-----|
| [![colorpalette](https://user-images.githubusercontent.com/32848391/50739011-2c94c200-11da-11e9-8f36-ede1b2a014a8.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/colorpalette.py)<br/> `colorpalette.py` | Generate a list of N colors starting from color1 to color2 in RGB or HSV space.|
|    |    |
| [![colorprint](https://user-images.githubusercontent.com/32848391/50739010-2bfc2b80-11da-11e9-94de-011e50a86e61.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/colorprint.py)<br/> `colorprint.py` | Printing in colors to the terminal.<br> Available colors: <br>0-black, 1-red, 2-green, 3-yellow, 4-blue, 5-magenta, 6-cyan, 7-white<br>Available modifiers:<br> c (foreground color), bc (background color), hidden, bold, blink, underline, dim, invert, box|
|    |    |
| [![icon](https://user-images.githubusercontent.com/32848391/50739009-2bfc2b80-11da-11e9-9e2e-a5e0e987a91a.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/icon.py)<br/> `icon.py` | Make an icon image and place it in one of the 4 corners. |
|    |    |
| [![logosh](https://shtools.oca.eu/shtools/images/company_logo.png)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/spherical_harmonics1.py)<br/> `spherical_harmonics1.py` |  First part of the example:<br>intersect an actor with lines from the origin with `intersectWithLine()` method and draw the intersection points in blue.<br>Second part of the example:<br>expand an arbitrary closed shape in spherical harmonics using package [SHTOOLS](https://shtools.oca.eu/shtools/) and then truncate the expansion to a specific lmax and reconstruct the projected points in red.|
|    |    |
| [![logosh](https://shtools.oca.eu/shtools/images/company_logo.png)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/spherical_harmonics2.py)<br/> `spherical_harmonics2.py` | Morph one shape into another using spherical harmonics package [SHTOOLS](https://shtools.oca.eu/shtools/). In this example we morph a sphere into a octahedron and viceversa.|
|    |    |
| [![makevideo](https://user-images.githubusercontent.com/32848391/50739007-2bfc2b80-11da-11e9-97e6-620a3541a6fa.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/makeVideo.py)<br/> `makeVideo.py` |  Make a video of a rotating spider (needs linux `ffmpeg`). <br/>Set `offscreen=True` to only produce the video without any graphical window showing.|
