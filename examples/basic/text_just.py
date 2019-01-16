# Text jutification
# (center, top-left, top-right, bottom-left, bottom-right)
import vtkplotter

vp = vtkplotter.Plotter(axes=8)

txt = 'I like\nplaying\nguitar'

vp.text(txt, pos=[1,2,0], s=.25, c=3, justify='top-left')
vp.point([1,2,0], c='r') # the text placement
vp.point([0,0,0])        # the axes origin

vp.show()