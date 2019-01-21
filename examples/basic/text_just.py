# Text jutification and positioning.
# (center, top-left, top-right, bottom-left, bottom-right)
import vtkplotter

vp = vtkplotter.Plotter(axes=8, verbose=0)

txt = 'I like\nto play\nguitar'

vp.text(txt, pos=[1,2,0], s=0.2, c=3, bc=1, depth=0.2, justify='top-left')
vp.point([1,2,0], c='r') # mark text origin
vp.point([0,0,0])        # mark axes origin

vp.show()