# 
# Normal jpg/png images can be loaded and rendered as any vtkImageActor
#
import vtkplotter

vp = vtkplotter.Plotter(axes=3)

for i in range(5): 
    a = vp.load('data/images/dog.jpg')
    a.scale(1-i/10.)                  # image can be scaled in size
    a.rotateX(20*i).pos([0, 0, 30*i]) # can concatenate methods

vp.show()
