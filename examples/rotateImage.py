# 
# Normal jpg/png images can be loaded and rendered as any vtkImageActor
#
import plotter

vp = plotter.vtkPlotter()

for i in range(5): 
    a = vp.load('../textures/dog.jpg', alpha=1)
    a.SetScale(1.0) # image can be scaled in size
    a.RotateX(20*i)
    a.SetPosition(0, 0, 30*i)

vp.show()
