"""Normal jpg/png image formats can be loaded,
cropped, rotated and positioned in 3D."""
from vedo import Plotter, Image, dataurl

plt = Plotter(axes=7)

pic = Image(dataurl+"images/dog.jpg")

for i in range(5):
    p = pic.clone()
    p.crop(bottom=0.20) # crop 20% from bottom
    p.scale(1-i/10.0).rotate_x(20*i).z(30*i)
    p.alpha(0.8)
    plt += p

plt += __doc__
plt.show().close()
