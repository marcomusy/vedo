"""Draw a line in 3D that fits a cloud of 20 Points,
Show the first set of 20 points and fit a plane to them."""
from vedo import *

# declare the class instance
plt = Plotter(axes=1)

# draw 500 fit lines superimposed and very transparent
for i in range(500):
    x = np.linspace(-2, 5, 20)  # generate every time 20 points
    y = np.linspace(1, 9, 20)
    z = np.linspace(-5, 3, 20)
    data = np.stack((x,y,z), axis=1)
    data+= np.random.normal(size=data.shape) * 0.8  # add gauss noise
    plt += fit_line(data).lw(4).alpha(0.04).c("violet")  # fit a line

# 'data' still contains the last iteration points
plt += Points(data).color("yellow").ps(10)

print("Line 0 Fit slope = ", plt.objects[0].slope)

plane = fit_plane(data).c("green4")  # fit a plane
print("Plane Fit normal =", plane.normal)

plt += plane, __doc__

plt.show().close()
