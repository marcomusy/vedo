"""1D Moving Least Squares (MLS)
to project a cloud of unordered points
to become a smooth, ordered line"""
from vedo import *

settings.default_font = "Antares"

N = 3  # nr. of iterations

# build some initial cloud of noisy points along a line
pts = [(sin(6*x), cos(2*x)*x, cos(9*x)) for x in np.arange(0,2, 0.001)]
# pts = [(0, sin(x), cos(x)) for x in np.arange(0,6, .002)]
# pts = [(sqrt(x), sin(x), x/5) for x in np.arange(0, 16, 0.01)]

pts += np.random.randn(len(pts), 3) / 15  # add noise
np.random.shuffle(pts)  # make sure points are not ordered

pts = Points(pts, r=5)

plt = Plotter(N=N, axes=1)
plt.at(0).show(pts, __doc__, viewup='z')

for i in range(1, N):
    pts = pts.clone().smooth_mls_1d(n=50).color(i)
    if i == N-1:
        # at the last iteration make sure points
        # are separated by tol (in % of the bounding box)
        pts.subsample(0.025)
    plt.at(i).show(pts, f"Iteration {i}, #points: {pts.npoints}")

line = pts.generate_segments().clean().join()
# printc("lines:", line.lines)

plt += [line, line.labels("id").bc("blue5")]

plt.interactive().close()
