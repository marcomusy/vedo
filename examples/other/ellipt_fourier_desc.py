"""Elliptic Fourier Descriptors
parametrizing a closed countour (in red)"""
import vedo, pyefd

shapes = vedo.load(vedo.dataurl+'timecourse1d.npy')

sh = shapes[55]
sr = vedo.Line(sh).mirror('x').reverse()
sm = vedo.merge(sh, sr).c('red5').lw(3)
pts = sm.points()[:,(0,1)]

rlines = []
for order in range(5,30, 5):
    coeffs = pyefd.elliptic_fourier_descriptors(pts, order=order, normalize=False)
    a0, c0 = pyefd.calculate_dc_coefficients(pts)
    rpts = pyefd.reconstruct_contour(coeffs, locus=(a0,c0), num_points=400)
    color = vedo.colorMap(order, "Blues", 5,30)
    rline = vedo.Line(rpts).lw(3).c(color)
    rlines.append(rline)

sm.z(0.1) # move it on top so it's visible
vedo.show(sm, *rlines, __doc__, axes=1, bg='k', size=(1190, 630), zoom=1.8).close()
