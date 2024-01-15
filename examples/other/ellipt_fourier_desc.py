"""Elliptic Fourier Descriptors
parametrizing a closed contour (in red)"""
import numpy as np
import vedo
import pyefd

shapes = vedo.Assembly(vedo.dataurl+'timecourse1d.npy')

s = shapes[55].c('red5').lw(3)
pts1 = s.vertices[:,(0,1)].copy()
pts2 = s.vertices[:,(0,1)].copy()
pts2[:,0] *= -1
pts2 = np.flip(pts2, axis=0)
pts = np.array(pts1.tolist() + pts2.tolist())

rlines = []
for order in range(5,30, 5):
    coeffs = pyefd.elliptic_fourier_descriptors(pts, order=order, normalize=False)
    a0, c0 = pyefd.calculate_dc_coefficients(pts)
    rpts = pyefd.reconstruct_contour(coeffs, locus=(a0,c0), num_points=400)
    color = vedo.color_map(order, "Blues", 5,30)
    rline = vedo.Line(rpts).lw(3).c(color)
    rlines.append(rline)

s.z(0.1) # move it on top so it's visible
vedo.show(s, *rlines, __doc__, axes=1, bg='k', size=(1190, 630), zoom=1.8).close()
