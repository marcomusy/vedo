""" "With four parameters I can fit an elephant,
and with five I can make him wiggle his trunk."
(John von Neumann)

"""
# Original Version
#     Author: Piotr A. Zolnierczuk (zolnierczukp at ornl dot gov)
#     Retrieved on 14 September 2011 from
#     http://www.johndcook.com/blog/2011/06/21/how-to-fit-an-elephant/
# Modified to wiggle trunk:
#     2 October 2011 by David Bailey (http://www.physics.utoronto.ca/~dbailey)
#
# Based on the paper:
#     "Drawing an elephant with four complex parameters", by
#     Jurgen Mayer, Khaled Khairy, and Jonathon Howard,
#     Am. J. Phys. 78, 648 (2010), DOI:10.1119/1.3254017
#
# Inspired by John von Neumann's famous quote (above) about overfitting data.
#     Attributed to von Neumann by Enrico Fermi, as quoted by
#       Freeman Dyson in "A meeting with Enrico Fermi" in
#       Nature 427 (22 January 2004) p. 297
#
# Adapted for vedo by M. Musy, February 2021 from source:
# https://www8.physics.utoronto.ca/~phy326/python/vonNeumann_elephant.py
#
import vedo
import numpy as np

# elephant parameters
p = [ 50 - 30j,
      18 +  8j,
      12 - 10j,
     -14 - 60j,
      40 + 20j ]

def fourier(t, C):
    f = np.zeros(t.shape) # initialize fourier values to zero
    for k in range(len(C)):
        f += C.real[k]*np.cos(k*t) + C.imag[k]*np.sin(k*t)
    return f

def elephant(t, p):
    Cx = np.zeros((6,), dtype='complex')
    Cy = np.zeros((6,), dtype='complex')
    Cx[1], Cy[1] = p[0].real*1j, p[3].imag + p[0].imag*1j
    Cx[2], Cy[2] = p[1].real*1j, p[1].imag*1j
    Cx[3], Cy[3] = p[2].real, p[2].imag*1j
    Cx[5] = p[3].real
    x =  np.append(fourier(t,Cy), [ p[4].imag])
    y = -np.append(fourier(t,Cx), [-p[4].imag])
    return np.array([x,y])


# draw the body of the elephant
ele =  elephant(np.linspace(0.4 +1.3*np.pi, 2*np.pi +0.9*np.pi, 100), p)
body = vedo.Line(ele.T).tomesh().lw(0).c('r4')
plt = vedo.show(body, __doc__+str(p), zoom=0.8, axes=1, interactive=False)
ltrunk = None

# wiggle trunk
#vd = vedo.Video()
for i in range(50):
    trunk = elephant(np.linspace(2*np.pi +0.9*np.pi, 0.4 +3.3*np.pi, 500), p)
    x, y = trunk
    for ii in range(len(y)-1):
        y[ii] -= np.sin(((x[ii]-x[0])*np.pi/len(y)))*np.sin(float(i))*p[4].real
    plt.remove(ltrunk) # remove old trunk before drawing at new position
    ltrunk = vedo.Line(trunk.T).lw(6).c('r3')
    plt.add(ltrunk)
    #vd.addFrame()

#vd.close()
vedo.interactive()
