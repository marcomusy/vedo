from vedo import *

# mode = 1 is maximum projection (default is 0=composite)
v1 = load(datadir+'embryo.tif').mode(1)
t1 = Text2D('Original volume', c='lg')

# cutoff range is roughly in the range of 1 / size of object
v2 = v1.clone().frequencyPassFilter(highcutoff=.001, order=1).mode(1)
t2 = Text2D('High freqs in the FFT\nare cut off:', c='lb')

printHistogram(v1, logscale=1, horizontal=1, c='g')
printHistogram(v2, logscale=1, horizontal=1, c='b')
v1.addScalarBar3D(c='w')
v2.addScalarBar3D(c='w')

show([(v1,t1), (v2,t2)], N=2, bg='bb', zoom=1.5, axes=dict(digits=2))
#write(v2, 'embryo_filtered.vti')
