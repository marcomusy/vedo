from vedo import *

# mode = 1 is maximum projection (default is 0=composite)
v1 = Volume(dataurl+'embryo.tif').mode(1)
v1.addScalarBar3D(c='w').printHistogram(logscale=1, horizontal=1, c='g')
t1 = Text2D('Original volume', c='lg')

# cutoff range is roughly in the range of 1 / size of object
v2 = v1.clone().frequencyPassFilter(highcutoff=.001, order=1).mode(1)
v2.addScalarBar3D(c='w').printHistogram(logscale=1, horizontal=1, c='b')
t2 = Text2D('High freqs in the FFT\nare cut off:', c='lb')

show([(v1,t1), (v2,t2)], N=2, bg='bb', zoom=1.5, axes=dict(digits=2)).close()
#write(v2, 'embryo_filtered.vti')
