from vedo import *

# mode = 1 is maximum projection (default is 0=composite)
t1 = Text2D('Original volume', c='lg')
v1 = Volume(dataurl+'embryo.tif').mode(1)
v1.add_scalarbar3d(c='w')
v1.scalarbar = v1.scalarbar.clone2d("top-right", 0.08)

# cutoff range is roughly in the range of 1 / size of object
t2 = Text2D('High freqs in the FFT are cut off', c='lb')
v2 = v1.clone().frequency_pass_filter(high_cutoff=.001, order=1).mode(1)
v2.add_scalarbar3d(c='w')
v2.scalarbar = v2.scalarbar.clone2d("top-right", 0.08)

show([(v1,t1), (v2,t2)], N=2, bg='bb', zoom=1.5, axes=dict(digits=2)).close()
#write(v2, 'embryo_filtered.vti')
