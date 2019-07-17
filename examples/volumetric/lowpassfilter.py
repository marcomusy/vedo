from vtkplotter import *

# mode = 1 is maximum projection (default is 0=composite)
v1 = load(datadir+'embryo.tif').mode(1)
t1 = Text('Original volume', c='lg')

# cutoff range is roughly in the range of 1 / size of object
v2 = frequencyPassFilter(v1, highcutoff=.001, order=1).mode(1)
t2 = Text('High freqs in the FFT\nare cut off:', c='lb')

printHistogram(v1, logscale=1, horizontal=1, c='g')
printHistogram(v2, logscale=1, horizontal=1, c='b')

show([(v1,t1), (v2,t2)], N=2, zoom=1.5)
#write(v2, 'embryo_filtered.vti')
