from vtkplotter import *

print('..this can take ~30 sec / n_cores to run!')

v1 = loadVolume(datadir+'embryo.tif').c('blue')

# cutoff range is roughly in the range of 1 / size of object
v2 = frequencyPassFilter(v1, highcutoff=.001, order=1).c('green')

printHistogram(v1, logscale=1, horizontal=1, c='blue')
printHistogram(v2, logscale=1, horizontal=1, c='green')
t1 = Text('Original volume', c='b')
t2 = Text('High freqs in the FFT\nare cut off:', c='g')

show([(v1,t1), (v2,t2)], N=2, bg='w', zoom=1.5)
#write(v2, 'embryo_filtered.vti')
