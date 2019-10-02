from vtkplotter import histogram, show
import numpy as np

data1 = np.random.randn(500)*3+10
data2 = np.random.randn(500)*2+ 7

h1 = histogram(data1, xtitle='random variable x', ytitle='dN/dx',
               yscale=0.2, fill=True, outline=False, errors=True)
h2 = histogram(data2, 
               yscale=0.2, fill=False, lc='firebrick', lw=4)
h2.z(0.1) # put h2 in front of h1

# pick the 16th bin and color it tomato
h1.getActors()[15].color('tomato').alpha(0.7)

show(h1, h2, bg='white', axes=1)
