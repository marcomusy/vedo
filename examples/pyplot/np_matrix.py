"""Visualize a numpy matrix"""
from vedo.pyplot import matrix, show
import numpy as np

n, m = (9, 9)
M = np.eye(n,m)/2 + np.random.randn(n,m)*0.1

mat = matrix(M,
             cmap='Reds',
             title='Correlation Matrix',
             xtitle='Genes of group A',
             ytitle='Genes of group B',
             xlabels=['hox1','hox2','hox3','hox4','hox5','hox6','hox7','hox8','hox9'],
             ylabels=['bmp1','bmp2','bmp3','bmp4','bmp5','bmp6','bmp7','bmp8','bmp9'],
             scale=0.16,  # size of bin labels; set it to 0 to remove labels
             lw=2,        # separator line width
            )
show(mat, __doc__, bg='k7', zoom=1.2)
