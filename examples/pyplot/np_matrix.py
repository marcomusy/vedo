"""Visualize a n\dotm numpy matrix"""
from vedo.pyplot import matrix, show
import numpy as np

n, m = (6, 5)
M = np.eye(n, m)/2 + np.random.randn(n, m)*0.1
# print(M)

mat = matrix(
    M,
    cmap='Reds',
    title='My numpy Matrix',
    xtitle='Genes of group A',
    ytitle='Genes of group B',
    xlabels=[f'hox{i}' for i in range(m)],
    ylabels=[f'bmp{i}' for i in range(n)],
    scale=0.15,  # size of bin labels; set it to 0 to remove labels
    lw=2,        # separator line width
)

show(mat, __doc__, bg='k7', zoom=1.2).close()
