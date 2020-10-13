"""Whisker plot with quantiles indication
(horizontal line shows the mean value)"""
from vedo.pyplot import whisker
from vedo import show, buildAxes
import numpy as np

# create 5 whisker bars with some random data
ws = []
for i in range(5):
    xval = i*2
    data = np.random.randn(25) + i/2
    w = whisker(data, bc=i, s=0.5).x(xval)
    ws.append(w)

# build custom axes
ax = buildAxes(xrange=[-1,9],
               yrange=[-3,5],
               htitle='\beta_c -expression: change in time',
               xtitle=' ',
               ytitle='Level of \beta_c  protein in \muM/l',
               xValuesAndLabels=[(0,'experiment^A\nat t=1h'),
                                 (4,'experiment^B\nat t=2h'),
                                 (8,'experiment^C\nat t=4h'),
                                ],
               xLabelSize=0.02,
               )

show(ws, ax, __doc__)
# print('whisker0:', ws[0].info)


