from vedo import Points, Axes, show
import numpy as np

pts = np.random.randn(2000,3)*[3,2,4]-[1,2,3]
vpts1 = Points(pts).alpha(0.2).c('blue2')
vpts2 = vpts1.clone().shift(5,6,7).c('green2')

axs = Axes([vpts1, vpts2],  # build axes for this set of objects
           xtitle="X-axis in \mum",
           ytitle="Variable Y in \mum",
           ztitle="Inverted Z in \mum",
           htitle='My \Gamma^2_ijk  plot',
           hTitleFont='Kanopus',
           hTitleJustify='bottom-right',
           hTitleColor='red2',
           hTitleSize=0.035,
           hTitleOffset=(0,0.075,0),
           hTitleRotation=45,
           zHighlightZero=True,
           xyFrameLine=2, yzFrameLine=1, zxFrameLine=1,
           xyFrameColor='red3',
           xyShift=1.05, # move xy 5% above the top of z-range
           yzGrid=True,
           zxGrid=True,
           zxShift=1.0,
           xTitleJustify='bottom-right',
           xTitleOffset=-1.175,
           xLabelOffset=-1.75,
           yLabelRotation=90,
           zInverted=True,
           tipSize=0.25,
)

show(vpts1, vpts2, axs, "Customizing Axes", viewup='z').close()
