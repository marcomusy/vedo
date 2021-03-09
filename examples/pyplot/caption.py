"""Attach a 2D caption to an object"""
from vedo import *

cone = Cone().rotateX(30).c('steelblue')

txt  = "Japanese\nこれは青い円錐形です\n"
txt += "Chinese\n這是一個藍錐\n"
txt += "Russian\nЭто синий конус\n"
txt += "English\nThis is a blue cone"
cone.caption(txt, size=(0.4,0.3), font="LogoType", c='lb')

# download the polygonized version of the font (19MB) on the fly
# (actual downloading only happens once, file is cached in /tmp area)
show(cone, __doc__, viewup='z', bg='k', bg2='bb',
     axes=dict(xtitle='マイクロメートル単位のx軸',
               ytitle='y軸にも長い説明があります',
               ztitle='Z軸始終來自中國',
               titleFont='LogoType',
               textScale=1.5,
              ),
    )
