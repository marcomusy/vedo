"""Attach a 2D caption to an object"""
from vedo import Cone, Axes, show

cone = Cone().rotateX(30).rotateZ(20).c('steelblue')

txt  = "Japanese\nこれは青い円錐形です\n"
txt += "Chinese\n這是一個藍錐\n"
txt += "Russian\nЭто синий конус\n"
txt += "English\nThis is a blue cone"

cone.caption(txt, size=(0.4,0.3), font="LogoType", c='lb')

axes = Axes(
    cone,
    xtitle='マイクロメートル単位のx軸',
    ytitle='y軸にも長い説明があります',
    ztitle='Z軸始終來自中國',
    titleFont='LogoType',
    textScale=1.5,
    c='white',
)

show(cone, axes, __doc__, viewup='z', bg='k', bg2='bb').close()

