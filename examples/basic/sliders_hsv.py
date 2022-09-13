"""Explore RGB and HSV color spaces"""
from vedo import Plotter, Cube, Text2D, precision, np
from vedo.colors import rgb2hex, rgb2hsv, hsv2rgb, getColorName

def update(rgb, hsv):
    box.color(rgb)
    RGB = np.round(np.array(rgb)*255).astype(int)
    tx1.text(f"RGB: {precision(rgb,3)}\n     {RGB}\nHEX: {rgb2hex(rgb)}")
    tx2.text(f"HSV: {precision(hsv,3)}\n{getColorName(rgb)}")

def funcRGB(w, e):
    r,g,b = slr.value, slg.value, slb.value
    h,s,v = rgb2hsv([r,g,b])
    slh.value = h
    sls.value = s
    slv.value = v
    update([r,g,b], [h,s,v])

def funcHSV(w, e):
    h,s,v = slh.value, sls.value, slv.value
    r,g,b = hsv2rgb([h,s,v])
    slr.value = r
    slg.value = g
    slb.value = b
    update([r,g,b], [h,s,v])

box = Cube().lw(2).color([1,0,0]).lighting("off")
tx1 = Text2D(font="Calco", s=1.5, pos="top-left",  bg="k5").text(__doc__)
tx2 = Text2D(font="Calco", s=1.5, pos="top-right", bg="k5")

plt = Plotter()

slr = plt.addSlider2D(funcRGB, 0,1, value=1, showValue=False, c="r3", pos=((0.05, 0.18),(0.4, 0.18)))
slg = plt.addSlider2D(funcRGB, 0,1, value=0, showValue=False, c="g3", pos=((0.05, 0.12),(0.4, 0.12)))
slb = plt.addSlider2D(funcRGB, 0,1, value=0, showValue=False, c="b3", pos=((0.05, 0.06),(0.4, 0.06)), title="RGB")

slh = plt.addSlider2D(funcHSV, 0,1, value=0, showValue=False, c="k1", pos=((0.6, 0.18),(0.95, 0.18)))
sls = plt.addSlider2D(funcHSV, 0,1, value=1, showValue=False, c="k1", pos=((0.6, 0.12),(0.95, 0.12)))
slv = plt.addSlider2D(funcHSV, 0,1, value=1, showValue=False, c="k1", pos=((0.6, 0.06),(0.95, 0.06)), title="HSV")

plt.show(box, tx1, tx2, viewup="z")
