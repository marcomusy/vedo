"""Explore RGB and HSV color spaces"""
from vedo import *
from vedo.colors import rgb2hsv, hsv2rgb, rgb2hex

def update_txt(rgb, hsv):
    RGB = np.round(np.array(rgb)*255).astype(int)
    HEX = rgb2hex(rgb)
    name = get_color_name(rgb)
    tx1.text(f"RGB: {precision(rgb, 3)}\n     {RGB}\nHEX: {HEX}")
    tx2.text(f"HSV: {precision(hsv, 3)}\n ~ {name}")
    box.color(rgb)

def func_rgb(w, e):
    rgb = slr.value, slg.value, slb.value
    hsv = rgb2hsv(rgb)
    slh.value, sls.value, slv.value = hsv
    update_txt(rgb, hsv)

def func_hsv(w, e):
    hsv = slh.value, sls.value, slv.value
    rgb = hsv2rgb(hsv)
    slr.value, slg.value, slb.value = rgb
    update_txt(rgb, hsv)

box = Cube().linewidth(2).color([0.5, 0.5, 0.5]).lighting("off")
tx1 = Text2D(font="Calco", s=1.4, pos="top-left",  bg="k5").text(__doc__)
tx2 = Text2D(font="Calco", s=1.4, pos="top-right", bg="k5")

plt = Plotter()

slr = plt.add_slider(func_rgb, 0, 1, value=0.5, show_value=False, c="r3", pos=((0.05,0.18),(0.40,0.18)))
slg = plt.add_slider(func_rgb, 0, 1, value=0.5, show_value=False, c="g3", pos=((0.05,0.12),(0.40,0.12)))
slb = plt.add_slider(func_rgb, 0, 1, value=0.5, show_value=False, c="b3", pos=((0.05,0.06),(0.40,0.06)), title="RGB")

slh = plt.add_slider(func_hsv, 0, 1, value=0.5, show_value=False, c="k1", pos=((0.60,0.18),(0.95,0.18)))
sls = plt.add_slider(func_hsv, 0, 1, value=0.0, show_value=False, c="k1", pos=((0.60,0.12),(0.95,0.12)))
slv = plt.add_slider(func_hsv, 0, 1, value=0.5, show_value=False, c="k1", pos=((0.60,0.06),(0.95,0.06)), title="HSV")

plt.show(box, tx1, tx2, viewup="z").close()

