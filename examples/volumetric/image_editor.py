# !/usr/bin/env python
# -*- coding: utf-8 -*-
# A simple image editor that allows the user to apply various filters
import sys
from vedo import Image, Plotter, Text2D, settings, dataurl


# check if file is given as command line argument
if len(sys.argv) > 1:
    filename = sys.argv[1]
else: # use a default image
    filename = dataurl + "e3_EGFP.jpg"
img0 = Image(filename)
img1 = img0.clone()


def slider_func(widget, e=""):
    value = widget.value
    img2 = img1.clone()
    if widget.is_enabled():
        img2.filterpass(highcutoff=value)
    plt.at(2).remove("Image").add(img2)


def key_func(evt):
    global img1
    if evt.keypress in ["q", "Ctrl+q", "Ctrl+w"]:
        plt.close()
        return
    elif evt.keypress == "Ctrl+e": img1.enhance()
    elif evt.keypress == "Ctrl+m": img1.median()
    elif evt.keypress == "Ctrl+f": img1.mirror()
    elif evt.keypress == "Ctrl+v": img1.flip()
    elif evt.keypress == "Ctrl+t": img1.rotate(90)
    elif evt.keypress == "Ctrl+b": img1.binarize()
    elif evt.keypress == "Ctrl+i": img1.invert()
    elif evt.keypress == "Ctrl+g": img1.smooth(sigma=1)
    elif evt.keypress == "Ctrl+a":
        plt.at(0)
        slider.toggle()
    elif evt.keypress == "Ctrl+r":
        img1 = img0.clone()
        plt.at(1).remove("Image").add(img1)
    elif evt.keypress == "Ctrl+s":
        img1.write("corrected_image.png")
        print("Image saved as: corrected_image.png")
    elif evt.keypress == "Ctrl+o":
        plt.color_picker(evt.picked2d, verbose=True)
    slider_func(slider)
    plt.render()


header = Text2D(
    f"File name:\n {filename[-35:]}\nSize: {img1.shape}",
    c="w",
    bg="k2",
    alpha=1,
    font="Calco",
    s=1.2,
)
instructions = Text2D(
    "Press ---------------------\n"
    "Ctrl+a to toggle FFT filter\n"
    "Ctrl+m to median filter\n"
    "Ctrl+g to gaussian filter\n"
    "Ctrl+e to enhance contrast\n"
    "Ctrl+f to flip horizontally\n"
    "Ctrl+v to flip vertically\n"
    "Ctrl+t to rotate\n"
    "Ctrl+b to binarize\n"
    "Ctrl+i to invert B/W\n"
    "Ctrl+o to pick level\n"
    "Ctrl+r to reset\n"
    "Ctrl+s to save\n"
    "Drag mouse to change contrast \n"
    "q to quit",
    pos=(0.01, 0.8),
    c="w",
    bg="y2",
    alpha=1,
    font="ComicMono",
    s=0.8,
)

settings.enable_default_keyboard_callbacks = False
plt = Plotter(size=(1800, 600), shape=(1, 3), title="vedo - Image Editor")
slider = plt.add_slider(
    slider_func, 0.001, 0.1, 0.02,
    pos=([0.01, 0.1], [0.3, 0.1]), title="FFT Value"
)
slider.off()  # disable slider initially
plt.add_callback("end interaction", lambda _evt: slider_func(slider))
plt.add_callback("key", key_func)
plt.at(0).add(header, instructions)
plt.at(1).add("Input", img1)
plt.at(2).add("Output").freeze()
plt.show(mode="2d")
plt.interactor.RemoveObservers("CharEvent")
slider_func(slider)  # update the slider
plt.reset_camera(0.01).interactive().close()
