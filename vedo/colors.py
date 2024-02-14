#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time

import numpy as np
import vedo.vtkclasses as vtki
import vedo

__docformat__ = "google"

__doc__ = """
Colors definitions and printing methods.

![](https://vedo.embl.es/images/basic/colorcubes.png)
"""

__all__ = [
    "printc",
    "printd",
    "get_color",
    "get_color_name",
    "color_map",
    "build_palette",
    "build_lut",
]


try:
    import matplotlib
    _has_matplotlib = True
    cmaps = {}
except ModuleNotFoundError:
    from vedo.cmaps import cmaps
    _has_matplotlib = False
    # see below, this is dealt with in color_map()

#########################################################
# handy global shortcuts for terminal printing
# Ex.: print(colors.red + "hello" + colors.reset)
#########################################################
red = "\x1b[1m\x1b[31;1m"
green = "\x1b[1m\x1b[32;1m"
yellow = "\x1b[1m\x1b[33;1m"
blue = "\x1b[1m\x1b[34;1m"
reset = "\x1b[0m"


#########################################################
# basic color schemes
#########################################################
colors = {
    # order kind of matters because of pyplot.plot()
    "blue9": "#a8cbfe",  # bootstrap5 colors
    "blue8": "#81b4fe",
    "blue7": "#5a9cfe",
    "blue6": "#3485fd",
    "blue5": "#0d6efd",
    "blue4": "#0b5cd5",
    "blue3": "#094bac",
    "blue2": "#073984",
    "blue1": "#05285b",
    "indigo9": "#c8a9fa",
    "indigo8": "#af83f8",
    "indigo7": "#975cf6",
    "indigo6": "#7e36f4",
    "indigo5": "#6610f2",
    "indigo4": "#560dcb",
    "indigo3": "#450ba5",
    "indigo2": "#35087e",
    "indigo1": "#250657",
    "purple9": "#cbbbe9",
    "purple8": "#b49ddf",
    "purple7": "#9d7ed5",
    "purple6": "#8660cb",
    "purple5": "#6f42c1",
    "purple4": "#5d37a2",
    "purple3": "#4b2d83",
    "purple2": "#3a2264",
    "purple1": "#281845",
    "pink9": "#f0b6d3",
    "pink8": "#ea95bf",
    "pink7": "#e374ab",
    "pink6": "#dd5498",
    "pink5": "#d63384",
    "pink4": "#b42b6f",
    "pink3": "#92235a",
    "pink2": "#6f1b45",
    "pink1": "#4d1230",
    "red9": "#f2b6bc",
    "red8": "#ed969e",
    "red7": "#e77681",
    "red6": "#e25563",
    "red5": "#dc3545",
    "red4": "#b92d3a",
    "red3": "#96242f",
    "red2": "#721c24",
    "red1": "#4f1319",
    "orange9": "#fed1aa",
    "orange8": "#febc85",
    "orange7": "#fea75f",
    "orange6": "#fd933a",
    "orange5": "#fd7e14",
    "orange4": "#d56a11",
    "orange3": "#ac560e",
    "orange2": "#84420a",
    "orange1": "#5b2d07",
    "yellow9": "#ffe9a6",
    "yellow8": "#ffdf7e",
    "yellow7": "#ffd556",
    "yellow6": "#ffcb2f",
    "yellow5": "#ffc107",
    "yellow4": "#d6a206",
    "yellow3": "#ad8305",
    "yellow2": "#856404",
    "yellow1": "#5c4503",
    "green9": "#b2dfbc",
    "green8": "#8fd19e",
    "green7": "#6dc381",
    "green6": "#4ab563",
    "green5": "#28a745",
    "green4": "#228c3a",
    "green3": "#1b722f",
    "green2": "#155724",
    "green1": "#0e3c19",
    "teal9": "#afecda",
    "teal8": "#8be3c9",
    "teal7": "#67dab8",
    "teal6": "#44d2a8",
    "teal5": "#20c997",
    "teal4": "#1ba97f",
    "teal3": "#168967",
    "teal2": "#11694f",
    "teal1": "#0c4836",
    "cyan9": "#abdee5",
    "cyan8": "#86cfda",
    "cyan7": "#61c0cf",
    "cyan6": "#3cb1c3",
    "cyan5": "#17a2b8",
    "cyan4": "#13889b",
    "cyan3": "#106e7d",
    "cyan2": "#0c5460",
    "cyan1": "#083a42",
    "gray9": "#f8f9fa",
    "gray8": "#e9edef",
    "gray7": "#dee2e6",
    "gray6": "#ced4da",
    "gray5": "#adb5bd",
    "gray4": "#6c757d",
    "gray3": "#495057",
    "gray2": "#343a40",
    "gray1": "#212529",
    "aliceblue": "#F0F8FF",  # matplotlib scheme
    "antiquewhite": "#FAEBD7",
    "aqua": "#00FFFF",
    "aquamarine": "#7FFFD4",
    "azure": "#F0FFFF",
    "beige": "#F5F5DC",
    "bisque": "#FFE4C4",
    "black": "#000000",
    "blanchedalmond": "#FFEBCD",
    "blue": "#0f00fb",  # "0000FF",
    "blueviolet": "#8A2BE2",
    "brown": "#A52A2A",
    "burlywood": "#DEB887",
    "cadetblue": "#5F9EA0",
    "chartreuse": "#7FFF00",
    "chocolate": "#D2691E",
    "coral": "#FF7F50",
    "cornflowerblue": "#6495ED",
    "cornsilk": "#FFF8DC",
    "crimson": "#DC143C",
    "cyan": "#00FFFF",
    "darkblue": "#00008B",
    "darkcyan": "#008B8B",
    "darkgoldenrod": "#B8860B",
    "darkgray": "#A9A9A9",
    "darkgreen": "#006400",
    "darkkhaki": "#BDB76B",
    "darkmagenta": "#8B008B",
    "darkolivegreen": "#556B2F",
    "darkorange": "#FF8C00",
    "darkorchid": "#9932CC",
    "darkred": "#8B0000",
    "darksalmon": "#E9967A",
    "darkseagreen": "#8FBC8F",
    "darkslateblue": "#483D8B",
    "darkslategray": "#2F4F4F",
    "darkturquoise": "#00CED1",
    "darkviolet": "#9400D3",
    "deeppink": "#FF1493",
    "deepskyblue": "#00BFFF",
    "dimgray": "#696969",
    "dodgerblue": "#1E90FF",
    "firebrick": "#B22222",
    "floralwhite": "#FFFAF0",
    "forestgreen": "#228B22",
    "fuchsia": "#FF00FF",
    "gainsboro": "#DCDCDC",
    "ghostwhite": "#F8F8FF",
    "gold": "#FFD700",
    "goldenrod": "#DAA520",
    "gray": "#808080",
    "green": "#047f10",  # "#008000",
    "greenyellow": "#ADFF2F",
    "honeydew": "#F0FFF0",
    "hotpink": "#FF69B4",
    "indianred": "#CD5C5C",
    "indigo": "#4B0082",
    "ivory": "#FFFFF0",
    "khaki": "#F0E68C",
    "lavender": "#E6E6FA",
    "lavenderblush": "#FFF0F5",
    "lawngreen": "#7CFC00",
    "lemonchiffon": "#FFFACD",
    "lightblue": "#ADD8E6",
    "lightcoral": "#F08080",
    "lightcyan": "#E0FFFF",
    "lightgray": "#D3D3D3",
    "lightgreen": "#90EE90",
    "lightpink": "#FFB6C1",
    "lightsalmon": "#FFA07A",
    "lightseagreen": "#20B2AA",
    "lightskyblue": "#87CEFA",
    "lightsteelblue": "#B0C4DE",
    "lightyellow": "#FFFFE0",
    "lime": "#00FF00",
    "limegreen": "#32CD32",
    "linen": "#FAF0E6",
    "magenta": "#FF00FF",
    "maroon": "#800000",
    "mediumaquamarine": "#66CDAA",
    "mediumblue": "#0000CD",
    "mediumorchid": "#BA55D3",
    "mediumpurple": "#9370DB",
    "mediumseagreen": "#3CB371",
    "mediumslateblue": "#7B68EE",
    "mediumspringgreen": "#00FA9A",
    "mediumturquoise": "#48D1CC",
    "mediumvioletred": "#C71585",
    "midnightblue": "#191970",
    "mintcream": "#F5FFFA",
    "mistyrose": "#FFE4E1",
    "moccasin": "#FFE4B5",
    "navajowhite": "#FFDEAD",
    "navy": "#000080",
    "oldlace": "#FDF5E6",
    "olive": "#808000",
    "olivedrab": "#6B8E23",
    "orange": "#FFA500",
    "orangered": "#FF4500",
    "orchid": "#DA70D6",
    "palegoldenrod": "#EEE8AA",
    "palegreen": "#98FB98",
    "paleturquoise": "#AFEEEE",
    "palevioletred": "#DB7093",
    "papayawhip": "#FFEFD5",
    "peachpuff": "#FFDAB9",
    "peru": "#CD853F",
    "pink": "#FFC0CB",
    "plum": "#DDA0DD",
    "powderblue": "#B0E0E6",
    "purple": "#800080",
    "rebeccapurple": "#663399",
    "red": "#fe1e1f",  # "#FF0000",
    "rosybrown": "#BC8F8F",
    "royalblue": "#4169E1",
    "saddlebrown": "#8B4513",
    "salmon": "#FA8072",
    "sandybrown": "#F4A460",
    "seagreen": "#2E8B57",
    "seashell": "#FFF5EE",
    "sienna": "#A0522D",
    "silver": "#C0C0C0",
    "skyblue": "#87CEEB",
    "slateblue": "#6A5ACD",
    "slategray": "#708090",
    "snow": "#FFFAFA",
    "blackboard": "#393939",
    "springgreen": "#00FF7F",
    "steelblue": "#4682B4",
    "tan": "#D2B48C",
    "teal": "#008080",
    "thistle": "#D8BFD8",
    "tomato": "#FF6347",
    "turquoise": "#40E0D0",
    "violet": "#EE82EE",
    "wheat": "#F5DEB3",
    "white": "#FFFFFF",
    "whitesmoke": "#F5F5F5",
    "yellow": "#ffff36",  # "#FFFF00",
    "yellowgreen": "#9ACD32",
}


color_nicks = {  # color nicknames
    "bb": "blackboard",
    "lb": "lightblue",  # light
    "lg": "lightgreen",
    "lr": "orangered",
    "lc": "lightcyan",
    "ls": "lightsalmon",
    "ly": "lightyellow",
    "dr": "darkred",  # dark
    "db": "darkblue",
    "dg": "darkgreen",
    "dm": "darkmagenta",
    "dc": "darkcyan",
    "ds": "darksalmon",
    "dv": "darkviolet",
    "b1": "blue1",  # bootstrap5 colors
    "b2": "blue2",
    "b3": "blue3",
    "b4": "blue4",
    "b5": "blue5",
    "b6": "blue6",
    "b7": "blue7",
    "b8": "blue8",
    "b9": "blue9",
    "i1": "indigo1",
    "i2": "indigo2",
    "i3": "indigo3",
    "i4": "indigo4",
    "i5": "indigo5",
    "i6": "indigo6",
    "i7": "indigo7",
    "i8": "indigo8",
    "i9": "indigo9",
    "p1": "purple1",
    "p2": "purple2",
    "p3": "purple3",
    "p4": "purple4",
    "p5": "purple5",
    "p6": "purple6",
    "p7": "purple7",
    "p8": "purple8",
    "p9": "purple9",
    "r1": "red1",
    "r2": "red2",
    "r3": "red3",
    "r4": "red4",
    "r5": "red5",
    "r6": "red6",
    "r7": "red7",
    "r8": "red8",
    "r9": "red9",
    "o1": "orange1",
    "o2": "orange2",
    "o3": "orange3",
    "o4": "orange4",
    "o5": "orange5",
    "o6": "orange6",
    "o7": "orange7",
    "o8": "orange8",
    "o9": "orange9",
    "y1": "yellow1",
    "y2": "yellow2",
    "y3": "yellow3",
    "y4": "yellow4",
    "y5": "yellow5",
    "y6": "yellow6",
    "y7": "yellow7",
    "y8": "yellow8",
    "y9": "yellow9",
    "g1": "green1",
    "g2": "green2",
    "g3": "green3",
    "g4": "green4",
    "g5": "green5",
    "g6": "green6",
    "g7": "green7",
    "g8": "green8",
    "g9": "green9",
    "k1": "gray1",
    "k2": "gray2",
    "k3": "gray3",
    "k4": "gray4",
    "k5": "gray5",
    "k6": "gray6",
    "k7": "gray7",
    "k8": "gray8",
    "k9": "gray9",
    "a": "aqua",
    "b": "blue",
    "c": "cyan",
    "d": "gold",
    "f": "fuchsia",
    "g": "green",
    "i": "indigo",
    "k": "black",
    "m": "magenta",
    "n": "navy",
    "l": "lavender",
    "o": "orange",
    "p": "purple",
    "r": "red",
    "s": "salmon",
    "t": "tomato",
    "v": "violet",
    "y": "yellow",
    "w": "white",
}


# available colormap names:
cmaps_names = (
    "Accent",
    "Accent_r",
    "Blues",
    "Blues_r",
    "BrBG",
    "BrBG_r",
    "BuGn",
    "BuGn_r",
    "BuPu",
    "BuPu_r",
    "CMRmap",
    "CMRmap_r",
    "Dark2",
    "Dark2_r",
    "GnBu",
    "GnBu_r",
    "Greens",
    "Greens_r",
    "Greys",
    "Greys_r",
    "OrRd",
    "OrRd_r",
    "Oranges",
    "Oranges_r",
    "PRGn",
    "PRGn_r",
    "Paired",
    "Paired_r",
    "Pastel1",
    "Pastel1_r",
    "Pastel2",
    "Pastel2_r",
    "PiYG",
    "PiYG_r",
    "PuBu",
    "PuBuGn",
    "PuBuGn_r",
    "PuBu_r",
    "PuOr",
    "PuOr_r",
    "PuRd",
    "PuRd_r",
    "Purples",
    "Purples_r",
    "RdBu",
    "RdBu_r",
    "RdGy",
    "RdGy_r",
    "RdPu",
    "RdPu_r",
    "RdYlBu",
    "RdYlBu_r",
    "RdYlGn",
    "RdYlGn_r",
    "Reds",
    "Reds_r",
    "Set1",
    "Set1_r",
    "Set2",
    "Set2_r",
    "Set3",
    "Set3_r",
    "Spectral",
    "Spectral_r",
    "Wistia",
    "Wistia_r",
    "YlGn",
    "YlGnBu",
    "YlGnBu_r",
    "YlGn_r",
    "YlOrBr",
    "YlOrBr_r",
    "YlOrRd",
    "YlOrRd_r",
    "afmhot",
    "afmhot_r",
    "autumn",
    "autumn_r",
    "binary",
    "binary_r",
    "bone",
    "bone_r",
    "brg",
    "brg_r",
    "bwr",
    "bwr_r",
    "cividis",
    "cividis_r",
    "cool",
    "cool_r",
    "coolwarm",
    "coolwarm_r",
    "copper",
    "copper_r",
    "cubehelix",
    "cubehelix_r",
    "flag",
    "flag_r",
    "gist_earth",
    "gist_earth_r",
    "gist_gray",
    "gist_gray_r",
    "gist_heat",
    "gist_heat_r",
    "gist_ncar",
    "gist_ncar_r",
    "gist_rainbow",
    "gist_rainbow_r",
    "gist_stern",
    "gist_stern_r",
    "gist_yarg",
    "gist_yarg_r",
    "gnuplot",
    "gnuplot2",
    "gnuplot2_r",
    "gnuplot_r",
    "gray_r",
    "hot",
    "hot_r",
    "hsv",
    "hsv_r",
    "inferno",
    "inferno_r",
    "jet",
    "jet_r",
    "magma",
    "magma_r",
    "nipy_spectral",
    "nipy_spectral_r",
    "ocean",
    "ocean_r",
    "pink_r",
    "plasma",
    "plasma_r",
    "prism",
    "prism_r",
    "rainbow",
    "rainbow_r",
    "seismic",
    "seismic_r",
    "spring",
    "spring_r",
    "summer",
    "summer_r",
    "tab10",
    "tab10_r",
    "tab20",
    "tab20_r",
    "tab20b",
    "tab20b_r",
    "tab20c",
    "tab20c_r",
    "terrain",
    "terrain_r",
    "twilight",
    "twilight_r",
    "twilight_shifted",
    "twilight_shifted_r",
    "viridis",
    "viridis_r",
    "winter",
    "winter_r",
)


# default color palettes when using an index
palettes = (
    (
       [1.        , 0.75686275, 0.02745098], # yellow5
       [0.99215686, 0.49411765, 0.07843137], # orange5
       [0.8627451 , 0.20784314, 0.27058824], # red5
       [0.83921569, 0.2       , 0.51764706], # pink5
       [0.1254902 , 0.78823529, 0.59215686], # teal5
       [0.15686275, 0.65490196, 0.27058824], # green5
       [0.09019608, 0.63529412, 0.72156863], # cyan5
       [0.05098039, 0.43137255, 0.99215686], # blue5
       [0.4       , 0.0627451 , 0.94901961], # indigo5
       [0.67843137, 0.70980392, 0.74117647], # gray5
    ),
    (
        (1.0, 0.832, 0.000),  # gold
        (0.960, 0.509, 0.188),
        (0.901, 0.098, 0.194),
        (0.235, 0.85, 0.294),
        (0.46, 0.48, 0.000),
        (0.274, 0.941, 0.941),
        (0.0, 0.509, 0.784),
        (0.1, 0.1, 0.900),
        (0.902, 0.7, 1.000),
        (0.941, 0.196, 0.901),
    ),
    (
        (1.0, 0.832, 0),    # gold
        (0.59, 0.0, 0.09),  # dark red
        (0.5, 0.5, 0),      # yellow-green
        (0.0, 0.66, 0.42),  # green blue
        (0.5, 1.0, 0.0),    # green
        (0.0, 0.18, 0.65),  # blue
        (0.4, 0.0, 0.4),    # plum
        (0.4, 0.0, 0.6),
        (0.2, 0.4, 0.6),
        (0.1, 0.3, 0.2),
    ),
    (
        (0.010, 0.0706, 0.098),  #  -> black
        (0.0196, 0.369, 0.447),
        (0.0745, 0.573, 0.584),
        (0.584, 0.820, 0.741),
        (0.914, 0.847, 0.663),
        (0.929, 0.616, 0.149),
        (0.788, 0.412, 0.110),
        (0.729, 0.259, 0.0902),
        (0.678, 0.153, 0.110),
        (0.604, 0.153, 0.165),  #  -> red3
    ),
    (
        (0.345, 0.188, 0.071),  #  -> orange1
        (0.498, 0.314, 0.161),
        (0.573, 0.404, 0.239),
        (0.651, 0.545, 0.400),
        (0.714, 0.678, 0.569),
        (0.761, 0.773, 0.671),
        (0.643, 0.675, 0.533),
        (0.396, 0.427, 0.298),
        (0.255, 0.282, 0.204),
        (0.200, 0.239, 0.165),  #  -> blackboard
    ),
    (
        (0.937, 0.969, 0.820),  #  -> beige
        (0.729, 0.851, 0.714),
        (0.671, 0.639, 0.396),
        (0.447, 0.180, 0.180),
        (0.259, 0.055, 0.082),  #  -> red1
        (0.937, 0.969, 0.820),  #  -> beige
        (0.729, 0.851, 0.714),
        (0.671, 0.639, 0.396),
        (0.447, 0.180, 0.180),
        (0.259, 0.055, 0.082),  #  -> red1
    ),
    (
        (0.933, 0.298, 0.443),  #  -> red6
        (0.996, 0.824, 0.431),
        (0.082, 0.835, 0.631),
        (0.094, 0.537, 0.690),
        (0.035, 0.231, 0.294),  #  -> cyan1
        (0.933, 0.298, 0.443),  #  -> red6
        (0.996, 0.824, 0.431),
        (0.082, 0.835, 0.631),
        (0.094, 0.537, 0.690),
        (0.035, 0.231, 0.294),  #  -> cyan1
    ),
)


emoji = {
    ":bomb:": "\U0001F4A5",
    ":sparks:": "\U00002728",
    ":thumbup:": "\U0001F44D",
    ":target:": "\U0001F3AF",
    ":save:": "\U0001F4BE",
    ":noentry:": "\U000026D4",
    ":video:": "\U0001F4FD",
    ":lightning:": "\U000026A1",
    ":camera:": "\U0001F4F8",
    ":times:": "\U0000274C",
    ":world:": "\U0001F30D",
    ":rainbow:": "\U0001F308",
    ":idea:": "\U0001F4A1",
    ":pin:": "\U0001F4CC",
    ":construction:": "\U0001F6A7",
    ":rocket:": "\U0001F680",
    ":hourglass:": "\U000023F3",
    ":prohibited:": "\U0001F6AB",
    ":checked:": "\U00002705",
    ":smile:": "\U0001F642",
    ":sad:": "\U0001F612",
    ":star:": "\U00002B50",
    ":zzz:": "\U0001F4A4",
    ":mu:": "\U000003BC",
    ":pi:": "\U000003C0",
    ":sigma:": "\U000003C3",
    ":rightarrow:": "\U000027A1",
}


# terminal or notebook can do color print
def _has_colors(stream):
    try:
        import IPython
        return True
    except:
        pass

    if not hasattr(stream, "isatty"):
        return False
    if not stream.isatty():
        return False
    return True
_terminal_has_colors = _has_colors(sys.stdout)


def _is_sequence(arg):
    # Check if input is iterable.
    if hasattr(arg, "strip"):
        return False
    if hasattr(arg, "__getslice__"):
        return True
    if hasattr(arg, "__iter__"):
        return True
    return False


def get_color(rgb=None, hsv=None):
    """
    Convert a color or list of colors to (r,g,b) format from many different input formats.

    Set `hsv` to input as (hue, saturation, value).

    Example:
         - `RGB    = (255, 255, 255)` corresponds to white
         - `rgb    = (1,1,1)` is again white
         - `hex    = #FFFF00` is yellow
         - `string = 'white'`
         - `string = 'w'` is white nickname
         - `string = 'dr'` is darkred
         - `string = 'red4'` is a shade of red
         - `int    =  7` picks color nr. 7 in a predefined color list
         - `int    = -7` picks color nr. 7 in a different predefined list


    Examples:
        - [colorcubes.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/colorcubes.py)

            ![](https://vedo.embl.es/images/basic/colorcubes.png)
    """
    # recursion, return a list if input is list of colors:
    if _is_sequence(rgb) and (len(rgb) > 3 or _is_sequence(rgb[0])):
        seqcol = []
        for sc in rgb:
            seqcol.append(get_color(sc))
        return seqcol

    # because they are most common:
    if isinstance(rgb, str):
        if rgb == "r":
            return (0.9960784313725, 0.11764705882352, 0.121568627450980)
        elif rgb == "g":
            return (0.0156862745098, 0.49803921568627, 0.062745098039215)
        elif rgb == "b":
            return (0.0588235294117, 0.0, 0.984313725490196)

    if str(rgb).isdigit():
        rgb = int(rgb)

    if hsv:
        c = hsv2rgb(hsv)
    else:
        c = rgb

    if _is_sequence(c):
        if c[0] <= 1 and c[1] <= 1 and c[2] <= 1:
            return c  # already rgb
        if len(c) == 3:
            return list(np.array(c) / 255.0)  # RGB
        return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0, c[3])  # RGBA

    elif isinstance(c, str):  # is string
        c = c.replace("grey", "gray").replace(" ", "")
        if 0 < len(c) < 3:  # single/double letter color
            if c.lower() in color_nicks:
                c = color_nicks[c.lower()]
            else:
                # vedo.logger.warning(
                #     f"Unknown color nickname {c}\nAvailable abbreviations: {color_nicks}"
                # )
                return (0.5, 0.5, 0.5)

        if c.lower() in colors:  # matplotlib name color
            c = colors[c.lower()]
            # from now format is hex!

        if c.startswith("#"):  # hex to rgb
            h = c.lstrip("#")
            rgb255 = list(int(h[i : i + 2], 16) for i in (0, 2, 4))
            rgbh = np.array(rgb255) / 255.0
            if np.sum(rgbh) > 3:
                vedo.logger.error(f"in get_color(): Wrong hex color {c}")
                return (0.5, 0.5, 0.5)
            return tuple(rgbh)

        else:  # vtk name color
            namedColors = vtki.new("NamedColors")
            rgba = [0, 0, 0, 0]
            namedColors.GetColor(c, rgba)
            return (rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0)

    elif isinstance(c, (int, float)):  # color number
        return palettes[vedo.settings.palette % len(palettes)][abs(int(c)) % 10]

    return (0.5, 0.5, 0.5)


def get_color_name(c):
    """Find the name of the closest color."""
    c = np.array(get_color(c))  # reformat to rgb
    mdist = 99.0
    kclosest = ""
    for key in colors:
        ci = np.array(get_color(key))
        d = np.linalg.norm(c - ci)
        if d < mdist:
            mdist = d
            kclosest = str(key)
    return kclosest


def hsv2rgb(hsv):
    """Convert HSV to RGB color."""
    ma = vtki.new("Math")
    rgb = [0, 0, 0]
    ma.HSVToRGB(hsv, rgb)
    return rgb


def rgb2hsv(rgb):
    """Convert RGB to HSV color."""
    ma = vtki.new("Math")
    hsv = [0, 0, 0]
    ma.RGBToHSV(get_color(rgb), hsv)
    return hsv


def rgb2hex(rgb):
    """Convert RGB to Hex color."""
    h = "#%02x%02x%02x" % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    return h


def hex2rgb(hx):
    """Convert Hex to rgb color."""
    h = hx.lstrip("#")
    rgb255 = [int(h[i : i + 2], 16) for i in (0, 2, 4)]
    return (rgb255[0] / 255.0, rgb255[1] / 255.0, rgb255[2] / 255.0)


def color_map(value, name="jet", vmin=None, vmax=None):
    """
    Map a real value in range [vmin, vmax] to a (r,g,b) color scale.

    Return the (r,g,b) color, or a list of (r,g,b) colors.

    Arguments:
        value : (float, list)
            scalar value to transform into a color
        name : (str, matplotlib.colors.LinearSegmentedColormap)
            color map name

    Very frequently used color maps are:

        ![](https://user-images.githubusercontent.com/32848391/50738804-577e1680-11d8-11e9-929e-fca17a8ac6f3.jpg)

    A more complete color maps list:

        ![](https://matplotlib.org/1.2.1/_images/show_colormaps.png)

    .. note:: Can also directly use and customize a matplotlib color map

    Example:
        ```python
        import matplotlib
        from vedo import color_map
        rgb = color_map(0.2, matplotlib.colormaps["jet"], 0, 1)
        print("rgb =", rgb)  # [0.0, 0.3, 1.0]
        ```

    Examples:
        - [plot_bars.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_bars.py)

            <img src="https://vedo.embl.es/images/pyplot/plot_bars.png" width="400"/>

    """
    cut = _is_sequence(value)  # to speed up later

    if cut:
        values = np.asarray(value)
        if vmin is None:
            vmin = np.min(values)
        if vmax is None:
            vmax = np.max(values)
        values = np.clip(values, vmin, vmax)
        values = (values - vmin) / (vmax - vmin)
    else:
        if vmin is None:
            vedo.logger.warning("in color_map() you must specify vmin! Assume 0.")
            vmin = 0
        if vmax is None:
            vedo.logger.warning("in color_map() you must specify vmax! Assume 1.")
            vmax = 1
        if vmax == vmin:
            values = [value - vmin]
        else:
            values = [(value - vmin) / (vmax - vmin)]

    if _has_matplotlib:
        # matplotlib is available, use it! ###########################
        if isinstance(name, str):
            mp = matplotlib.colormaps[name]
        else:
            mp = name  # assume matplotlib.colors.LinearSegmentedColormap
        result = mp(values)[:, [0, 1, 2]]

    else:
        # matplotlib not available ###################################
        invert = False
        if name.endswith("_r"):
            invert = True
            name = name.replace("_r", "")
        try:
            cmap = cmaps[name]
        except KeyError:
            vedo.logger.error(f"in color_map(), no color map with name {name} or {name}_r")
            vedo.logger.error(f"Available color maps are:\n{cmaps.keys()}")
            return np.array([0.5, 0.5, 0.5])

        result = []
        n = len(cmap) - 1
        for v in values:
            iv = int(v * n)
            if invert:
                iv = n - iv
            rgb = hex2rgb(cmap[iv])
            result.append(rgb)
        result = np.array(result)

    if cut:
        return result
    return result[0]


def build_palette(color1, color2, n, hsv=True):
    """
    Generate N colors starting from `color1` to `color2`
    by linear interpolation in HSV or RGB spaces.

    Arguments:
        N : (int)
            number of output colors.
        color1 : (color)
            first color.
        color2 : (color)
            second color.
        hsv : (bool)
            if `False`, interpolation is calculated in RGB space.

    Examples:
        - [mesh_custom.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_custom.py)

            ![](https://vedo.embl.es/images/basic/mesh_custom.png)
    """
    if hsv:
        color1 = rgb2hsv(color1)
        color2 = rgb2hsv(color2)
    c1 = np.array(get_color(color1))
    c2 = np.array(get_color(color2))
    cols = []
    for f in np.linspace(0, 1, n, endpoint=True):
        c = c1 * (1 - f) + c2 * f
        if hsv:
            c = np.array(hsv2rgb(c))
        cols.append(c)
    return np.array(cols)


def build_lut(
    colorlist,
    vmin=None,
    vmax=None,
    below_color=None,
    above_color=None,
    nan_color=None,
    below_alpha=1,
    above_alpha=1,
    nan_alpha=1,
    interpolate=False,
):
    """
    Generate colors in a lookup table (LUT).

    Return the `vtkLookupTable` object. This can be fed into `cmap()` method.

    Arguments:
        colorlist : (list)
            a list in the form `[(scalar1, [r,g,b]), (scalar2, 'blue'), ...]`.
        vmin : (float)
            specify minimum value of scalar range
        vmax : (float)
            specify maximum value of scalar range
        below_color : (color)
            color for scalars below the minimum in range
        below_alpha : (float)
            opacity for scalars below the minimum in range
        above_color : (color)
            color for scalars above the maximum in range
        above_alpha : (float)
            alpha for scalars above the maximum in range
        nan_color : (color)
            color for invalid (nan) scalars
        nan_alpha : (float)
            alpha for invalid (nan) scalars
        interpolate : (bool)
            interpolate or not intermediate scalars

    Examples:
        - [mesh_lut.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_lut.py)

            ![](https://vedo.embl.es/images/basic/mesh_lut.png)
    """
    ctf = vtki.new("ColorTransferFunction")
    ctf.SetColorSpaceToRGB()
    ctf.SetScaleToLinear()

    alpha_x, alpha_vals = [], []
    for sc in colorlist:
        if len(sc) >= 3:
            scalar, col, alf = sc[:3]
        else:
            alf = 1
            scalar, col = sc
        r, g, b = get_color(col)
        ctf.AddRGBPoint(scalar, r, g, b)
        alpha_x.append(scalar)
        alpha_vals.append(alf)

    ncols = 8 * len(colorlist)
    if not interpolate:
        ncols = len(colorlist)

    lut = vtki.new("LookupTable")
    lut.SetNumberOfTableValues(ncols)

    x0, x1 = ctf.GetRange()  # range of the introduced values
    if vmin is not None:
        x0 = vmin
    if vmax is not None:
        x1 = vmax
    ctf.SetRange(x0, x1)
    lut.SetRange(x0, x1)

    if below_color is not None:
        lut.SetBelowRangeColor(list(get_color(below_color)) + [below_alpha])
        lut.SetUseBelowRangeColor(True)
    if above_color is not None:
        lut.SetAboveRangeColor(list(get_color(above_color)) + [above_alpha])
        lut.SetUseAboveRangeColor(True)
    if nan_color is not None:
        lut.SetNanColor(list(get_color(nan_color)) + [nan_alpha])

    if interpolate:
        for i in range(ncols):
            p = i / (ncols-1)
            x = (1 - p) * x0 + p * x1
            alf = np.interp(x, alpha_x, alpha_vals)
            rgba = list(ctf.GetColor(x)) + [alf]
            lut.SetTableValue(i, rgba)
    else:
        for i in range(ncols):
            if len(colorlist[i]) > 2: 
                alpha = colorlist[i][2]
            else:
                alpha = 1.0
            # print("colorlist entry:", colorlist[i])
            rgba = list(get_color(colorlist[i][1])) + [alpha]
            lut.SetTableValue(i, rgba)

    lut.Build()
    return lut


#########################################################################
def printc(
    *strings,
    c=None,
    bc=None,
    bold=True,
    italic=False,
    blink=False,
    underline=False,
    strike=False,
    dim=False,
    invert=False,
    box="",
    link="",
    end="\n",
    flush=True,
    return_string=False,
):
    """
    Print to terminal in color (any color!).

    Arguments:
        c : (color)
            foreground color name or (r,g,b)
        bc : (color)
            background color name or (r,g,b)
        bold : (bool)
            boldface [True]
        italic : (bool)
            italic [False]
        blink : (bool)
            blinking text [False]
        underline : (bool)
            underline text [False]
        strike : (bool)
            strike through text [False]
        dim : (bool)
            make text look dimmer [False]
        invert : (bool)
            invert background and forward colors [False]
        box : (bool)
            print a box with specified text character ['']
        link : (str)
            print a clickable url link (works on Linux)
            (must press Ctrl+click to open the link)
        flush : (bool)
            flush buffer after printing [True]
        return_string : (bool)
            return the string without printing it [False]
        end : (str)
            the end character to be printed [newline]

    Example:
        ```python
        from vedo.colors import printc
        printc('anything', c='tomato', bold=False, end=' ')
        printc('anything', 455.5, c='lightblue')
        printc(299792.48, c=4)
        ```

    Examples:
        - [printc.py](https://github.com/marcomusy/vedo/tree/master/examples/other/printc.py)

        ![](https://user-images.githubusercontent.com/32848391/50739010-2bfc2b80-11da-11e9-94de-011e50a86e61.jpg)
    """

    if not vedo.settings.enable_print_color or not _terminal_has_colors:
        if return_string:
            return ''.join(strings)
        else:
            print(*strings, end=end, flush=flush)
            return

    try:  # -------------------------------------------------------------

        txt = str()
        ns = len(strings) - 1
        separator = " "
        offset = 0
        for i, s in enumerate(strings):
            if i == ns:
                separator = ""
            if ":" in repr(s):
                for k in emoji:
                    if k in str(s):
                        s = s.replace(k, emoji[k])
                        offset += 1
                for k, rp in vedo.shapes._reps:  # check symbols in shapes._reps
                    if k in str(s):
                        s = s.replace(k, rp)
                        offset += 1

            txt += str(s) + separator

        special, cseq = "", ""
        oneletter_colors = {
            "k": "\u001b[30m",  # because these are supported by most terminals
            "r": "\u001b[31m",
            "g": "\u001b[32m",
            "y": "\u001b[33m",
            "b": "\u001b[34m",
            "m": "\u001b[35m",
            "c": "\u001b[36m",
            "w": "\u001b[37m",
        }

        if c is not None:
            if c is True:
                c = "g"
            elif c is False:
                c = "r"

            if isinstance(c, str) and c in oneletter_colors:
                cseq += oneletter_colors[c]
            else:
                r, g, b = get_color(c)  # not all terms support this syntax
                cseq += f"\x1b[38;2;{int(r*255)};{int(g*255)};{int(b*255)}m"

        if bc:
            if bc in oneletter_colors:
                cseq += oneletter_colors[bc]
            else:
                r, g, b = get_color(bc)
                cseq += f"\x1b[48;2;{int(r*255)};{int(g*255)};{int(b*255)}m"

        if box is True:
            box = "-"
        if underline and not box:
            special += "\x1b[4m"
        if strike and not box:
            special += "\x1b[9m"
        if dim:
            special += "\x1b[2m"
        if invert:
            special += "\x1b[7m"
        if bold:
            special += "\x1b[1m"
        if italic:
            special += "\x1b[3m"
        if blink:
            special += "\x1b[5m"

        if box and "\n" not in txt:
            box = box[0]
            boxv = box
            if box in ["_", "=", "-", "+", "~"]:
                boxv = "|"

            if box in ("_", "."):
                outtxt = special + cseq + " " + box * (len(txt) + offset + 2) + " \n"
                outtxt += boxv + " " * (len(txt) + 2) + boxv + "\n"
            else:
                outtxt = special + cseq + box * (len(txt) + offset + 4) + "\n"

            outtxt += boxv + " " + txt + " " + boxv + "\n"

            if box == "_":
                outtxt += "|" + box * (len(txt) + offset + 2) + "|" + reset + end
            else:
                outtxt += box * (len(txt) + offset + 4) + reset + end

            sys.stdout.write(outtxt)

        else:

            out = special + cseq + txt + reset

            if link:
                # embed a link in the terminal
                out = f"\x1b]8;;{link}\x1b\\{out}\x1b]8;;\x1b\\"

            if return_string:
                return out + end
            else:
                sys.stdout.write(out + end)

    except:  # --------------------------------------------------- fallback

        if return_string:
            return ''.join(strings)

        try:
            print(*strings, end=end)
        except UnicodeEncodeError as e:
            print(e, end=end)
            pass

    if flush:
        sys.stdout.flush()


def printd(*strings, q=False):
    """
    Print debug information about the environment where the printd() is called.
    Local variables are printed out with their current values.

    Use `q` to quit (exit) the python session after the printd call.
    """
    from inspect import currentframe, getframeinfo

    cf = currentframe().f_back
    cfi = getframeinfo(cf)

    fname = os.path.basename(getframeinfo(cf).filename)
    print("\x1b[7m\x1b[3m\x1b[37m" + fname + " line:\x1b[1m" + str(cfi.lineno) + reset, end="")
    print("\x1b[3m\x1b[37m\x1b[2m", "\U00002501" * 30, time.ctime(), reset)
    if strings:
        print("    \x1b[37m\x1b[1mMessage : ", *strings)
    print("    \x1b[37m\x1b[1mFunction:\x1b[0m\x1b[37m " + str(cfi.function))
    print("    \x1b[1mLocals  :" + reset)
    for loc in cf.f_locals.keys():
        obj = cf.f_locals[loc]
        var = repr(obj)
        if 'module ' in var: continue
        if 'function ' in var: continue
        if 'class ' in var: continue
        if loc.startswith('_'): continue
        if hasattr(obj, 'name'):
            if not obj.name:
                oname = str(type(obj))
            else:
                oname = obj.name
            var = oname + ", at " + vedo.utils.precision(obj.GetPosition(), 3)

        var = var.replace("vtkmodules.", "")
        print("      \x1b[37m", loc, "\t\t=", var[:60].replace("\n", ""), reset)
        if vedo.utils.is_sequence(obj) and len(obj) > 4:
            print('           \x1b[37m\x1b[2m\x1b[3m len:', len(obj),
                  ' min:', vedo.utils.precision(min(obj), 4),
                  ' max:', vedo.utils.precision(max(obj), 4),
                  reset)

    if q:
        print(f"    \x1b[1m\x1b[37mExiting python now (q={bool(q)}).\x1b[0m\x1b[37m")
        sys.exit(0)
    sys.stdout.flush()
