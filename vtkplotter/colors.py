from __future__ import division, print_function
import vtk
import numpy as np
import sys
import vtkplotter.docs as docs

__doc__ = (
"""
Colors definitions and printing methods.
"""
    + docs._defs
)

__all__ = [
    "printc",
    "getColor",
    "getColorName",
    "colorMap",
    "makePalette",
    "makeLUTfromCTF",
    "kelvin2rgb",
    "printHistogram",
]


try:
    import matplotlib
    import matplotlib.cm as cm_mpl

    _mapscales = {
        "jet": cm_mpl.jet,
        "hot": cm_mpl.hot,
        "afmhot": cm_mpl.afmhot,
        "rainbow": cm_mpl.rainbow,
        "binary": cm_mpl.binary,
        "gray": cm_mpl.gray,
        "bone": cm_mpl.bone,
        "winter": cm_mpl.winter,
        "cool": cm_mpl.cool,
        "copper": cm_mpl.copper,
        "coolwarm": cm_mpl.coolwarm,
        "gist_earth": cm_mpl.gist_earth,
        "viridis": cm_mpl.viridis,
        "plasma": cm_mpl.plasma,
    }
except:
    _mapscales = None
    pass # see below, this is dealt with in colorMap()


#########################################################
# basic color schemes
#########################################################
colors = {  # from matplotlib
    'aliceblue':            '#F0F8FF', 'antiquewhite':         '#FAEBD7',
    'aqua':                 '#00FFFF', 'aquamarine':           '#7FFFD4',
    'azure':                '#F0FFFF', 'beige':                '#F5F5DC',
    'bisque':               '#FFE4C4', 'black':                '#000000',
    'blanchedalmond':       '#FFEBCD', 'blue':                 '#0000FF',
    'blueviolet':           '#8A2BE2', 'brown':                '#A52A2A',
    'burlywood':            '#DEB887', 'cadetblue':            '#5F9EA0',
    'chartreuse':           '#7FFF00', 'chocolate':            '#D2691E',
    'coral':                '#FF7F50', 'cornflowerblue':       '#6495ED',
    'cornsilk':             '#FFF8DC', 'crimson':              '#DC143C',
    'cyan':                 '#00FFFF', 'darkblue':             '#00008B',
    'darkcyan':             '#008B8B', 'darkgoldenrod':        '#B8860B',
    'darkgray':             '#A9A9A9', 'darkgreen':            '#006400',
    'darkkhaki':            '#BDB76B',
    'darkmagenta':          '#8B008B', 'darkolivegreen':       '#556B2F',
    'darkorange':           '#FF8C00', 'darkorchid':           '#9932CC',
    'darkred':              '#8B0000', 'darksalmon':           '#E9967A',
    'darkseagreen':         '#8FBC8F', 'darkslateblue':        '#483D8B',
    'darkslategray':        '#2F4F4F',
    'darkturquoise':        '#00CED1', 'darkviolet':           '#9400D3',
    'deeppink':             '#FF1493', 'deepskyblue':          '#00BFFF',
    'dimgray':              '#696969',
    'dodgerblue':           '#1E90FF', 'firebrick':            '#B22222',
    'floralwhite':          '#FFFAF0', 'forestgreen':          '#228B22',
    'fuchsia':              '#FF00FF', 'gainsboro':            '#DCDCDC',
    'ghostwhite':           '#F8F8FF', 'gold':                 '#FFD700',
    'goldenrod':            '#DAA520', 'gray':                 '#808080',
    'green':                '#008000', 'greenyellow':          '#ADFF2F',
    'honeydew':             '#F0FFF0',
    'hotpink':              '#FF69B4', 'indianred':            '#CD5C5C',
    'indigo':               '#4B0082', 'ivory':                '#FFFFF0',
    'khaki':                '#F0E68C', 'lavender':             '#E6E6FA',
    'lavenderblush':        '#FFF0F5', 'lawngreen':            '#7CFC00',
    'lemonchiffon':         '#FFFACD', 'lightblue':            '#ADD8E6',
    'lightcoral':           '#F08080', 'lightcyan':            '#E0FFFF',
    'lightgray':            '#D3D3D3', 'lightgreen':           '#90EE90',
    'lightpink':            '#FFB6C1',
    'lightsalmon':          '#FFA07A', 'lightseagreen':        '#20B2AA',
    'lightskyblue':         '#87CEFA',
    'lightsteelblue':       '#B0C4DE',
    'lightyellow':          '#FFFFE0', 'lime':                 '#00FF00',
    'limegreen':            '#32CD32', 'linen':                '#FAF0E6',
    'magenta':              '#FF00FF', 'maroon':               '#800000',
    'mediumaquamarine':     '#66CDAA', 'mediumblue':           '#0000CD',
    'mediumorchid':         '#BA55D3', 'mediumpurple':         '#9370DB',
    'mediumseagreen':       '#3CB371', 'mediumslateblue':      '#7B68EE',
    'mediumspringgreen':    '#00FA9A', 'mediumturquoise':      '#48D1CC',
    'mediumvioletred':      '#C71585', 'midnightblue':         '#191970',
    'mintcream':            '#F5FFFA', 'mistyrose':            '#FFE4E1',
    'moccasin':             '#FFE4B5', 'navajowhite':          '#FFDEAD',
    'navy':                 '#000080', 'oldlace':              '#FDF5E6',
    'olive':                '#808000', 'olivedrab':            '#6B8E23',
    'orange':               '#FFA500', 'orangered':            '#FF4500',
    'orchid':               '#DA70D6', 'palegoldenrod':        '#EEE8AA',
    'palegreen':            '#98FB98', 'paleturquoise':        '#AFEEEE',
    'palevioletred':        '#DB7093', 'papayawhip':           '#FFEFD5',
    'peachpuff':            '#FFDAB9', 'peru':                 '#CD853F',
    'pink':                 '#FFC0CB', 'plum':                 '#DDA0DD',
    'powderblue':           '#B0E0E6', 'purple':               '#800080',
    'rebeccapurple':        '#663399', 'red':                  '#FF0000',
    'rosybrown':            '#BC8F8F', 'royalblue':            '#4169E1',
    'saddlebrown':          '#8B4513', 'salmon':               '#FA8072',
    'sandybrown':           '#F4A460', 'seagreen':             '#2E8B57',
    'seashell':             '#FFF5EE', 'sienna':               '#A0522D',
    'silver':               '#C0C0C0', 'skyblue':              '#87CEEB',
    'slateblue':            '#6A5ACD', 'slategray':            '#708090',
    'snow':                 '#FFFAFA', 'blackboard':           '#393939',
    'springgreen':          '#00FF7F', 'steelblue':            '#4682B4',
    'tan':                  '#D2B48C', 'teal':                 '#008080',
    'thistle':              '#D8BFD8', 'tomato':               '#FF6347',
    'turquoise':            '#40E0D0', 'violet':               '#EE82EE',
    'wheat':                '#F5DEB3', 'white':                '#FFFFFF',
    'whitesmoke':           '#F5F5F5', 'yellow':               '#FFFF00',
    'yellowgreen':          '#9ACD32',
}

color_nicks = {  # color nicknames
    "a": "aqua",
    "b": "blue",
    "bb": "blackboard",
    "c": "cyan",
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
}


def _isSequence(arg):
    # Check if input is iterable.
    if hasattr(arg, "strip"):
        return False
    if hasattr(arg, "__getslice__"):
        return True
    if hasattr(arg, "__iter__"):
        return True
    return False


def getColor(rgb=None, hsv=None):
    """
    Convert a color to (r,g,b) format from many input formats.

    :param bool hsv: if set to `True`, rgb is assumed as (hue, saturation, value).

    Example:
         - RGB    = (255, 255, 255), corresponds to white
         - rgb    = (1,1,1) is white
         - hex    = #FFFF00 is yellow
         - string = 'white'
         - string = 'w' is white nickname
         - string = 'dr' is darkred
         - int    =  7 picks color nr. 7 in a predefined color list
         - int    = -7 picks color nr. 7 in a different predefined list

    .. hint:: |colorcubes| |colorcubes.py|_
    """
    if str(rgb).isdigit():
        rgb = int(rgb)

    if hsv:
        c = hsv2rgb(hsv)
    else:
        c = rgb

    if _isSequence(c):
        if c[0] <= 1 and c[1] <= 1 and c[2] <= 1:
            return c  # already rgb
        else:
            if len(c) == 3:
                return list(np.array(c) / 255.0)  # RGB
            else:
                return [c[0] / 255.0, c[1] / 255.0, c[2] / 255.0, c[3]]  # RGBA

    elif isinstance(c, str):  # is string
        c = c.replace(",", " ").replace("/", " ").replace("alpha=", "")
        c = c.replace("grey", "gray")
        c = c.split()[0]  # ignore possible opacity float inside string
        if 0 < len(c) < 3:  # single/double letter color
            if c.lower() in color_nicks.keys():
                c = color_nicks[c.lower()]
            else:
                print("Unknow color nickname:", c)
                print("Available abbreviations:", color_nicks)
                return [0.5, 0.5, 0.5]

        if c.lower() in colors.keys():  # matplotlib name color
            c = colors[c.lower()]
        else:  # vtk name color
            namedColors = vtk.vtkNamedColors()
            rgba = [0, 0, 0, 0]
            namedColors.GetColor(c, rgba)
            return list(np.array(rgba[0:3]) / 255.0)

        if "#" in c:  # hex to rgb
            h = c.lstrip("#")
            rgb255 = list(int(h[i : i + 2], 16) for i in (0, 2, 4))
            rgbh = np.array(rgb255) / 255.0
            if np.sum(rgbh) > 3:
                print("Error in getColor(): Wrong hex color", c)
                return [0.5, 0.5, 0.5]
            return list(rgbh)

    elif isinstance(c, int):  # color number
        if c >= 0:
            return colors1[c % 10]
        else:
            return colors2[-c % 10]

    elif isinstance(c, float):
        if c >= 0:
            return colors1[int(c) % 10]
        else:
            return colors2[int(-c) % 10]

    print("Unknown color:", c)
    return [0.5, 0.5, 0.5]


def getColorName(c):
    """Find the name of a color.

    .. hint:: |colorpalette| |colorpalette.py|_
    """
    c = np.array(getColor(c))  # reformat to rgb
    mdist = 99.0
    kclosest = ""
    for key in colors.keys():
        ci = np.array(getColor(key))
        d = np.linalg.norm(c - ci)
        if d < mdist:
            mdist = d
            kclosest = str(key)
    return kclosest


def hsv2rgb(hsv):
    """Convert HSV to RGB color."""
    ma = vtk.vtkMath()
    return ma.HSVToRGB(hsv)


def rgb2hsv(rgb):
    """Convert RGB to HSV color."""
    ma = vtk.vtkMath()
    return ma.RGBToHSV(getColor(rgb))
    

def colorMap(value, name="jet", vmin=None, vmax=None):
    """Map a real value in range [vmin, vmax] to a (r,g,b) color scale.

    :param value: scalar value to transform into a color
    :type value: float, list
    :param name: color map name
    :type name: str, matplotlib.colors.LinearSegmentedColormap

    :return: (r,g,b) color, or a list of (r,g,b) colors.

    .. note:: Available color maps:

        |colormaps|

    .. tip:: Can also use directly a matplotlib color map:

        :Example:
            >>> from vtkplotter import colorMap
            >>> import matplotlib.cm as cm
            >>> print( colorMap(0.2, cm.flag, 0, 1) )
            (1.0, 0.809016994374948, 0.6173258487801733)
    """
    if not _mapscales:
        print("-------------------------------------------------------------------")
        print("WARNING : cannot import matplotlib.cm (colormaps will show up gray).")
        print("Try e.g.: sudo apt-get install python3-matplotlib")
        print("     or : pip install matplotlib")
        print("     or : build your own map (see example in basic/mesh_custom.py).")
        return (0.5, 0.5, 0.5)

    if isinstance(name, matplotlib.colors.LinearSegmentedColormap):
        mp = name
    else:
        if name in _mapscales.keys():
            mp = _mapscales[name]
        else:
            print("Error in colorMap():", name, "\navaliable maps =", sorted(_mapscales.keys()))
            exit(0)

    if _isSequence(value):
        values = np.array(value)
        if vmin is None:
            vmin = np.min(values)
        if vmax is None:
            vmax = np.max(values)
        values = np.clip(values, vmin, vmax)
        values -= vmin
        values /= vmax - vmin
        cols = []
        mp = _mapscales[name]
        for v in values:
            cols.append(mp(v)[0:3])
        return np.array(cols)
    else:
        value -= vmin
        value /= vmax - vmin
        if value > 0.999:
            value = 0.999
        elif value < 0:
            value = 0
        return mp(value)[0:3]


def makePalette(color1, color2, N, hsv=True):
    """
    Generate N colors starting from `color1` to `color2` 
    by linear interpolation HSV in or RGB spaces.

    :param int N: number of output colors.
    :param color1: first rgb color.
    :param color2: second rgb color.
    :param bool hsv: if `False`, interpolation is calculated in RGB space.

    .. hint:: Example: |colorpalette.py|_
    """
    if hsv:
        color1 = rgb2hsv(color1)
        color2 = rgb2hsv(color2)
    c1 = np.array(getColor(color1))
    c2 = np.array(getColor(color2))
    cols = []
    for f in np.linspace(0, 1, N - 1, endpoint=True):
        c = c1 * (1 - f) + c2 * f
        if hsv:
            c = np.array(hsv2rgb(c))
        cols.append(c)
    return cols


def makeLUTfromCTF(sclist, N=None):
    """
    Use a Color Transfer Function to generate colors in a vtk lookup table.
    See `here <http://www.vtk.org/doc/nightly/html/classvtkColorTransferFunction.html>`_.

    :param list sclist: a list in the form ``[(scalar1, [r,g,b]), (scalar2, 'blue'), ...]``.     
    :return: the lookup table object ``vtkLookupTable``. This can be fed into ``colorMap``.
    """
    ctf = vtk.vtkColorTransferFunction()
    ctf.SetColorSpaceToDiverging()

    for sc in sclist:
        scalar, col = sc
        r, g, b = getColor(col)
        ctf.AddRGBPoint(scalar, r, g, b)

    if N is None:
        N = len(sclist)

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(N)
    lut.Build()

    for i in range(N):
        rgb = list(ctf.GetColor(float(i) / N)) + [1]
        lut.SetTableValue(i, rgb)

    return lut


def kelvin2rgb(temperature):
    """
    Converts from Kelvin temperature to an RGB color.

    Algorithm credits: |tannerhelland|_
    """
    # range check
    if temperature < 1000:
        temperature = 1000
    elif temperature > 40000:
        temperature = 40000

    tmp_internal = temperature / 100.0

    # red
    if tmp_internal <= 66:
        red = 255
    else:
        tmp_red = 329.698727446 * np.power(tmp_internal - 60, -0.1332047592)
        if tmp_red < 0:
            red = 0
        elif tmp_red > 255:
            red = 255
        else:
            red = tmp_red

    # green
    if tmp_internal <= 66:
        tmp_green = 99.4708025861 * np.log(tmp_internal) - 161.1195681661
        if tmp_green < 0:
            green = 0
        elif tmp_green > 255:
            green = 255
        else:
            green = tmp_green
    else:
        tmp_green = 288.1221695283 * np.power(tmp_internal - 60, -0.0755148492)
        if tmp_green < 0:
            green = 0
        elif tmp_green > 255:
            green = 255
        else:
            green = tmp_green

    # blue
    if tmp_internal >= 66:
        blue = 255
    elif tmp_internal <= 19:
        blue = 0
    else:
        tmp_blue = 138.5177312231 * np.log(tmp_internal - 10) - 305.0447927307
        if tmp_blue < 0:
            blue = 0
        elif tmp_blue > 255:
            blue = 255
        else:
            blue = tmp_blue

    return [red / 255, green / 255, blue / 255]


# default sets of colors
colors1 = [
    [1.0, 0.832, 0.000],  # gold
    [0.960, 0.509, 0.188],
    [0.901, 0.098, 0.194],
    [0.235, 0.85, 0.294],
    [0.46, 0.48, 0.000],
    [0.274, 0.941, 0.941],
    [0.0, 0.509, 0.784],
    [0.1, 0.1, 0.900],
    [0.902, 0.7, 1.000],
    [0.941, 0.196, 0.901],
]
# negative integer color number get this:
colors2 = [
    (0.99, 0.83, 0),  # gold
    (0.59, 0.0, 0.09),  # dark red
    (0.5, 1.0, 0.0),  # green
    (0.5, 0.5, 0),  # yellow-green
    (0.0, 0.66, 0.42),  # green blue
    (0.0, 0.18, 0.65),  # blue
    (0.4, 0.0, 0.4),  # plum
    (0.4, 0.0, 0.6),
    (0.2, 0.4, 0.6),
    (0.1, 0.3, 0.2),
]


# terminal color print
def _has_colors(stream):
    if not hasattr(stream, "isatty"):
        return False
    if not stream.isatty():
        return False
    try:
        import curses

        curses.setupterm()
        return curses.tigetnum("colors") > 2
    except:
        return False
_terminal_has_colors = _has_colors(sys.stdout)

_terminal_cols = {
    "black": 0,
    "red": 1,
    "green": 2,
    "yellow": 3,
    "blue": 4,
    "magenta": 5,
    "cyan": 6,
    "white": 7,
    "k": 0,
    "r": 1,
    "g": 2,
    "y": 3,
    "b": 4,
    "m": 5,
    "c": 6,
    "w": 7,
}

emoji = {
    "~bomb": u"\U0001F4A5",
    "~sparks": u"\U00002728",
    "~thumbup": u"\U0001F44d",
    "~target": u"\U0001F3af",
    "~save": u"\U0001F4be",
    "~noentry": u"\U000026d4",
    "~video": u"\U0001F4fd ",
    "~lightning": u"\U000026a1",
    "~camera": u"\U0001F4f8",
    "~!?": u"\U00002049",
    "~times": u"\U0000274c",
    "~world": u"\U0001F30d",
    "~rainbow": u"\U0001F308",
    "~idea": u"\U0001F4a1",
    "~pin": u"\U0001F4CC",
    "~construction": u"\U0001F6A7",
    "~uparrow": u"\U00002b06",
    "~rightarrow": u"\U000027a1",
    "~leftarrow": u"\U00002b05",
    "~downarrow": u"\U00002b07",
    "~plus": u"\U00002795",
    "~minus": u"\U00002796",
    "~division": u"\U00002797",
    "~rocket": u"\U0001F680",
    "~hourglass": u"\U000023f3",
    "~diamomd": u"\U0001F48e",
    "~dna": u"\U0001F9ec",
    "~prohibited": u"\U0001F6ab",
    "~checked": u"\U00002705",
    "~copyright": u"\U000000a9",
    "~registered": u"\U000000ae",
    "~trademark": u"\U00002122",
    "~flag": u"\U0001F3c1",
    "~smile": u"\U0001F642",
    "~sad": u"\U0001F612",
    "~bigstar": u"\U00002B50",
    "~smallstar": u"\U00002733",
    "~redtriangle": u"\U0001F53a",
    "~orangesquare": u"\U0001F538",
    "~bluesquare": u"\U0001F537",
    "~zzz": u"\U0001F4a4",
}


def printc(*strings, **keys):
    """
    Print to terminal in colors. (python3 only).

    Available colors are:
        black, red, green, yellow, blue, magenta, cyan, white.

    :param c: foreground color ['']
    :param bc: background color ['']
    :param hidden: do not show text [False]
    :param bold: boldface [True]
    :param blink: blinking text [False]
    :param underline: underline text [False]
    :param dim: make text look dimmer [False]
    :param invert: invert background anf forward colors [False]
    :param box: print a box with specified text character ['']
    :param flush: flush buffer after printing [True]
    :param end: end character to be printed [return]

    :Example:

    >>>  from vtkplotter.colors import printc
    >>>  printc('anything', c='red', bold=False, end='' )
    >>>  printc('anything', 455.5, vtkObject, c='green')
    >>>  printc(299792.48, c=4) # 4 is blue

    .. hint:: |colorprint| |colorprint.py|_
    """

    end = keys.pop("end", "\n")
    flush = keys.pop("flush", True)

    if not _terminal_has_colors or sys.version_info[0]<3:
        for s in strings:
            if "~" in str(s):  # "in" for some reasons changes s
                for k in emoji.keys():
                    if k in s:
                        s = s.replace(k, '')
            print(s, end=' ')
        print(end=end)
        if flush:
            sys.stdout.flush()
        return

    c = keys.pop("c", None)  # hack to be compatible with python2
    bc = keys.pop("bc", None)
    hidden = keys.pop("hidden", False)
    bold = keys.pop("bold", True)
    blink = keys.pop("blink", False)
    underline = keys.pop("underline", False)
    dim = keys.pop("dim", False)
    invert = keys.pop("invert", False)
    box = keys.pop("box", "")
    
    if c is True:
        c = 'green'
    elif c is False:
        c = 'red'

    try:
        txt = str()
        ns = len(strings) - 1
        separator = " "
        offset = 0
        for i, s in enumerate(strings):
            if i == ns:
                separator = ""
                
            #txt += str(s) + separator
            if "~" in str(s):  # "in" for some reasons changes s
                for k in emoji.keys():
                    if k in str(s):
                        s = s.replace(k, emoji[k])
                        offset += 1
            txt += str(s) + separator

        if c:
            if isinstance(c, int):
                cf = abs(c) % 8
            elif isinstance(c, str):
                cf = _terminal_cols[c.lower()]
            else:
                print("Error in printc(): unknown color c=", c)
                exit()
        if bc:
            if isinstance(bc, int):
                cb = abs(bc) % 8
            elif isinstance(bc, str):
                cb = _terminal_cols[bc.lower()]
            else:
                print("Error in printc(): unknown color c=", c)
                exit()

        special, cseq = "", ""
        if hidden:
            special += "\x1b[8m"
            box = ""
        else:
            if c:
                cseq += "\x1b[" + str(30 + cf) + "m"
            if bc:
                cseq += "\x1b[" + str(40 + cb) + "m"
            if underline and not box:
                special += "\x1b[4m"
            if dim:
                special += "\x1b[2m"
            if invert:
                special += "\x1b[7m"
            if bold:
                special += "\x1b[1m"
            if blink:
                special += "\x1b[5m"

        if box and not ("\n" in txt):
            if len(box) > 1:
                box = box[0]
            if box in ["_", "=", "-", "+", "~"]:
                boxv = "|"
            else:
                boxv = box

            if box == "_" or box == ".":
                outtxt = special + cseq + " " + box * (len(txt) + offset + 2) + " \n"
                outtxt += boxv + " " * (len(txt) + 2) + boxv + "\n"
            else:
                outtxt = special + cseq + box * (len(txt) + offset + 4) + "\n"

            outtxt += boxv + " " + txt + " " + boxv + "\n"

            if box == "_":
                outtxt += "|" + box * (len(txt) + offset + 2) + "|" + "\x1b[0m" + end
            else:
                outtxt += box * (len(txt) + offset + 4) + "\x1b[0m" + end

            sys.stdout.write(outtxt)
        else:
            sys.stdout.write(special + cseq + txt + "\x1b[0m" + end)
    
    except:
        print(*strings, end=end)

    if flush:
        sys.stdout.flush()


def printHistogram(data, bins=10, height=10, logscale=False,
                   horizontal=False, char=u"\U00002589",
                   c=None, bold=True, title='Histogram'):
    """
    Ascii histogram printing.

    :param int bins: number of histogram bins
    :param int height: height of the histogram in character units
    :param bool logscale: use logscale for frequencies
    :param bool horizontal: show histogram horizontally
    :param str char: character to be used
    :param str,int c: ascii color
    :param bool char: use boldface
    :param str title: histogram title
    
    :Example:
        
        >>> from vtkplotter import printHistogram
        >>> import numpy as np
        >>> d = np.random.normal(size=1000)
        >>> printHistogram(d, c='blue', logscale=True, title='my scalars')
        >>> printHistogram(d, c=1, horizontal=1)

    """
    # Adapted from
    # http://pyinsci.blogspot.com/2009/10/ascii-histograms.html
    
    if not horizontal: # better aspect ratio
        bins *= 2
    
    h = np.histogram(data, bins=bins)
    
    if char == u"\U00002589" and horizontal:
        char = u"\U00002586"
    
    if logscale:
        h0 = np.log10(h[0]+1)
        maxh0 = int(max(h0)*100)/100
        title = '(logscale) ' + title
    else:
        h0 = h[0]
        maxh0 = max(h0)

    def _v():
        his = ""
        if title:
            his += title +"\n"
        bars = h0 / maxh0 * height
        for l in reversed(range(1, height + 1)):
            line = ""
            if l == height:
                line = "%s " % maxh0  
            else:
                line = " " * (len(str(maxh0)) + 1) 
            for c in bars:
                if c >= np.ceil(l):
                    line += char
                else:
                    line += " "
            line += "\n"
            his += line
        his += "%.2f" % h[1][0] + "." * (bins) + "%.2f" % h[1][-1] + "\n"
        return his

    def _h():
        his = ""
        if title:
            his += title +"\n"
        xl = ["%.2f" % n for n in h[1]]
        lxl = [len(l) for l in xl]
        bars = h0 / maxh0 * height
        his += " " * int(max(bars) + 2 + max(lxl)) + "%s\n" % maxh0
        for i, c in enumerate(bars):
            line = (xl[i] + " " * int(max(lxl) - lxl[i]) + "| " + char * int(c) + "\n")
            his += line
        return his

    if horizontal:
        height *= 2
        printc(_h(), c=c, bold=bold)
    else:
        printc(_v(), c=c, bold=bold)

