# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 20:06:00 2017

@author: mmusy
"""
from __future__ import division, print_function
import utils 
import numpy as np


#########################################################
# basic color schemes
######################################################### 
colors = { # from matplotlib
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
    'darkgrey':             '#A9A9A9', 'darkkhaki':            '#BDB76B',
    'darkmagenta':          '#8B008B', 'darkolivegreen':       '#556B2F',
    'darkorange':           '#FF8C00', 'darkorchid':           '#9932CC',
    'darkred':              '#8B0000', 'darksalmon':           '#E9967A',
    'darkseagreen':         '#8FBC8F', 'darkslateblue':        '#483D8B',
    'darkslategray':        '#2F4F4F', 'darkslategrey':        '#2F4F4F',
    'darkturquoise':        '#00CED1', 'darkviolet':           '#9400D3',
    'deeppink':             '#FF1493', 'deepskyblue':          '#00BFFF',
    'dimgray':              '#696969', 'dimgrey':              '#696969',
    'dodgerblue':           '#1E90FF', 'firebrick':            '#B22222',
    'floralwhite':          '#FFFAF0', 'forestgreen':          '#228B22',
    'fuchsia':              '#FF00FF', 'gainsboro':            '#DCDCDC',
    'ghostwhite':           '#F8F8FF', 'gold':                 '#FFD700',
    'goldenrod':            '#DAA520', 'gray':                 '#808080',
    'green':                '#008000', 'greenyellow':          '#ADFF2F',
    'grey':                 '#808080', 'honeydew':             '#F0FFF0',
    'hotpink':              '#FF69B4', 'indianred':            '#CD5C5C',
    'indigo':               '#4B0082', 'ivory':                '#FFFFF0',
    'khaki':                '#F0E68C', 'lavender':             '#E6E6FA',
    'lavenderblush':        '#FFF0F5', 'lawngreen':            '#7CFC00',
    'lemonchiffon':         '#FFFACD', 'lightblue':            '#ADD8E6',
    'lightcoral':           '#F08080', 'lightcyan':            '#E0FFFF',
    'lightgray':            '#D3D3D3', 'lightgreen':           '#90EE90',
    'lightgrey':            '#D3D3D3', 'lightpink':            '#FFB6C1',
    'lightsalmon':          '#FFA07A', 'lightseagreen':        '#20B2AA',
    'lightskyblue':         '#87CEFA', 'lightslategray':       '#778899',
    'lightslategrey':       '#778899', 'lightsteelblue':       '#B0C4DE',
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
    'slategrey':            '#708090', 'snow':                 '#FFFAFA',
    'springgreen':          '#00FF7F', 'steelblue':            '#4682B4',
    'tan':                  '#D2B48C', 'teal':                 '#008080',
    'thistle':              '#D8BFD8', 'tomato':               '#FF6347',
    'turquoise':            '#40E0D0', 'violet':               '#EE82EE',
    'wheat':                '#F5DEB3', 'white':                '#FFFFFF',
    'whitesmoke':           '#F5F5F5', 'yellow':               '#FFFF00',
    'yellowgreen':          '#9ACD32'}

color_nicks = {
        'b': 'blue',
        'g': 'green',
        'r': 'red',
        'c': 'cyan',
        'm': 'magenta',
        'y': 'yellow',
        'k': 'black',
        'w': 'white',
        't': 'tomato',
        'o': 'olive',
        'p': 'purple',
        's': 'salmon',
        'v': 'violet'}
color_nicks.update({   # light
        'lb': 'lightblue',
        'lg': 'lightgreen',
        'lc': 'lightcyan',
        'ls': 'lightsalmon',
        'ly': 'lightyellow'})
color_nicks.update({   # dark
        'dr': 'darkred',
        'db': 'darkblue',
        'dg': 'darkgreen',
        'dm': 'darkmagenta',
        'dc': 'darkcyan',
        'ds': 'darksalmon',
        'dv': 'darkviolet'})


def getColor(rgb=None, hsv=None):
    """
    Convert a color to (r,g,b) format from many input formats, e.g.:
     RGB    = (255, 255, 255), corresponds to white
     rgb    = (1,1,1) 
     hex    = #FFFF00 is yellow
     string = 'white'
     string = 'dr' is darkred
     int    = 7 picks color #7 in list colors1
     if hsv is set to (hue,saturation,value), rgb is calculated from it 
    """
    if hsv: c = hsv2rgb(hsv)
    else: c = rgb 
    if utils.isSequence(c) :
        if c[0]<=1 and c[1]<=1 and c[2]<=1: return c #already rgb
        else: return list(np.array(c)/255.) #RGB

    elif isinstance(c, str):
        c = c.replace(',',' ').replace('/',' ').replace('alpha=','')
        c = c.split()[0] #ignore possible opacity float inside string
        if 0 < len(c) < 3: 
            try: # single/double letter color
                c = color_nicks[c.lower()] 
            except KeyError:
                print("Unknow color nickname:", c)
                print ("Available abbreviations:", color_nicks)
                return [0.5,0.5,0.5]
    
        try: # full name color
            c = colors[c.lower()] 
        except KeyError: 
            import vtk
            if vtk.vtkVersion().GetVTKMajorVersion() > 5:
                namedColors = vtk.vtkNamedColors()
                rgba=[0,0,0,0]
                namedColors.GetColor(c, rgba)
                return rgba[0:3]
            print("Unknow color name:", c)
            print ("Available colors:", colors.keys())
            return [0.5,0.5,0.5]

        if '#' in c: #hex to rgb
            h = c.lstrip('#')
            rgb255 = list(int(h[i:i+2], 16) for i in (0, 2 ,4))
            rgbh = np.array(rgb255)/255.
            if np.sum(rgbh)>3: 
                print("Error in getColor(): Wrong hex color", c)
                return [0.5,0.5,0.5]
            return list(rgbh)
            
    elif isinstance(c, int): 
        return colors1[c%10] 

    print('Unknown color:', c)
    return [0.5,0.5,0.5]
    

def getAlpha(c):
    "Check if color string contains a float representing opacity"
    if isinstance(c, str):
        sc = c.replace(',',' ').replace('/',' ').replace('alpha=','').split()
        if len(sc)==1: return None
        return float(sc[-1])
    return None


def getColorName(c):
    """Convert any rgb color or numeric code to closest name color"""
    c = np.array(getColor(c)) #reformat to rgb
    mdist = 99.
    kclosest = ''
    for key in colors.keys():
        ci = np.array(getColor(key))
        d = np.linalg.norm(c-ci)
        if d<mdist: 
            mdist = d
            kclosest = str(key)
    return kclosest


def hsv2rgb(hsv):
    import vtk
    ma = vtk.vtkMath()
    return ma.HSVToRGB(hsv)    
def rgb2hsv(rgb):
    import vtk
    ma = vtk.vtkMath()
    return ma.RGBToHSV(getColor(rgb))

try:
    import matplotlib.cm as cm_mpl
    mapscales = {
        'jet':cm_mpl.jet,
        'hot':cm_mpl.hot,
        'afmhot':cm_mpl.afmhot,
        'rainbow': cm_mpl.rainbow,
        'binary':cm_mpl.binary,
        'gray':cm_mpl.gray,
        'bone':cm_mpl.bone,
        'winter':cm_mpl.winter,
        'cool': cm_mpl.cool,
        'copper':cm_mpl.copper,
        'coolwarm':cm_mpl.coolwarm,
        'gist_earth':cm_mpl.gist_earth
    }
except: mapscales = None
    
def colorMap(value, name='jet', vmin=0, vmax=1): # maps [0,1] into a color scale
    value = value - vmin
    value = value / vmax
    if mapscales:
        if value>.999: value=.999
        elif value<0: value=0
        try: 
            return mapscales[name](value)[0:3]
        except:
            print('Error in colorMap(): avaliable maps =', sorted(mapscales.keys()))
            exit(0)
    return (0.5,0.5,0.5)


def kelvin2rgb(temperature):
    """
    Converts from K to RGB, algorithm courtesy of 
    http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
    https://gist.github.com/petrklus/b1f427accdf7438606a6#file-rgb_to_kelvin-py
    """
    #range check
    if temperature < 1000: temperature = 1000
    elif temperature > 40000: temperature = 40000
    
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
    if tmp_internal <=66:
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
    if tmp_internal >=66:
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
    
    return [red/255, green/255, blue/255]
    

########## other sets of colors
colors1=[]
colors1.append((1.0,0.647,0.0))     # orange
colors1.append((0.59,0.0,0.09))     # dark red
colors1.append((0.5,1.0,0.0))       # green
colors1.append((0.5,0.5,0))         # yellow-green
colors1.append((0.0, 0.66,0.42))    # green blue
colors1.append((0.0,0.18,0.65))     # blue
colors1.append((0.4,0.0,0.4))       # plum
colors1.append((0.4,0.0,0.6))
colors1.append((0.2,0.4,0.6))
colors1.append((0.1,0.3,0.2))

colors2=[]
colors2.append((0.99,0.83,0))       # gold
colors2.append((0.59, 0.0,0.09))    # dark red
colors2.append((.984,.925,.354))    # yellow
colors2.append((0.5,  0.5,0))       # yellow-green
colors2.append((0.5,  1.0,0.0))     # green
colors2.append((0.0, 0.66,0.42))    # green blue
colors2.append((0.0, 0.18,0.65))    # blue
colors2.append((0.4,  0.0,0.4))     # plum
colors2.append((0.5,  0.5,0))       # yellow-green
colors2.append((.984,.925,.354))    # yellow

colors3=[]
for i in range(10):
    pc = (i+0.5)/10
    r = np.exp(-((pc    )/.2)**2/2)
    g = np.exp(-((pc-0.5)/.2)**2/2)
    b = np.exp(-((pc-1.0)/.2)**2/2)
    colors3.append((r,g,b))



