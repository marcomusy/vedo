"""Earthquakes of magnitude 2.5+ in the past 30 days
areas are proportional to energy release
[hover mouse to get more info]"""
import pandas
from vedo import *

num = 50  # nr of earthquakes to be visualized at once
path = download("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_month.csv", force=True)
usecols = ['time', 'place', 'latitude', 'longitude', 'depth', 'mag']
data = pandas.read_csv(path, usecols=usecols)[usecols][::-1].reset_index(drop=True) # reverse list

pic = Image(dataurl + "images/eo_base_2020_clean_3600x1800.png")
pic.pickable(False).level(185).window(120)  # add some contrast to the original image
scale = [pic.shape[0]/2, pic.shape[1]/2, 1]
comment = Text2D(__doc__, bg='green9', alpha=0.7, font='Ubuntu')

centers = []
for i, d in progressbar(data.iterrows()):
    M = d['mag']                                       # earthquake estimated magnitude
    E = np.sqrt(np.exp(5.24+1.44*M) * scale[0])/10000  # empirical formula for sqrt(energy_release(M))
    rgb = color_map(E, name='Reds', vmin=0, vmax=7)    # map energy to color
    lat = np.deg2rad(d['latitude'])
    lon = np.deg2rad(d['longitude'])
    ce = GeoCircle(lat, lon, E/50).scale(scale).z(num/M)
    ce.color(rgb, 0.7).force_opaque().use_bounds(False)
    ce.time = i
    ce.info = '\n'.join(str(d).split('\n')[:-1])       # remove of the last line in string d
    if i < len(data) - num: 
        ce.off()                                       # switch off older ones: make circles invisible
    centers.append(ce)


def sliderfunc(widget, event):
    val = widget.value                                 # get the slider current value
    widget.title = f"{data['time'][int(val)][:10]}"
    for ce in centers:
        isinside = abs(val-ce.time) < num              # switch on if inside of time window
        ce.on() if isinside else ce.off()

plt = Plotter(size=(2200,1100), title="vedo - Earthquake Browser").parallel_projection(True)
plt.add_slider(sliderfunc, 0, len(centers)-1, value=len(centers)-1, show_value=False)
plt.add_hover_legend(use_info=True, alpha=1, c='white', bg='red2', s=1)
plt.show(pic, centers, comment, zoom="tightest", mode='image').close()
