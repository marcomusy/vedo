"""Earthquakes of magnitude 2.5+ in the past 30 days
areas are proportional to energy release
[hover mouse to get more info]"""
import pandas, numpy as np
from vedo import download, Picture, Polygon, ProgressBar, colorMap, Plotter, Text2D

num = 50  # nr of earthquakes to be visualized to define a time window
path = download("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_month.csv")
usecols = ['time','place','latitude','longitude','depth','mag']
data = pandas.read_csv(path, usecols=usecols)[usecols][::-1].reset_index(drop=True) # reverse list

pic = Picture("https://eoimages.gsfc.nasa.gov/images/imagerecords/147000/147190/eo_base_2020_clean_3600x1800.png")
pic.pickable(False).level(185).window(120)  # add some contrast to the original image
scale = [pic.shape[0]/2, pic.shape[1]/2, 1]

def GeoCircle(lat, lon, r, res=50):
    coords = []
    sinr, cosr = np.sin(r), np.cos(r)
    sinlat, coslat = np.sin(lat), np.cos(lat)
    for phi in np.linspace(0, 2*np.pi, num=res, endpoint=False):
        clat = np.arcsin(sinlat * cosr + coslat * sinr * np.cos(phi))
        clng = lon + np.arctan2(np.sin(phi) * sinr * coslat, cosr - sinlat * np.sin(clat))
        coords.append([clng/np.pi + 1, clat*2/np.pi + 1, 0])
    return Polygon(nsides=res).points(coords)  # warp polygon points to match geo projection


centers = []
pb = ProgressBar(0, len(data))
for i, d in data.iterrows():
    pb.print("Parsing USGS data..")
    M = d['mag']                                       # earthquake estimated magnitude
    E = np.sqrt(np.exp(5.24+1.44*M) * scale[0])/10000  # empirical formula for sqrt(energy_release(M))
    rgb = colorMap(E, name='Reds', vmin=0, vmax=7)     # map energy to color
    lat, lon = np.deg2rad(d['latitude']), np.deg2rad(d['longitude'])
    ce = GeoCircle(lat, lon, E/50).scale(scale).z(num/M).c(rgb).lw(0.1).useBounds(False)
    ce.time = i
    ce.info = '\n'.join(str(d).split('\n')[:-1])       # remove of the last line in string d
    #if M > 6.5: ce.alpha(0.8)                         # make the big ones slightly transparent
    if i < len(data)-num: ce.off()                     # switch off older ones: make circles invisible
    centers.append(ce)


def sliderfunc(widget, event):
    value = widget.GetRepresentation().GetValue()      # get the slider current value
    widget.GetRepresentation().SetTitleText(f"{data['time'][int(value)][:10]}")
    for ce in centers:
        isinside = abs(value-ce.time) < num            # switch on if inside of time window
        ce.on() if isinside else ce.off()
    plt.render()

plt = Plotter(size=(2200,1100), title="Earthquake Browser")
plt.addSlider2D(sliderfunc, 0, len(centers)-1, value=len(centers)-1, showValue=False, title="today")
plt.addHoverLegend(useInfo=True, alpha=1, c='white', bg='red2', s=1)
comment = Text2D(__doc__, bg='green9', alpha=0.7)
plt.show(pic, centers, comment, zoom=2.27, mode='image').close()
