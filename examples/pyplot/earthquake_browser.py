"""Browse earthquakes of magnitude 2.5+ in the past 30 days"""
import pandas, numpy as np
from vedo import *

num = 50  # nr of earthquakes to be visualized in the time window

printc("..downloading USGS data.. please wait..", invert=True)
path = download("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_month.csv")
usecols = ['time','place','latitude','longitude','depth','mag']
data = pandas.read_csv(path, usecols=usecols)[usecols][::-1].reset_index(drop=True) # reverse list

pic = Picture("https://eoimages.gsfc.nasa.gov/images/imagerecords/147000/147190/eo_base_2020_clean_3600x1800.png")
emag = Picture('https://www.dropbox.com/s/ynp7kts2lhf32cb/earthq_mag_e.jpg').scale(0.4).pos(1400,10,1)
picscale = [pic.shape[0]/2, pic.shape[1]/2, 1]

def GeoCircle(lat, lon, r, res=50):
    coords = []
    for phi in np.linspace(0, 2*np.pi, num=res, endpoint=False):
        clat = np.arcsin(sin(lat) * cos(r) + cos(lat) * sin(r) * cos(phi))
        clng = lon + np.arctan2(sin(phi) * sin(r) * cos(lat), cos(r) - sin(lat) * sin(clat))
        coords.append([clng/np.pi + 1, clat*2/np.pi + 1, 0])
    return Polygon(nsides=res).points(coords) # reset polygon points


centers = []
pb = ProgressBar(0, len(data))
for i, d in data.iterrows():
    pb.print("Parsing USGS data..")
    M = d['mag']                                    # earthquake estimated magnitude
    E = sqrt(exp(5.24+1.44*M) * picscale[0])/10000  # empirical formula for sqrt(energy_release(M))
    rgb = colorMap(E, name='Reds', vmin=0, vmax=7)  # map energy to color
    lat, long = np.deg2rad(d['latitude']), np.deg2rad(d['longitude'])
    ce = GeoCircle(lat, long, E/50).scale(picscale).z(num/M).c(rgb).lw(0.1).useBounds(False)
    ce.time = i
    ce.info = '\n'.join(str(d).split('\n')[:-1])    # remove of the last line in string d
    #if M > 6.5: ce.alpha(0.8)                       # make the big ones slightly transparent
    if i < len(data)-num: ce.off()                  # switch off older ones: make circles invisible
    centers.append(ce)


def sliderfunc(widget, event):
    value = widget.GetRepresentation().GetValue()   # get the slider current value
    widget.GetRepresentation().SetTitleText(f"{data['time'][int(value)][:10]}")
    for ce in centers:
        isinside = abs(value-ce.time) < num         # switch on if inside of time window
        ce.on() if isinside else ce.off()
    plt.render()

plt = Plotter(size=(2200,1100), title=__doc__)
plt.addSlider2D(sliderfunc, 0, len(centers)-1, value=len(centers)-1, showValue=False, title="today")
plt.addHoverLegend(useInfo=True, alpha=1, c='w', bg='red2', s=1)
comment = Text2D("Areas are proportional to energy release\n[hover mouse to get more info]", bg='g9', alpha=.7)
plt.show(pic.pickable(False), emag.pickable(False), centers, comment, zoom=2.27).close()
