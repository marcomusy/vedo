# Credits and terms of use: https://github.com/CSSEGISandData/COVID-19
from vtkplotter import *
import numpy as np
import sys

# Download the data:
date = '03-22-2020'
if len(sys.argv)>1:
    date = sys.argv[1]
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data'
f = download(url+'/csse_covid_19_daily_reports/'+date+'.csv')
data = np.genfromtxt(f, delimiter=',', dtype=None, invalid_raise=False, encoding='utf-8')

# Create the scene:
for d in data:
    if d[6] == 'Latitude': continue
    theta, phi = -float(d[6])/57.3+np.pi/2, float(d[7])/57.3+np.pi
    confirmed, deaths = int(d[3]), int(d[4])
    if not confirmed: continue
    pos = spher2cart(1, theta, phi)
    region=''
    if d[0] and d[0] != d[1]:
        region='\n'+d[0]
    fl = d[1]+region+'\ncases: '+str(confirmed)+'\ndeaths: '+str(deaths)
    Sphere(pos, np.power(confirmed, 1/3)/500, alpha=0.4).flag(fl)
    Sphere(pos, np.power(deaths   , 1/3)/500, alpha=0.4, c='k')

allconf = np.sum(data[1:,3].astype(int))
allreco = np.sum(data[1:,5].astype(int))
alldeat = np.sum(data[1:,4].astype(int))
Text2D('COVID-19 spread on '+date
       +'\n#cases : '+str(allconf)
       +'\n#recovd: '+str(allreco)
       +'\n#deaths: '+str(alldeat))
Earth()

show(..., axes=12, bg2='lb', viewup='z', zoom=1.5)
