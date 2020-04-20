import numpy as np
import sys


# ----------------------------------------------------------------------
# pick today date
date="04-18-2020"
if len(sys.argv)>1: 
    date = sys.argv[1]
else:
    import datetime
    from vtkplotter import download
    for i in range(3):
        try:
            yesterday = datetime.datetime.now() - datetime.timedelta(days=i)
            date = yesterday.strftime("%m-%d-%Y")
            url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master'
            fpath = download(url+'/csse_covid_19_data/csse_covid_19_daily_reports/'+date+'.csv')
            break
        except:
            continue

def load_data():
    # Download and read the data from the Johns Hopkins University repo:
    # Credits and terms of use: https://github.com/CSSEGISandData/COVID-19
    #0    1       2             3              4           5   6     
    #FIPS,Admin2,Province_State,Country_Region,Last_Update,Lat,Long_,
    #7          8     9          10    11
    #Confirmed,Deaths,Recovered,Active,Combined_Key
    with open(fpath, "r") as f: lines = f.readlines()
    data, allconf, allreco, alldeat, conf_us, reco_us, deat_us = [], 0,0,0, 0,0,0
    for i,ln in enumerate(lines):
        if i<1: continue
        ln = ln.replace("Korea,", "Korea").split(",")
        try:
            if not ln[5]: continue
            theta, phi = -float(ln[5])/57.3+np.pi/2, float(ln[6])/57.3+np.pi
            confirmed, deaths, recos = int(ln[7]), int(ln[8]), int(ln[9])
            Admin, Province, Country = ln[1:4]
            if Country == 'US':  # group US data
                conf_us += confirmed
                deat_us += deaths
                reco_us += recos
                continue
            else:
                place = ''
                if Admin:    place += Admin+'\n'
                if Province: place += Province+'\n'
                if Country:  place += Country+'\n'
                allconf += confirmed
                alldeat += deaths
                allreco += recos
            if not confirmed: continue
        except:
            print('Line', i, ln[:4], "... skipped.")
            continue
        data.append([place, theta, phi, confirmed, deaths, recos])
    data.append(['US\n', 0.9, 1.4, conf_us, deat_us, reco_us])
    return data, allconf, alldeat, allreco


# -------------------------------------------------------------------------
# Create the scene:
from vtkplotter import *

data, allconf, alldeat, allreco = load_data()
for place, theta, phi, confd, deaths, recos in data:
    pos = spher2cart(1, theta, phi)
    fl = place + 'cases: '+str(confd)+'\ndeaths: '+str(deaths)
    Sphere(pos, np.power(confd,  1/3)/1000, alpha=0.4, res=12).flag(fl)
    Sphere(pos, np.power(deaths, 1/3)/1000, alpha=0.4, c='k', res=10)
    #Text(place, pos*1.01, s=0.003, c='w', justify='center').orientation(pos)

Text2D('COVID-19 spread on '+date
       +'\n#cases : '+str(allconf)
       +'\n#deaths: '+str(alldeat)
       +'\n#recovd: '+str(allreco), font="Overspray", s=0.9)
Earth()
show(..., axes=12, bg2='lb', zoom=1.7, elevation=-70, size='fullscreen')

