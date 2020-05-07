"""Recreate a model of a geothermal reservoir, Utah
(Credits: A. Pollack, SCRF)"""
from vtkplotter import Plotter, Mesh, Points, Line, Lines, printc
import pandas as pd
from scipy.spatial import Delaunay

#import the file from github
basegithuburl = 'https://raw.githubusercontent.com/ahinoamp/Example3DGeologicModelUsingVTKPlotter/master/'


#Load surfaces
printc("...loading data, please wait", invert=1)

fileVertices = basegithuburl+'land_surface_vertices.csv'
landSurfacePD =pd.read_csv(fileVertices)

fileVertices = basegithuburl+'175C_vertices.csv'
vertices_175CPD =pd.read_csv(fileVertices)

fileVertices = basegithuburl+'225C_vertices.csv'
vertices_225CPD =pd.read_csv(fileVertices)

fileVertices = basegithuburl+'Negro_Mag_Fault_vertices.csv'
Negro_Mag_Fault_verticesPD =pd.read_csv(fileVertices)

fileVertices = basegithuburl+'Opal_Mound_Fault_vertices.csv'
Opal_Mound_Fault_verticesPD =pd.read_csv(fileVertices)

fileVertices = basegithuburl+'top_granitoid_vertices.csv'
top_granitoid_verticesPD =pd.read_csv(fileVertices)

fileVertices = basegithuburl+'top_granitoid_vertices.csv'
border =pd.read_csv(fileVertices)

fileVertices = basegithuburl+'Microseismic.csv'
microseismic =pd.read_csv(fileVertices)

#The well path and different logs for the well paths
filepath = basegithuburl+'path5832.csv'
well_5832_path =pd.read_csv(filepath)

filepath = basegithuburl+'temperature5832.csv'
temp_well =pd.read_csv(filepath)

filepath = basegithuburl+'nphi5832.csv'
nphi_well =pd.read_csv(filepath)

filepath = basegithuburl+'pressure5832.csv'
pressure_well =pd.read_csv(filepath)

#Since most of the wells in the area were just vertical, I split them into two files:
#One file with the top of the wells and the other with the bottom point of the wellbore
file = basegithuburl + 'MinPointsWells.csv'
wellsmin =pd.read_csv(file)
file = basegithuburl + 'MaxPointsWells.csv'
wellsmax =pd.read_csv(file)

#Project boundary area on the surface
file = basegithuburl + 'FORGE_Border.csv'
border = pd.read_csv(file)


####################
## 1. land surface: a mesh with varying color
####################
printc("...analyzing...", invert=1)

#perform a 2D Delaunay triangulation to get the cells from the point cloud
tri = Delaunay(landSurfacePD.values[:, 0:2])

#create a mesh object for the land surface
landSurface = Mesh([landSurfacePD.values, tri.simplices])

#in order to color it by the elevation, we extract the z value
elevation = landSurface.cellCenters()[:, 2]   # pick z coordinates of cells
landSurface.cellColors(elevation, cmap='terrain', vmin=1000)

#We give the object a name
landSurface.name='Land Surface'

# Create a plot
plot = Plotter(axes=1, bg='white')
plot += landSurface.flag()


####################
## 2. Different meshes with constant colors
####################
#Mesh of 175 C isotherm
tri = Delaunay(vertices_175CPD.values[:, 0:2])
vertices_175C = Mesh([vertices_175CPD.values, tri.simplices]).c("orange").opacity(0.3)
vertices_175C.name='175C temperature isosurface'
plot += vertices_175C.flag()

#Mesh of 225 C isotherm
tri = Delaunay(vertices_225CPD.values[:, 0:2])
vertices_225CT = Mesh([vertices_225CPD.values, tri.simplices]).c("red").opacity(0.4)
vertices_225CT.name='225C temperature isosurface'
plot += vertices_225CT.flag()

#Negro fault
tri = Delaunay(Negro_Mag_Fault_verticesPD.values[:, 1:3])
Negro_Mag_Fault_vertices = Mesh([Negro_Mag_Fault_verticesPD.values, tri.simplices]).c("f").opacity(0.4)
Negro_Mag_Fault_vertices.name='Negro Fault'
plot += Negro_Mag_Fault_vertices.flag()

#Opal fault
tri = Delaunay(Opal_Mound_Fault_verticesPD.values[:, 1:3])
Opal_Mound_Fault_vertices = Mesh([Opal_Mound_Fault_verticesPD.values, tri.simplices]).c("g").opacity(0.4)
Opal_Mound_Fault_vertices.name='Opal Mound Fault'
plot += Opal_Mound_Fault_vertices.flag()

#Top Granite
xyz = top_granitoid_verticesPD.values
xyz[:, 2] = top_granitoid_verticesPD.values[:,2]-20
tri = Delaunay(top_granitoid_verticesPD.values[:, 0:2])
top_granitoid_vertices = Mesh([xyz, tri.simplices]).c("darkcyan")
top_granitoid_vertices.name='Top of granite surface'
plot += top_granitoid_vertices.flag()

####################
## 3. Point objects
####################
printc("...plotting...", invert=1)

#FORGE Boundary
#Since the boundary area did not have a Z column, I assigned a Z value for where I wanted it to appear
border['zcoord'] = 1650
borderxyz = border[['xcoord', 'ycoord', 'zcoord']]
boundary = Points(borderxyz.values).c('k')
boundary.name='FORGE area boundary'
plot+=boundary.flag()

#Microseismic
microseismicxyz = microseismic[['xloc','yloc','zloc']]
scals = microseismic[['mw']]
microseismicPts = Points(microseismicxyz.values, r=3).cellColors(scals, cmap="jet")
microseismicPts.name='Microseismic events'
plot+=microseismicPts.flag()

####################
## 4. Line objects
####################
#The path of well 58_32
xyz = well_5832_path[['X', 'Y', 'Z']].values
Well = Line(xyz)
Well.name='Well 58-32'
plot+=Well.flag()

#A porosity log in the well
xyz = nphi_well[['X', 'Y', 'Z']].values
porosity = nphi_well['Nphi'].values
Well = Line(xyz).c('gold').lw(2)
Well.name='Porosity log well 58-32'
plot+=Well.flag()

#This well data is actually represented by points since as of right now,
#since the k3d embedding does not support colors on the lines, and I wanted to show the colors
xyz = pressure_well[['X', 'Y', 'Z']].values
pressure = pressure_well['Pressure'].values
Well = Points(xyz, r=1).pointColors(pressure, cmap="cool")
Well.name='Pressure log well 58-32'
plot+=Well.flag()

#Temperatue log
xyz = temp_well[['X', 'Y', 'Z']].values
scals = temp_well['Temperature'].values
Well = Points(xyz, r=1).pointColors(scals, cmap="seismic")
Well.name='Temperature log well 58-32'
plot+=Well.flag()


####################
## 5. Multi-line objects
####################
#There is some preprocessing that needs to be done here in order to get two lists of points
#defining the start and end of the lines that will be representing the wellbores
xyzmin = wellsmin[['x', 'y', 'z']].values
xyzmax = wellsmax[['x', 'y', 'z']].values

Wells = Lines(xyzmin, xyzmax, c='gray', alpha=1, lw=3)
Wells.name='Pre-existing wellbores'
plot+=Wells.flag()

####################
## 6. Done. show the plot
####################
plot += __doc__
plot.show(viewup='z')
