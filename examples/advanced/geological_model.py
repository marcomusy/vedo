"""Recreate a model of a geothermal reservoir, Utah
(Credits: A. Pollack, SCRF)"""
from vedo import printc, dataurl, settings, delaunay2D, Line, Lines, Points, Plotter
import pandas as pd

settings.useDepthPeeling = True

# Load surfaces, import the file from github
printc("...loading data...", invert=1, end='')
url = "https://raw.githubusercontent.com/ahinoamp/Example3DGeologicModelUsingVTKPlotter/master/"

landSurfacePD   = pd.read_csv(url+"land_surface_vertices.csv")
vertices_175CPD = pd.read_csv(url+"175C_vertices.csv")
vertices_225CPD = pd.read_csv(url+"225C_vertices.csv")
microseismic    = pd.read_csv(url+"Microseismic.csv")
Negro_Mag_Fault_verticesPD = pd.read_csv(url+"Negro_Mag_Fault_vertices.csv")
Opal_Mound_Fault_verticesPD= pd.read_csv(url+"Opal_Mound_Fault_vertices.csv")
top_granitoid_verticesPD   = pd.read_csv(url+"top_granitoid_vertices.csv")

# The well path and different logs for the well paths
well_5832_path= pd.read_csv(url+"path5832.csv")
pressure_well = pd.read_csv(url+"pressure5832.csv")
temp_well     = pd.read_csv(url+"temperature5832.csv")
nphi_well     = pd.read_csv(url+"nphi5832.csv")

# Since most of the wells in the area were just vertical, I split them into two files:
# One file with the top of the wells and the other with the bottom point of the wellbore
wellsmin = pd.read_csv(url+"MinPointsWells.csv")
wellsmax = pd.read_csv(url+"MaxPointsWells.csv")

# Project boundary area on the surface
border = pd.read_csv(url+"FORGE_Border.csv")

#############################################
## land surface: a mesh with varying color
printc("analyzing...", invert=1, end='')

# create a mesh object from the 2D Delaunay triangulation of the point cloud
landSurface = delaunay2D(landSurfacePD.values)

# in order to color it by the elevation, we use the z values of the mesh
zvals = landSurface.points()[:, 2]
landSurface.cmap("terrain", zvals, vmin=1100)
landSurface.name = "Land Surface" # give the object a name

# Create a plotter and add landSurface to it
plt = Plotter(axes=dict(xtitle='km', ytitle=' ', ztitle='km*1.5', yzGrid=False),
              bg2='lb', size=(1200,900)) # screen size
plt += landSurface.flag()                # this adds a flag when hoovering the mouse
plt += landSurface.isolines(5).lw(1).c('k')

#############################################
## Different meshes with constant colors
# Mesh of 175 C isotherm
vertices_175C = delaunay2D(vertices_175CPD.values)
vertices_175C.name = "175C temperature isosurface"
plt += vertices_175C.c("orange").opacity(0.3).flag()

# Mesh of 225 C isotherm
vertices_225CT = delaunay2D(vertices_225CPD.values)
vertices_225CT.name = "225C temperature isosurface"
plt += vertices_225CT.c("red").opacity(0.4).flag()

# Negro fault, mode=fit is used because point cloud is not in xy plane
Negro_Mag_Fault_vertices = delaunay2D(Negro_Mag_Fault_verticesPD.values, mode='fit')
Negro_Mag_Fault_vertices.name = "Negro Fault"
plt += Negro_Mag_Fault_vertices.c("f").opacity(0.6).flag()

# Opal fault
Opal_Mound_Fault_vertices = delaunay2D(Opal_Mound_Fault_verticesPD.values, mode='fit')
Opal_Mound_Fault_vertices.name = "Opal Mound Fault"
plt += Opal_Mound_Fault_vertices.c("g").opacity(0.6).flag()

# Top Granite, (shift it a bit to avoid overlapping)
xyz = top_granitoid_verticesPD.values - [0,0,20]
top_granitoid_vertices = delaunay2D(xyz).texture(dataurl+'textures/paper2.jpg')
top_granitoid_vertices.name = "Top of granite surface"
plt += top_granitoid_vertices.flag()

###################################################
printc("plotting...", invert=1)

# Microseismic
microseismicxyz = microseismic[["xloc", "yloc", "zloc"]].values
scals = microseismic[["mw"]]
microseismicPts = Points(microseismicxyz, r=5).cmap("jet", scals)
microseismicPts.name = "Microseismic events"
plt += microseismicPts.flag()

# FORGE Boundary. Since the boundary area did not have a Z column,
# I assigned a Z value for where I wanted it to appear
border["zcoord"] = 1650
borderxyz = border[["xcoord", "ycoord", "zcoord"]]
boundary = Line(borderxyz.values).extrude(zshift=120, cap=False)
boundary.lw(0).texture(dataurl+'textures/wood1.jpg')
boundary.name = "FORGE area boundary"
plt += boundary.flag()

# The path of well 58_32
Well1 = Line(well_5832_path[["X", "Y", "Z"]].values, lw=2, c='k')
Well1.name = "Well 58-32"
plt += Well1.flag()

# A porosity log in the well
xyz = nphi_well[["X", "Y", "Z"]].values
porosity = nphi_well["Nphi"].values
Well2 = Line(xyz, lw=3).cmap("hot", porosity)
Well2.name = "Porosity log well 58-32"
plt += Well2.flag()

# This well data is actually represented by points since as of right now,
xyz = pressure_well[["X", "Y", "Z"]].values
pressure = pressure_well["Pressure"].values
Well3 = Line(xyz, lw=3).cmap("cool", pressure)
Well3.name = "Pressure log well 58-32"
plt += Well3.flag()

# Temperature log
xyz = temp_well[["X", "Y", "Z"]].values
temp = temp_well["Temperature"].values
Well4 = Line(xyz, lw=3).cmap("seismic", temp)
Well4.name = "Temperature log well 58-32"
plt += Well4.flag()

# defining the start and end of the lines that will be representing the wellbores
Wells = Lines(wellsmin[["x", "y", "z"]].values, # start points
              wellsmax[["x", "y", "z"]].values, # end points
              c="gray", alpha=1, lw=3)
Wells.name = "Pre-existing wellbores"
plt += Wells.flag()

for a in plt.actors:
    # change scale to kilometers in x and y, but expand z scale by 1.5!
    a.scale([0.001, 0.001, 0.001*1.5])

#########################
## show the plot
plt += __doc__
plt.show(viewup="z", zoom=1.2)
#plt.export("page.html") # k3d is the default
plt.close()
