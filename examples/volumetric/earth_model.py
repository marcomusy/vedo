"""Visualization of a discretized Earth model"""
import vedo

vedo.settings.defaultFont = 'Kanopus'

tet = vedo.TetMesh(vedo.dataurl+'earth_model.vtu')
conductor = tet.clone().threshold('cell_scalars', above=0, below=4)

# Crop the initial mesh
box = vedo.Box(size=[503500, 505000, 6414000, 6417000, -1830, 600])
tet.cutWithMesh(box, wholeCells=True)

# We need to build a look up table for our color bar
lut_table = [
    #value, color,   alpha, category_label
    ( 0.0, 'black',      1, "Cond_0"),
    ( 1.0, 'cyan',       1, "Cond_1"),
    ( 2.0, 'skyblue',    1, "Cond_2"),
    ( 3.0, 'dodgerblue', 1, "Cond_3"),
    ( 4.0, 'blue',       1, "Cond_4"),
    ( 5.0, 'gray',       1, "Overburden"),
    ( 6.0, 'yellow',     1, "Layer^A"),
    ( 7.0, 'gold',       1, "Layer^B"),
    ( 9.0, 'red',        1, "Layer^C"),
    (11.0, 'powderblue', 1, "Layer^D"),
    (13.0, 'lime',       1, "Layer^E"),
    (15.0, 'seagreen',   1, "Layer^V"),
]
lut = vedo.buildLUT(lut_table)

msh = tet.tomesh(shrink=0.95).cmap(lut, 'cell_scalars', on='cells')
msh.addScalarBar3D(
                categories=lut_table,
                pos=(505500, 6416900, -630),
                title='Units',
                titleSize=1.25,
                labelSize=1.5,
                s=[100, 2200],
)
# put scalarbar vertical, tell camera to keep bounds into account
msh.scalarbar.rotateX(90, around='itself').rotateZ(60, around='itself')
msh.scalarbar.useBounds()

# Create cmap for conductor
cond = conductor.tomesh().cmap(lut, 'cell_scalars', on='cells')

axes = vedo.Axes(
    msh + cond,
    xtitle='Easting (m)',
    ytitle='Northing (m)',
    ztitle='Elevation (m)',
    xTitlePosition=0.65,
    yTitlePosition=0.65,
    zTitlePosition=0.65,
    yTitleOffset=-0.22,
    zTitleOffset= 0.06,
    yLabelRotation=90,
    yLabelOffset=-1.5,
    zAxisRotation=15,
    axesLineWidth=3,
    gridLineWidth=2,
    yShiftAlongX=1,
    tipSize=0,
    yzGrid=True,
    xyFrameLine=True,
)

vedo.show(msh, cond, axes, __doc__, size=(1305, 1020),
          roll=-80, azimuth=50, elevation=-10, zoom=1.25).close()
