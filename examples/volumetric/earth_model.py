"""Visualization of a discretized Earth model"""
import vedo

vedo.settings.default_font = 'Kanopus'

tet = vedo.TetMesh(vedo.dataurl+'earth_model.vtu')
conductor = tet.clone().threshold('cell_scalars', above=0, below=4)

# Crop the initial mesh
box = vedo.Box([503500, 505000, 6414000, 6417000, -1830, 600])
tet.cut_with_mesh(box, whole_cells=True)

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
lut = vedo.build_lut(lut_table, interpolate=1)

msh = tet.tomesh(shrink=0.95, fill=True)
msh.cmap(lut, 'cell_scalars', on='cells')
msh.add_scalarbar3d(
    categories=lut_table,
    pos=(505700, 6417950, -1630),
    title='Units',
    title_size=1.25,
    label_size=1.5,
    size=[100, 2200],
)

# Put scalarbar vertical, tell camera to keep bounds into account
# msh.scalarbar.rotate_x(90).rotate_z(60).use_bounds(True)

# OR: use clone2d to create a 2D scalarbar from the 3D one
msh.scalarbar = msh.scalarbar.clone2d(pos=[0.7,-0.95], size=0.3)
 
# Create cmap for conductor
cond = conductor.tomesh().cmap(lut, 'cell_scalars', on='cells')

axes = vedo.Axes(
    msh + cond,
    xtitle='Easting (m)',
    ytitle='Northing (m)',
    ztitle='Elevation (m)',
    xtitle_position=0.65,
    ytitle_position=0.65,
    ztitle_position=0.65,
    ytitle_offset=-0.22,
    ztitle_offset= 0.06,
    ylabel_rotation=90,
    ylabel_offset=-1.5,
    zaxis_rotation=15,
    axes_linewidth=3,
    grid_linewidth=2,
    yshift_along_x=1,
    tip_size=0,
    yzgrid=True,
    xyframe_line=True,
)

vedo.show(msh, cond, axes, __doc__, size=(1305, 1020),
          roll=-80, azimuth=50, elevation=-10, zoom=1.2).close()
