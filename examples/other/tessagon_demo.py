"""RhombusTessagon on Klein's bottle
DodecaTessagon on hyperboloid
FloretTessagon on torus surfaces
"""
# install with:
# pip install tessagon
# See instructions at: https://github.com/cwant/tessagon
from tessagon.types.rhombus_tessagon import RhombusTessagon
from tessagon.types.dodeca_tessagon import DodecaTessagon
from tessagon.types.floret_tessagon import FloretTessagon
from tessagon.adaptors.vtk_adaptor import VtkAdaptor
from tessagon.misc.shapes import general_torus, one_sheet_hyperboloid, klein, warp_var

from vedo import Mesh, show


# ---------------------------------------------------------
options = dict(
    u_range=[0.0, 1.0],
    v_range=[0.0, 1.0],
    u_num=40,
    v_num=6,
    v_twist=True,
    function=klein,
    adaptor_class=VtkAdaptor,
)
poly_data = RhombusTessagon(**options).create_mesh()
rhombus = Mesh(poly_data).x(-5).computeNormals()
rhombus.lineWidth(1).backColor('tomato')


# ---------------------------------------------------------
options = dict(
    u_range=[-1.0, 1.0],
    v_range=[ 0.0, 1.0],
    u_num=4,
    v_num=10,
    u_cyclic=False,
    v_cyclic=True,
    function=one_sheet_hyperboloid,
    adaptor_class=VtkAdaptor,
)
poly_data = DodecaTessagon(**options).create_mesh()
dodeca = Mesh(poly_data).x(5).computeNormals()
dodeca.lineWidth(1).backColor('tomato')


# ---------------------------------------------------------
def chubby_torus(u, v):
    return general_torus(5, 1.5, v, warp_var(u, 0.2))

options = dict(
    u_range=[0.0, 1.0],
    v_range=[0.0, 1.0],
    u_num=2,
    v_num=12,
    color_pattern=1,
    function=chubby_torus,
    adaptor_class=VtkAdaptor,
)
poly_data = FloretTessagon(**options).create_mesh()
poly_data.GetCellData().GetScalars().SetName("color_pattern")
floret = Mesh(poly_data).reverse().y(-9).scale(0.7)
floret.cmap('Greens_r', input_array="color_pattern", on='cells').lineWidth(0.1)

# ---------------------------------------------------------
show(rhombus, dodeca, floret, __doc__, axes=1)

