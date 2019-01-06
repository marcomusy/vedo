########################################### imports
from vtkplotter import Plotter
from vtkplotter.analysis import boundaries
try:
	# Credits:
	# https://github.com/MarcoAttene/MeshFix-V2.1
	# https://github.com/akaszynski/pymeshfix
	from pymeshfix._meshfix import PyTMesh, Repair
	from pymeshfix.examples import bunny_scan
except:
	print('Install pymeshfix with: pip install pymeshfix')
	exit()

########################################### PyTMesh repair
inputmesh = bunny_scan # 'pymeshfix/examples/StanfordBunny.ply'
ouputmesh = 'meshfix_repaired.ply' # try e.g. vtkconvert -to vtk repaired.ply

tm = PyTMesh()
tm.LoadFile(inputmesh)
Repair(tm, verbose=True, joincomp=True, removeSmallestComponents=True)
tm.SaveFile(ouputmesh)

########################################### vtkplotter
vp = Plotter(shape=(2,1))

act_original = vp.load(inputmesh)
bo = boundaries(act_original)

act_repaired = vp.load(ouputmesh)
br = boundaries(act_repaired)

vp.show([act_original, bo], at=0)
vp.show([act_repaired, br], at=1, interactive=1)

