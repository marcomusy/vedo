from vedo import *
lines = load("https://discourse.vtk.org/uploads/short-url/nC2RjJgTerpHKR0jD02Na6BRHVl.vtp")
# for k in lines.pointdata.keys(): 
#     print("array:", k, lines.pointdata[k].dtype)
lines.cmap("rainbow", "cluster_idx")
show(lines, axes=1, bg='blackboard')