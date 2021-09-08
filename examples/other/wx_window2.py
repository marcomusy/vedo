import wx
import vedo
from vtk.wx.wxVTKRenderWindowInteractor import wxVTKRenderWindowInteractor

##################################################### wx app
app = wx.App(False)
frame = wx.Frame(None, -1, "vedo with wxpython", size=(800,800))
widget = wxVTKRenderWindowInteractor(frame, -1)
sizer = wx.BoxSizer(wx.VERTICAL)
sizer.Add(widget, 1, wx.EXPAND)
frame.SetSizer(sizer)
frame.Layout()
widget.Enable(1)
widget.AddObserver("ExitEvent", lambda o,e,f=frame: f.Close())

##################################################### vedo
def funcMove(event):
    mesh = event.actor
    if not mesh: return

    ptid = mesh.closestPoint(event.picked3d, returnPointId=True)
    txt = f"Probed point:\n{vedo.utils.precision(event.picked3d, 3)}\n" \
          f"value = {vedo.utils.precision(arr[ptid], 2)}"

    vpt = vedo.shapes.Sphere(mesh.points(ptid), r=0.01, c='orange2').pickable(False)
    vig = vpt.vignette(txt, s=.05, offset=(0.5,0.5), font="VictorMono").followCamera()

    msg.text(txt)               # update the 2d text message
    plt.remove(plt.actors[-2:]).add([vpt, vig]) # remove last 2 objects, add the new ones
    widget.Render()             # need to manually call Render

msg = vedo.Text2D(pos='bottom-left', font="VictorMono")
msh = vedo.shapes.ParametricShape("RandomHills").cmap('terrain')
axs = vedo.Axes(msh)
arr = msh.pointdata["Scalars"]

plt = vedo.Plotter(bg='moccasin', bg2='blue9', wxWidget=widget)
plt.add([msh, axs, msg]).resetCamera()
plt.actors += [None,None,None]  # place holder for sphere, vignette, text2d
plt.addCallback('MouseMove', funcMove)

#####################################################
# Show everything
frame.Show()
app.MainLoop()