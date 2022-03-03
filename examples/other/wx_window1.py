import wx
from vtk.wx.wxVTKRenderWindowInteractor import wxVTKRenderWindowInteractor
import vedo

#####################################################
# Every wx app needs an app
app = wx.App(False)

# create the top-level frame, sizer and wxVTKRWI
frame = wx.Frame(None, -1, "vedo with wxpython", size=(600,600))
widget = wxVTKRenderWindowInteractor(frame, -1)
sizer = wx.BoxSizer(wx.VERTICAL)
sizer.Add(widget, 1, wx.EXPAND)
frame.SetSizer(sizer)
frame.Layout()

# It would be more correct (API-wise) to call widget.Initialize() and
# widget.Start() here, but Initialize() calls RenderWindow.Render().
# That Render() call will get through before we can setup the
# RenderWindow() to render via the wxWidgets-created context; this
# causes flashing on some platforms and downright breaks things on
# other platforms.  Instead, we call widget.Enable().
widget.Enable(1)
widget.AddObserver("ExitEvent", lambda o,e,f=frame: f.Close())

##################################################### vedo example
def func(evt):
    print("Event dump:\n", evt)
    plt.camera.Azimuth(10) # rotate one camera

cone = vedo.shapes.Cone(c='green8')
axes = vedo.Axes(cone, c='white')
cube = vedo.shapes.Cube()

# Create 2 subwindows with a cone and a cube
plt = vedo.Plotter(N=2, bg='blue2', bg2='blue8', wxWidget=widget)
plt.addCallback("right mouse click", func)
plt.at(0).add([cone, axes, "right-click anywhere"]).resetCamera()
plt.at(1).add(cube).resetCamera()
# plt.show() # vedo.show() is now disabled in wx

#####################################################
# Show everything
frame.Show()
app.MainLoop()
