import vtk
from vtkplotter.addons import addScalarBar
from vtkplotter.plotter import Plotter
from vtkplotter.pyplot import cornerHistogram
from vtkplotter.utils import mag, precision
from vtkplotter.colors import printc
from vtkplotter.shapes import Text2D
import numpy as np

__all__ = ["slicer", "slicer2d"]

_cmap_slicer='gist_ncar_r'

def slicer(volume,
           alpha=1,
           cmaps=('gist_ncar_r', "hot_r", "bone_r", "jet", "Spectral_r"),
           map2cells=False,  # buggy
           clamp=True,
           useSlider3D=False,
           size=(850,700),
           screensize="auto",
           title="",
           bg="white",
           bg2="lightblue",
           axes=1,
           showHisto=True,
           showIcon=True,
           draggable=False,
           verbose=True,
           ):
    """
    Generate a ``Plotter`` window with slicing planes for the input Volume.
    Returns the ``Plotter`` object.

    :param float alpha: transparency of the slicing planes
    :param list cmaps: list of color maps names to cycle when clicking button
    :param bool map2cells: scalars are mapped to cells, not intepolated.
    :param bool clamp: clamp scalar to reduce the effect of tails in color mapping
    :param bool useSlider3D: show sliders attached along the axes
    :param list size: rendering window size in pixels
    :param list screensize: size of the screen can be specified
    :param str title: window title
    :param bg: background color
    :param bg2: background gradient color
    :param int axes: axis type number
    :param bool showHisto: show histogram on bottom left
    :param bool showIcon: show a small 3D rendering icon of the volume
    :param bool draggable: make the icon draggable
    """
    global _cmap_slicer

    if verbose: printc("Slicer tool", invert=1, c="m")
    ################################
    vp = Plotter(bg=bg, bg2=bg2,
                 size=size,
                 screensize=screensize,
                 title=title,
                 interactive=False,
                 verbose=verbose)

    ################################
    box = volume.box().wireframe().alpha(0)

    vp.show(box, viewup="z", axes=axes)
    if showIcon:
        vp.showInset(volume, pos=(.85,.85), size=0.15, c='w', draggable=draggable)

    # inits
    la, ld = 0.7, 0.3 #ambient, diffuse
    dims = volume.dimensions()
    data = volume.getPointArray()
    rmin, rmax = volume.imagedata().GetScalarRange()
    sbwidth = 800
    if clamp:
        hdata, edg = np.histogram(data, bins=50)
        logdata = np.log(hdata+1)
        # mean  of the logscale plot
        meanlog = np.sum(np.multiply(edg[:-1], logdata))/np.sum(logdata)
        rmax = min(rmax, meanlog+(meanlog-rmin)*0.9)
        rmin = max(rmin, meanlog-(rmax-meanlog)*0.9)
        if verbose:
            printc('scalar range clamped to: (' +
                   precision(rmin, 3) +', '+  precision(rmax, 3)+')', c='m', bold=0)
    _cmap_slicer = cmaps[0]
    visibles = [None, None, None]
    msh = volume.zSlice(int(dims[2]/2))
    msh.alpha(alpha).lighting('', la, ld, 0)
    msh.pointColors(cmap=_cmap_slicer, vmin=rmin, vmax=rmax)
    if map2cells: msh.mapPointsToCells()
    vp.renderer.AddActor(msh)
    visibles[2] = msh
    addScalarBar(msh, pos=(0.04,0.0), horizontal=True, width=sbwidth,
                 vmin=rmin, vmax=rmax, titleFontSize=0)

    def sliderfunc_x(widget, event):
        i = int(widget.GetRepresentation().GetValue())
        msh = volume.xSlice(i).alpha(alpha).lighting('', la, ld, 0)
        msh.pointColors(cmap=_cmap_slicer, vmin=rmin, vmax=rmax)
        if map2cells: msh.mapPointsToCells()
        vp.renderer.RemoveActor(visibles[0])
        if i and i<dims[0]: vp.renderer.AddActor(msh)
        visibles[0] = msh

    def sliderfunc_y(widget, event):
        i = int(widget.GetRepresentation().GetValue())
        msh = volume.ySlice(i).alpha(alpha).lighting('', la, ld, 0)
        msh.pointColors(cmap=_cmap_slicer, vmin=rmin, vmax=rmax)
        if map2cells: msh.mapPointsToCells()
        vp.renderer.RemoveActor(visibles[1])
        if i and i<dims[1]: vp.renderer.AddActor(msh)
        visibles[1] = msh

    def sliderfunc_z(widget, event):
        i = int(widget.GetRepresentation().GetValue())
        msh = volume.zSlice(i).alpha(alpha).lighting('', la, ld, 0)
        msh.pointColors(cmap=_cmap_slicer, vmin=rmin, vmax=rmax)
        if map2cells: msh.mapPointsToCells()
        vp.renderer.RemoveActor(visibles[2])
        if i and i<dims[2]: vp.renderer.AddActor(msh)
        visibles[2] = msh

    cx, cy, cz, ch = 'dr', 'dg', 'db', (0.3,0.3,0.3)
    if np.sum(vp.renderer.GetBackground()) < 1.5:
        cx, cy, cz = 'lr', 'lg', 'lb'
        ch = (0.8,0.8,0.8)

    if not useSlider3D:
        vp.addSlider2D(sliderfunc_x, 0, dims[0], title='X',
                       pos=[(0.8,0.12), (0.95,0.12)], showValue=False, c=cx)
        vp.addSlider2D(sliderfunc_y, 0, dims[1], title='Y',
                       pos=[(0.8,0.08), (0.95,0.08)], showValue=False, c=cy)
        vp.addSlider2D(sliderfunc_z, 0, dims[2], title='Z', value=int(dims[2]/2),
                       pos=[(0.8,0.04), (0.95,0.04)], showValue=False, c=cz)
    else: # 3d sliders attached to the axes bounds
        bs = box.bounds()
        vp.addSlider3D(sliderfunc_x,
            pos1=(bs[0], bs[2], bs[4]),
            pos2=(bs[1], bs[2], bs[4]),
            xmin=0, xmax=dims[0],
            t=box.diagonalSize()/mag(box.xbounds())*0.6,
            c=cx,
            showValue=False,
        )
        vp.addSlider3D(sliderfunc_y,
            pos1=(bs[1], bs[2], bs[4]),
            pos2=(bs[1], bs[3], bs[4]),
            xmin=0, xmax=dims[1],
            t=box.diagonalSize()/mag(box.ybounds())*0.6,
            c=cy,
            showValue=False,
        )
        vp.addSlider3D(sliderfunc_z,
            pos1=(bs[0], bs[2], bs[4]),
            pos2=(bs[0], bs[2], bs[5]),
            xmin=0, xmax=dims[2],
            value=int(dims[2]/2),
            t=box.diagonalSize()/mag(box.zbounds())*0.6,
            c=cz,
            showValue=False,
        )


    #################
    def buttonfunc():
        global _cmap_slicer
        bu.switch()
        _cmap_slicer = bu.status()
        for mesh in visibles:
            if mesh:
                mesh.pointColors(cmap=_cmap_slicer, vmin=rmin, vmax=rmax)
                if map2cells:
                    mesh.mapPointsToCells()
        vp.renderer.RemoveActor(mesh.scalarbar)
        mesh.scalarbar = addScalarBar(mesh,
                                      pos=(0.04,0.0),
                                      horizontal=True,
                                      width=sbwidth,
                                      vmin=rmin, vmax=rmax,
                                      titleFontSize=0)
        vp.renderer.AddActor(mesh.scalarbar)

    bu = vp.addButton(buttonfunc,
        pos=(0.27, 0.005),
        states=cmaps,
        c=["db"]*len(cmaps),
        bc=["lb"]*len(cmaps),  # colors of states
        size=14,
        bold=True,
    )

    #################
    hist = None
    if showHisto:
        hist = cornerHistogram(data, s=0.2,
                               bins=25, logscale=1, pos=(0.02, 0.02),
                               c=ch, bg=ch, alpha=0.7)

#    vp.backgroundRenderer = vtk.vtkRenderer()
#    vp.backgroundRenderer.SetLayer(1)
#    vp.backgroundRenderer.InteractiveOff()
#    actor = histogram(data, logscale=1, alpha=0.8).scale(4/150)
#    vp.window.AddRenderer(vp.backgroundRenderer)
#    vp.backgroundRenderer.AddActor(actor)
#    cam = vtk.vtkCamera()
#    cam.SetPosition( -10,-10, 40)
#    #actor.SetPosition(-4,-4,  0)
#    actor.SetOrigin(-8,-8,0)
#    cam.SetParallelScale(1)
#    vp.backgroundRenderer.SetActiveCamera(cam)
#    vp.backgroundRenderer.ResetCameraClippingRange()

    comment = None
    if verbose:
        comment = Text2D("Use sliders to slice volume\nClick button to change colormap",
                         font='Montserrat', s=0.8)

    vp.show(msh, hist, comment, interactive=False)
    vp.interactive = True
    if verbose:
        printc("Press button to cycle through color maps:\n", cmaps, c="m")
        printc("Use sliders to select the slicing planes.", c="m")
    return vp



def slicer2d(volume, size=(900,900), bg=(0.6,0.6,0.7), zoom=1.3):
    """Create a 2D window with a single balck a nd white
    slice of a Volume, wich can be oriented arbitrarily in space.
    """
    img = volume.imagedata()

    ren1 = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren1)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    im = vtk.vtkImageResliceMapper()
    im.SetInputData(img)
    im.SliceFacesCameraOn()
    im.SliceAtFocalPointOn()
    im.BorderOn()

    ip = vtk.vtkImageProperty()
    ip.SetInterpolationTypeToLinear()

    ia = vtk.vtkImageSlice()
    ia.SetMapper(im)
    ia.SetProperty(ip)

    ren1.AddViewProp(ia)
    ren1.SetBackground(bg)
    renWin.SetSize(size)

    iren = vtk.vtkRenderWindowInteractor()
    style = vtk.vtkInteractorStyleImage()
    style.SetInteractionModeToImage3D()
    iren.SetInteractorStyle(style)
    renWin.SetInteractor(iren)

    renWin.Render()
    cam1 = ren1.GetActiveCamera()
    cam1.ParallelProjectionOn()
    ren1.ResetCameraClippingRange()
    cam1.Zoom(zoom)
    renWin.Render()

    printc("Slicer2D tool", invert=1, c="m")
    printc(
        """Press  SHIFT+Left mouse    to rotate the camera for oblique slicing
       SHIFT+Middle mouse  to slice perpendicularly through the image
       Left mouse and Drag to modify luminosity and contrast
       X                   to Reset to sagittal view
       Y                   to Reset to coronal view
       Z                   to Reset to axial view
       R                   to Reset the Window/Levels
       Q                   to Quit.""",
        c="m",
    )

    iren.Start()
    return iren