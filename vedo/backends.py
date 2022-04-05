import os

import numpy
import vedo
import vedo.colors as colors
import vedo.shapes as shapes
import vedo.utils as utils
import vtk
from vedo import settings
from vedo.mesh import Mesh
from vedo.pointcloud import Points
from vedo.volume import Volume

__all__ = []


def getNotebookBackend(actors2show, zoom, viewup):

    plt = vedo.plotter_instance

    if zoom == 'tight':
        zoom=1 # disable it

    if isinstance(plt.shape, str) or sum(plt.shape) > 2:
        vedo.logger.error("Multirendering is not supported in jupyter")
        return


    ####################################################################################
    # https://github.com/InsightSoftwareConsortium/itkwidgets
    #  /blob/master/itkwidgets/widget_viewer.py
    if 'itk' in vedo.notebookBackend:
        from itkwidgets import view

        vedo.notebook_plotter = view(actors=actors2show,
                                         cmap='jet', ui_collapsed=True,
                                         gradient_opacity=False)


    ####################################################################################
    elif vedo.notebookBackend == 'k3d':
        try:
            import k3d # https://github.com/K3D-tools/K3D-jupyter
        except:
            print("Cannot find k3d, install with:  pip install k3d")
            return

        actors2show2 = []
        for ia in actors2show:
            if not ia:
                continue
            if isinstance(ia, vtk.vtkAssembly): #unpack assemblies
                acass = ia.unpack()
                actors2show2 += acass
            else:
                actors2show2.append(ia)

        # vbb, sizes, _, _ = addons.computeVisibleBounds()
        # kgrid = vbb[0], vbb[2], vbb[4], vbb[1], vbb[3], vbb[5]

        vedo.notebook_plotter = k3d.plot(
            axes=['x', 'y', 'z'],
            menu_visibility=settings.k3dMenuVisibility,
            height=settings.k3dPlotHeight,
            antialias=settings.k3dAntialias,
        )
        # vedo.notebook_plotter.grid = kgrid
        vedo.notebook_plotter.lighting = settings.k3dLighting

        # set k3d camera
        vedo.notebook_plotter.camera_auto_fit = settings.k3dCameraAutoFit
        vedo.notebook_plotter.grid_auto_fit = settings.k3dGridAutoFit

        vedo.notebook_plotter.axes_helper = settings.k3dAxesHelper

        if vedo.plotter_instance and vedo.plotter_instance.camera:
            k3dc =  utils.vtkCameraToK3D(vedo.plotter_instance.camera)
            if zoom:
                k3dc[0] /= zoom
                k3dc[1] /= zoom
                k3dc[2] /= zoom
            vedo.notebook_plotter.camera = k3dc
        # else:
        #     vsx, vsy, vsz = vbb[0]-vbb[1], vbb[2]-vbb[3], vbb[4]-vbb[5]
        #     vss = numpy.linalg.norm([vsx, vsy, vsz])
        #     if zoom:
        #         vss /= zoom
        #     vfp = (vbb[0]+vbb[1])/2, (vbb[2]+vbb[3])/2, (vbb[4]+vbb[5])/2 # camera target
        #     if viewup == 'z':
        #         vup = (0,0,1) # camera up vector
        #         vpos= vfp[0] + vss/1.9, vfp[1] + vss/1.9, vfp[2]+vss*0.01  # camera position
        #     elif viewup == 'x':
        #         vup = (1,0,0)
        #         vpos= vfp[0]+vss*0.01, vfp[1] + vss/1.5, vfp[2]  # camera position
        #     else:
        #         vup = (0,1,0)
        #         vpos= vfp[0]+vss*0.01, vfp[1]+vss*0.01, vfp[2] + vss/1.5  # camera position
        #     vedo.notebook_plotter.camera = [vpos[0], vpos[1], vpos[2],
        #                                           vfp[0],  vfp[1],  vfp[2],
        #                                           vup[0],  vup[1],  vup[2] ]
        if not plt.axes:
            vedo.notebook_plotter.grid_visible = False

        for ia in actors2show2:

            if isinstance(ia, (vtk.vtkCornerAnnotation, vtk.vtkAssembly)):
                continue

            kobj = None
            kcmap= None
            name = None
            if hasattr(ia, 'filename'):
                if ia.filename:
                    name = os.path.basename(ia.filename)
                if ia.name:
                    name = os.path.basename(ia.name)

            #####################################################################scalars
            # work out scalars first, Points Lines are also Mesh objs
            if isinstance(ia, (Mesh, shapes.Line, Points)):
#                print('scalars', ia.name, ia.N())
                iap = ia.GetProperty()

                if isinstance(ia, (shapes.Line, Points)):
                    iapoly = ia.polydata()
                else:
                    iapoly = ia.clone().clean().triangulate().computeNormals().polydata()

                vtkscals = None
                color_attribute = None
                if ia.mapper().GetScalarVisibility():
                    vtkdata = iapoly.GetPointData()
                    vtkscals = vtkdata.GetScalars()

                    if vtkscals is None:
                        vtkdata = iapoly.GetCellData()
                        vtkscals = vtkdata.GetScalars()
                        if vtkscals is not None:
                            c2p = vtk.vtkCellDataToPointData()
                            c2p.SetInputData(iapoly)
                            c2p.Update()
                            iapoly = c2p.GetOutput()
                            vtkdata = iapoly.GetPointData()
                            vtkscals = vtkdata.GetScalars()

                    if vtkscals is not None:
                        if not vtkscals.GetName():
                            vtkscals.SetName('scalars')
                        scals_min, scals_max = ia.mapper().GetScalarRange()
                        color_attribute = (vtkscals.GetName(), scals_min, scals_max)
                        lut = ia.mapper().GetLookupTable()
                        lut.Build()
                        kcmap=[]
                        nlut = lut.GetNumberOfTableValues()
                        for i in range(nlut):
                            r,g,b,a = lut.GetTableValue(i)
                            kcmap += [i/(nlut-1), r,g,b]


            #####################################################################Volume
            if isinstance(ia, Volume):
#                print('Volume', ia.name, ia.dimensions())
                kx, ky, kz = ia.dimensions()
                arr = ia.pointdata[0]
                kimage = arr.reshape(-1, ky, kx)

                colorTransferFunction = ia.GetProperty().GetRGBTransferFunction()
                kcmap=[]
                for i in range(128):
                    r,g,b = colorTransferFunction.GetColor(i/127)
                    kcmap += [i/127, r,g,b]

                kbounds = numpy.array(ia.imagedata().GetBounds()) \
                    + numpy.repeat(numpy.array(ia.imagedata().GetSpacing()) / 2.0, 2)\
                    * numpy.array([-1,1] * 3)

                kobj = k3d.volume(kimage.astype(numpy.float32),
                                  color_map=kcmap,
                                  #color_range=ia.imagedata().GetScalarRange(),
                                  alpha_coef=10,
                                  bounds=kbounds,
                                  name=name,
                                  )
                vedo.notebook_plotter += kobj

            #####################################################################text
            elif hasattr(ia, 'info') and 'formula' in ia.info.keys():
                pos = (ia.GetPosition()[0],ia.GetPosition()[1])
                kobj = k3d.text2d(ia.info['formula'], position=pos)
                vedo.notebook_plotter += kobj


            #####################################################################Mesh
            elif isinstance(ia, Mesh) and ia.N() and len(ia.faces()):
                # print('Mesh', ia.name, ia.N(), len(ia.faces()))
                kobj = k3d.vtk_poly_data(iapoly,
                                         name=name,
                                         # color=_rgb2int(iap.GetColor()),
                                         color_attribute=color_attribute,
                                         color_map=kcmap,
                                         opacity=iap.GetOpacity(),
                                         wireframe=(iap.GetRepresentation()==1))

                if iap.GetInterpolation() == 0:
                    kobj.flat_shading = True
                vedo.notebook_plotter += kobj

            #####################################################################Points
            elif isinstance(ia, Points):
                # print('Points', ia.name, ia.N())
                kcols=[]
                if color_attribute is not None:
                    scals = utils.vtk2numpy(vtkscals)
                    kcols = k3d.helpers.map_colors(scals, kcmap,
                                                   [scals_min,scals_max]).astype(numpy.uint32)
                # sqsize = numpy.sqrt(numpy.dot(sizes, sizes))

                kobj = k3d.points(ia.points().astype(numpy.float32),
                                  color=_rgb2int(iap.GetColor()),
                                  colors=kcols,
                                  opacity=iap.GetOpacity(),
                                  shader=settings.k3dPointShader,
                                  point_size=iap.GetPointSize(),
                                  name=name,
                                  )
                vedo.notebook_plotter += kobj


            #####################################################################Lines
            elif ia.polydata(False).GetNumberOfLines():
                # print('Line', ia.name, ia.N(), len(ia.faces()),
                #       ia.polydata(False).GetNumberOfLines(), len(ia.lines()),
                #       color_attribute, [vtkscals])

                # kcols=[]
                # if color_attribute is not None:
                #     scals = utils.vtk2numpy(vtkscals)
                #     kcols = k3d.helpers.map_colors(scals, kcmap,
                #                                    [scals_min,scals_max]).astype(numpy.uint32)

                # sqsize = numpy.sqrt(numpy.dot(sizes, sizes))

                for i, ln_idx in enumerate(ia.lines()):
                    if i>200:
                        print('WARNING: K3D nr of line segments is limited to 200.')
                        break
                    pts = ia.points()[ln_idx]
                    kobj = k3d.line(pts.astype(numpy.float32),
                                    color=_rgb2int(iap.GetColor()),
                                    opacity=iap.GetOpacity(),
                                    shader=settings.k3dLineShader,
                                    # width=iap.GetLineWidth()*sqsize/1000,
                                    name=name,
                                    )

                    vedo.notebook_plotter += kobj


    ####################################################################################
    elif vedo.notebookBackend == 'panel' and hasattr(plt, 'window') and plt.window:

        import panel # https://panel.pyviz.org/reference/panes/VTK.html
        plt.renderer.ResetCamera()
        vedo.notebook_plotter = panel.pane.VTK(plt.window,
                                               width=int(plt.size[0]/1.5),
                                               height=int(plt.size[1]/2),
        )

    ####################################################################################
    elif 'ipyvtk' in vedo.notebookBackend and hasattr(plt, 'window') and plt.window:

        from ipyvtklink.viewer import ViewInteractiveWidget
        plt.renderer.ResetCamera()
        vedo.notebook_plotter = ViewInteractiveWidget(
            plt.window, allow_wheel=True, quality=100, quick_quality=50,
        )

    ####################################################################################
    elif 'ipygany' in vedo.notebookBackend:

        from ipygany import PolyMesh, Scene, IsoColor, RGB, Component
        from ipygany import Alpha, ColorBar, colormaps, PointCloud
        from ipywidgets import FloatRangeSlider, Dropdown, VBox, AppLayout, jslink

        bgcol = colors.rgb2hex(colors.getColor(plt.backgrcol))

        actors2show2 = []
        for ia in actors2show:
            if not ia:
                continue
            if isinstance(ia, vedo.Assembly): #unpack assemblies
                assacts = ia.unpack()
                for ja in assacts:
                    if isinstance(ja, vedo.Assembly):
                        actors2show2 += ja.unpack()
                    else:
                        actors2show2.append(ja)
            else:
                actors2show2.append(ia)

        pmeshes = []
        colorbar = None
        for obj in actors2show2:
#            print("ipygany processing:", [obj], obj.name)

            if isinstance(obj, vedo.shapes.Line):
                lg = obj.diagonalSize()/1000 * obj.GetProperty().GetLineWidth()
                vmesh = vedo.shapes.Tube(obj.points(), r=lg, res=4).triangulate()
                vmesh.c(obj.c())
                faces = vmesh.faces()
                # todo: Lines
            elif isinstance(obj, Mesh):
                vmesh = obj.triangulate()
                faces = vmesh.faces()
            elif isinstance(obj, Points):
                vmesh = obj
                faces = []
            elif isinstance(obj, Volume):
                vmesh = obj.isosurface()
                faces = vmesh.faces()
            elif isinstance(obj, vedo.TetMesh):
                vmesh = obj.tomesh(fill=False)
                faces = vmesh.faces()
            else:
                print("ipygany backend: cannot process object type", [obj])
                continue

            vertices = vmesh.points()
            scals = vmesh.inputdata().GetPointData().GetScalars()
            if scals and not colorbar: # there is an active array, only pick the first
                aname = scals.GetName()
                arr = vmesh.pointdata[aname]
                parr = Component(name=aname, array=arr)
                if len(faces):
                    pmesh = PolyMesh(vertices=vertices, triangle_indices=faces, data={aname: [parr]})
                else:
                    pmesh = PointCloud(vertices=vertices, data={aname: [parr]})
                rng = scals.GetRange()
                colored_pmesh = IsoColor(pmesh, input=aname, min=rng[0], max=rng[1])
                if obj.scalarbar:
                    colorbar = ColorBar(colored_pmesh)
                    colormap_slider_range = FloatRangeSlider(value=rng,
                                                             min=rng[0], max=rng[1],
                                                             step=(rng[1] - rng[0]) / 100.)
                    jslink((colored_pmesh, 'range'), (colormap_slider_range, 'value'))
                    colormap = Dropdown(
                        options=colormaps,
                        description='Colormap:'
                    )
                    jslink((colored_pmesh, 'colormap'), (colormap, 'index'))

            else:
                if len(faces):
                    pmesh = PolyMesh(vertices=vertices, triangle_indices=faces)
                else:
                    pmesh = PointCloud(vertices=vertices)
                if vmesh.alpha() < 1:
                    colored_pmesh = Alpha(RGB(pmesh, input=tuple(vmesh.color())), input=vmesh.alpha())
                else:
                    colored_pmesh = RGB(pmesh, input=tuple(vmesh.color()))

            pmeshes.append(colored_pmesh)

        if colorbar:
            scene = AppLayout(
                    left_sidebar=Scene(pmeshes, background_color=bgcol),
                    right_sidebar=VBox((colormap_slider_range, #not working
                                        colorbar,
                                        colormap)),
                    pane_widths=[2, 0, 1],
            )
        else:
            scene = Scene(pmeshes, background_color=bgcol)

        vedo.notebook_plotter = scene


    ####################################################################################
    elif '2d' in vedo.notebookBackend.lower() and hasattr(plt, 'window') and plt.window:
        import PIL.Image
        try:
            import IPython
        except ImportError:
            raise Exception('IPython not available.')

        from vedo.io import screenshot
        settings.screeshotLargeImage = True
        nn = screenshot(asarray=True, scale=settings.screeshotScale)
        pil_img = PIL.Image.fromarray(nn)
        vedo.notebook_plotter = IPython.display.display(pil_img)

    return vedo.notebook_plotter


def _rgb2int(rgb_tuple):
    #Return the int number of a color from (r,g,b), with 0<r<1 etc.
    rgb = (int(rgb_tuple[0] * 255), int(rgb_tuple[1] * 255), int(rgb_tuple[2] * 255))
    return 65536 * rgb[0] + 256 * rgb[1] + rgb[2]
