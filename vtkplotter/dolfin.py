# FEniCS/Dolfin support API
#
from __future__ import division, print_function

import vtk
from vtk.util.numpy_support import numpy_to_vtk

import numpy as np

import vtkplotter.utils as utils
from vtkplotter.utils import printHistogram, ProgressBar

import vtkplotter.docs as docs

from vtkplotter.colors import printc

import vtkplotter.settings as settings
from vtkplotter.settings import datadir

from vtkplotter.actors import Actor, isolines

import vtkplotter.vtkio as vtkio
from vtkplotter.vtkio import load, screenshot, Video, exportWindow

import vtkplotter.shapes as shapes
from vtkplotter.shapes import Text, Latex

from vtkplotter.plotter import show, clear, Plotter, plotMatrix, closeWindow, interactive

__doc__ = (
    """
`FEniCS/Dolfin <https://fenicsproject.org>`_ support submodule.

Install with commands (e.g. in Anaconda3):

    .. code-block:: bash

        conda install -c conda-forge fenics
        pip install vtkplotter

Basic example:

    .. code-block:: python

        import dolfin
        from vtkplotter.dolfin import datadir, plot

        mesh = dolfin.Mesh(datadir+"dolfin_fine.xml")

        plot(mesh)

    |dolfinmesh|

Find many more examples in
`vtkplotter/examples/dolfin <https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin>`_


Image Gallery
=============

+-------------------------------------------------+-------------------------------------------------+
|                                                 | *(click on the figure to get to the script)*    |
+-------------------------------------------------+-------------------------------------------------+
| |ex03_poisson|                                  |   |ex02_tetralize-mesh|                         |
+-------------------------------------------------+-------------------------------------------------+
| Poisson equation with Dirichlet conditions      | Generate a tet-mesh from a polygonal surface    |
+-------------------------------------------------+-------------------------------------------------+
| |demo_submesh|                                  |   |pi_estimate|                                 |
+-------------------------------------------------+-------------------------------------------------+
| Extract submesh boundaries                      | Get pi from the integral of a circle            |
+-------------------------------------------------+-------------------------------------------------+
| |ex06_elasticity1|                              |   |ex06_elasticity2|                            |
+-------------------------------------------------+-------------------------------------------------+
| Solve a hyperelasticity problem...              | ...with different types of visulizations.       |
+-------------------------------------------------+-------------------------------------------------+
| |ft04_heat_gaussian|                            |   |demo_cahn-hilliard|                          |
+-------------------------------------------------+-------------------------------------------------+
| Diffusion of a Gaussian hill                    | Solve the Cahn-Hilliard equation                |
+-------------------------------------------------+-------------------------------------------------+
| |navier-stokes_lshape|                          |   |stokes1|                                     |
+-------------------------------------------------+-------------------------------------------------+
| The Navier-Stokes equations on L-shaped domain  | Stokes equations with Taylor-Hood elements      |
+-------------------------------------------------+-------------------------------------------------+
| |elastodynamics|                                |   |ft02_poisson_membrane|                       |
+-------------------------------------------------+-------------------------------------------------+
| Time-integration of the elastodynamics equation | Deflection of a membrane under a point load     |
+-------------------------------------------------+-------------------------------------------------+
| |magnetostatics|                                |   |turing_pattern|                              |
+-------------------------------------------------+-------------------------------------------------+
| Magnetic field of a solenoid                    | Patterns of Turing type reaction-diffusion      |
+-------------------------------------------------+-------------------------------------------------+
| |scalemesh|                                     |   |heatconv|                                    |
+-------------------------------------------------+-------------------------------------------------+
| Scale and elevate a mesh along one coordinate   | Heat equation in a moving media                 |
+-------------------------------------------------+-------------------------------------------------+
| |elasticbeam|                                   |   |wavy_1d|                                     |
+-------------------------------------------------+-------------------------------------------------+
| A soft beam deforming under its own weight      | The 1D wave eq. with the Crank Nicolson method  |
+-------------------------------------------------+-------------------------------------------------+
| |customAxes|                                    | |awefem|                                        |
+-------------------------------------------------+-------------------------------------------------+
| Customizing axes style and appearence           |The wave equation in arbitrary nr. of dimensions |
+-------------------------------------------------+-------------------------------------------------+
"""
    + docs._defs
)

__all__ = [
    "plot",
    "MeshActor",
    "MeshPoints",
    "MeshLines",
    "MeshArrows",
    "load",
    "show",
    "clear",
    "printc",
    "printHistogram",
    "Plotter",
    "ProgressBar",
    "Text",
    "Latex",
    "datadir",
    "screenshot",
    "Video",
    "plotMatrix",
    "isolines",
    "exportWindow",
    "closeWindow",
    "interactive",
]


def _inputsort(obj):
    u = None
    mesh = None
    if not utils.isSequence(obj):
        obj = [obj]

    for ob in obj:
        inputtype = str(type(ob))
        #printc('inputtype is', inputtype, c=2)

        if "vtk" in inputtype: # skip vtk objects, will be added later
            continue

        if "dolfin" in inputtype:
            if "MeshFunction" in inputtype:
                mesh = ob.mesh()

                import dolfin
                V = dolfin.FunctionSpace(mesh, "CG", 1)
                u = dolfin.Function(V)
                #print(mesh.cells())
                #print(len(mesh.cells()), len(mesh.coordinates()), len(ob.array()))
                #print(mesh.num_cells())
                #print(u.vector()[:])

                v2d = dolfin.vertex_to_dof_map(V)
                u.vector()[v2d] = ob.array()

#                r = ob.dim()
#                    if r == 0:
#                        V = dolfin.FunctionSpace(mesh, "CG", 1)
#                    elif r == 1:
#                        V = dolfin.VectorFunctionSpace(mesh, "CG", 1, dim=r)
#                    else:
#                        V = dolfin.TensorFunctionSpace(mesh, "CG", 1, shape=(r,r))
#                except:
#                    printc('~times Sorry could not deal with your MeshFunction', c=1)
#                    return None
#                tdim = mesh.topology().dim()
#                d = ob.dim()
#                if tdim == 2 and d == 2:
#                    import matplotlib.tri as tri
#                    xy = mesh.coordinates()
#                    mh = buildPolyData(xy, mesh.cells())
#                    show(mh)
#                    print( tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells()) )
#                    exit()

            elif "Function" in inputtype or "Expression" in inputtype:
                u = ob
            elif "Mesh" in inputtype:
                mesh = ob

        if "str" in inputtype:
            import dolfin
            mesh = dolfin.Mesh(ob)

    if u and not mesh and hasattr(u, "function_space"):
        V = u.function_space()
        if V:
            mesh = V.mesh()
    if u and not mesh and hasattr(u, "mesh"):
        mesh = u.mesh()

    if not mesh:
        printc("~times Error: dolfin mesh is not defined.", c=1)
        raise RuntimeError()

    #printc('------------------------------------')
    #printc('mesh.topology dim=', mesh.topology().dim())
    #printc('mesh.geometry dim=', mesh.geometry().dim())
    #if u: printc('u.value_rank()', u.value_rank())
    return (mesh, u)



def plot(*inputobj, **options):
    """
    Plot the object(s) provided.

    Input can be: ``vtkActor``, ``vtkVolume``, ``dolfin.Mesh``, ``dolfin.MeshFunction*``,
    ``dolfin.Expression`` or ``dolfin.Function``.

    :return: the current ``Plotter`` class instance.

    :param str mode: one or more of the following can be combined in any order

        - `mesh`/`color`, will plot the mesh, by default colored with a scalar if available

            - `warp`, mesh will be modified by a displacement function
            - `contour`, to be implemented
        - `arrows`, mesh displacements are plotted as scaled arrows.
        - `lines`, mesh displacements are plotted as scaled lines.
        - `tensors`, to be implemented

    :param bool add: add the input objects without clearing the already plotted ones
    :param float density: show only a subset of lines or arrows [0-1]
    :param bool wire[frame]: visualize mesh as wireframe [False]
    :param c[olor]: set mesh color [None]
    :param float alpha: set object's transparency [1]
    :param float lw: line width of the mesh (set to zero to hide mesh) [0.5]
    :param float ps: set point size of mesh vertices [None]
    :param float z: add a constant to z-coordinate (useful to show 2D slices as function of time)
    :param str legend: add a legend to the top-right of window [None]
    :param bool scalarbar: add a scalarbar to the window ['vertical']
    :param float vmin: set the minimum for the range of the scalar [None]
    :param float vmax: set the maximum for the range of the scalar [None]
    :param float scale: add a scaling factor to arrows and lines sizes [1]
    :param str cmap: choose a color map for scalars
    :param int bands: group colors in `n` bands
    :param str shading: mesh shading ['flat', 'phong', 'gouraud']
    :param str text: add a gray text comment to the top-left of the window [None]

    :param dict isolines: dictionary of isolines properties

        - n, (int) - add this number of isolines to the mesh
        - c, - isoline color
        - lw, (float) - isoline width
        - z, (float) - add to the isoline z coordinate to make them more visible

    :param float warpZfactor: elevate z-axis by scalar value (useful for 2D geometries)
    :param float warpYfactor: elevate z-axis by scalar value (useful for 1D geometries)

    :param bool newPlotter: spawn a new instance of Plotter class, pops up a new window
    :param int at: renderer number to plot to
    :param list shape: subdvide window in (n,m) rows and columns
    :param int N: automatically subdvide window in N renderers
    :param list pos: (x,y) coordinates of the window position on screen
    :param size: window size (x,y)

    :param str title: window title
    :param bg: background color name of window
    :param bg2: second background color name to create a color gradient
    :param int style: choose a predefined style [0-4]

      - 0, `vtkplotter`, style (blackboard background, rainbow color map)
      - 1, `matplotlib`, style (white background, viridis color map)
      - 2, `paraview`, style
      - 3, `meshlab`, style
      - 4, `bw`, black and white style.

    :param int axes: axes type number

      - 0,  no axes,
      - 1,  draw three customizable gray grid walls. See ``show()`` documentation.
      - 2,  show cartesian axes from (0,0,0)
      - 3,  show positive range of cartesian axes from (0,0,0)
      - 4,  show a triad at bottom left
      - 5,  show a cube at bottom left
      - 6,  mark the corners of the bounding box
      - 7,  draw a simple ruler at the bottom of the window
      - 8,  show the `vtkCubeAxesActor` object,
      - 9,  show the bounding box outLine,
      - 10, show three circles representing the maximum bounding box.

    :param bool infinity: if True fugue point is set at infinity (no perspective effects)
    :param bool sharecam: if False each renderer will have an independent vtkCamera
    :param bool interactive: if True will stop after show() to allow interaction w/ window
    :param bool depthpeeling: depth-peel volumes along with the translucent geometry
    :param bool offscreen: if True will not show the rendering window

    :param float zoom: camera zooming factor
    :param viewup: camera view-up direction ['x','y','z', or a vector direction]
    :param float azimuth: add azimuth rotation of the scene, in degrees
    :param float elevation: add elevation rotation of the scene, in degrees
    :param float roll: add roll-type rotation of the scene, in degrees

    :param dict camera: Camera parameters can further be specified with a dictionary assigned to the ``camera`` keyword:
        (E.g. `show(camera={'pos':(1,2,3), 'thickness':1000,})`)

        - pos, `(list)`,  the position of the camera in world coordinates
        - focalPoint `(list)`, the focal point of the camera in world coordinates
        - viewup `(list)`, the view up direction for the camera
        - distance `(float)`, set the focal point to the specified distance from the camera position.
        - clippingRange `(float)`, distance of the near and far clipping planes along the direction of projection.
        - parallelScale `(float)`,
            scaling used for a parallel projection, i.e. the height of the viewport
            in world-coordinate distances. The default is 1. Note that the "scale" parameter works as
            an "inverse scale", larger numbers produce smaller images.
            This method has no effect in perspective projection mode.
        - thickness `(float)`,
            set the distance between clipping planes. This method adjusts the far clipping
            plane to be set a distance 'thickness' beyond the near clipping plane.
        - viewAngle `(float)`,
            the camera view angle, which is the angular height of the camera view
            measured in degrees. The default angle is 30 degrees.
            This method has no effect in parallel projection mode.
            The formula for setting the angle up for perfect perspective viewing is:
            angle = 2*atan((h/2)/d) where h is the height of the RenderWindow
            (measured by holding a ruler up to your screen) and d is the distance from your eyes to the screen.

    :param int interactorStyle: change the style of muose interaction of the scene
    :param bool q: exit python session after returning.
    """

    if len(inputobj) == 0:
        return interactive()

    mesh, u = _inputsort(inputobj)

    mode = options.pop("mode", 'mesh')
    ttime = options.pop("z", None)

    add = options.pop("add", False)

    wire = options.pop("wire", False)
    wireframe = options.pop("wireframe", None)
    if wireframe is not None:
        wire = wireframe

    c = options.pop("c", None)
    color = options.pop("color", None)
    if color is not None:
        c = color

    lc = options.pop("lc", None)

    alpha = options.pop("alpha", 1)
    lw = options.pop("lw", 0.5)
    ps = options.pop("ps", None)
    legend = options.pop("legend", None)
    scbar = options.pop("scalarbar", 'v')
    vmin = options.pop("vmin", None)
    vmax = options.pop("vmax", None)
    cmap = options.pop("cmap", None)
    bands = options.pop("bands", None)
    scale = options.pop("scale", 1)
    shading = options.pop("shading", None)
    text = options.pop("text", None)
    style = options.pop("style", 'vtk')
    isolns = options.pop("isolines", dict())
    warpZfactor = options.pop("warpZfactor", None)
    warpYfactor = options.pop("warpYfactor", None)
    lighting = options.pop("lighting", None)

    # refresh axes titles for axes type = 8 (vtkCubeAxesActor)
    settings.xtitle = options.pop("xtitle", 'x')
    settings.ytitle = options.pop("ytitle", 'y')
    settings.ztitle = options.pop("ztitle", 'z')
    if settings.plotter_instance:
        if settings.ytitle!='x':
            if 'at' in options.keys():
                at = options['at']
            else:
                at = 0
            aet = settings.plotter_instance.axes_instances
            if len(aet)>at and isinstance(aet[at], vtk.vtkCubeAxesActor):
                aet[at].SetXTitle(settings.xtitle)
        if settings.ytitle!='y':
            if 'at' in options.keys():
                at = options['at']
            else:
                at = 0
            aet = settings.plotter_instance.axes_instances
            if len(aet)>at and isinstance(aet[at], vtk.vtkCubeAxesActor):
                aet[at].SetYTitle(settings.ytitle)
        if settings.ytitle!='z':
            if 'at' in options.keys():
                at = options['at']
            else:
                at = 0
            aet = settings.plotter_instance.axes_instances
            if len(aet)>at and isinstance(aet[at], vtk.vtkCubeAxesActor):
                aet[at].SetZTitle(settings.ztitle)


    # change some default to emulate standard behaviours
    options['verbose'] = False # dont disturb
    if  style == 0 or style == 'vtk':
        font = 'courier'
        axes = options.pop('axes', None)
        if axes is None:
            options['axes'] = {
                    'xyGrid':False,
                    'yzGrid':False,
                    'zxGrid':False,
                   }
        else:
            options['axes'] = axes # put back
        if cmap is None:
            cmap = 'rainbow'
    elif style == 1 or style == 'matplotlib':
        font = 'courier'
        bg = options.pop('bg', None)
        if bg is None:
            options['bg'] = 'white'
        else:
            options['bg'] = bg
        axes = options.pop('axes', None)
        if axes is None:
            options['axes'] =  {
                    'xyGrid':False,
                    'yzGrid':False,
                    'zxGrid':False,
                   }
        else:
            options['axes'] = axes # put back
        if cmap is None:
            cmap = 'viridis'
    elif style == 2 or style == 'paraview':
        font = 'arial'
        bg = options.pop('bg', None)
        if bg is None:
            options['bg'] = (82, 87, 110)
        else:
            options['bg'] = bg
        if cmap is None:
            cmap = 'coolwarm'
    elif style == 3 or style == 'meshlab':
        font = 'courier'
        bg = options.pop('bg', None)
        if bg is None:
            options['bg'] = (8, 8, 16)
            options['bg2'] = (117, 117, 234)
        else:
            options['bg'] = bg
        axes = options.pop('axes', None)
        if axes is None:
            options['axes'] = 10
        else:
            options['axes'] = axes # put back
        if cmap is None:
            cmap = 'afmhot'
    elif style == 4 or style == 'bw':
        font = 'courier'
        bg = options.pop('bg', None)
        if bg is None:
            options['bg'] = (217, 255, 238)
        else:
            options['bg'] = bg
        axes = options.pop('axes', None)
        if axes is None:
            options['axes'] =  {
                    'xyGrid':False,
                    'yzGrid':False,
                    'zxGrid':False,
                   }
        else:
            options['axes'] = axes # put back
        if cmap is None:
            cmap = 'binary'


    #################################################################
    actors = []
    if add and settings.plotter_instance:
        actors = settings.plotter_instance.actors

    if 'mesh' in mode or 'color' in mode or 'warp' in mode or 'displac' in mode:
        if 'warp' in mode: #deprecation
            printc("~bomb Please use 'displacement' instead of 'warp' in mode!", c=1)

        actor = MeshActor(u, mesh, wire=wire)
        if lighting:
            actor.lighting(lighting)
        if ttime:
            actor.z(ttime)
        if legend:
            actor.legend(legend)
        if c:
            actor.color(c)
        if lc:
            actor.lineColor(lc)
        if alpha:
            alpha = min(alpha, 1)
            actor.alpha(alpha*alpha)
        if lw:
            actor.lineWidth(lw)
            if wire and alpha:
                lw1 = min(lw, 1)
                actor.alpha(alpha*lw1)
        if ps:
            actor.pointSize(ps)
        if shading:
            if shading == 'phong':
                actor.phong()
            elif shading == 'flat':
                actor.flat()
            elif shading[0] == 'g':
                actor.gouraud()
        delta = None
        if cmap and u and c is None:
            delta = [u(p) for p in mesh.coordinates()]
            #delta = u.compute_vertex_values(mesh) # needs reshape
            if u.value_rank() > 0: # wiil show the size of the vector
                actor.pointColors(utils.mag(delta),
                                  cmap=cmap, bands=bands, vmin=vmin, vmax=vmax)
            else:
                actor.pointColors(delta, cmap=cmap, bands=bands, vmin=vmin, vmax=vmax)
        if scbar and c is None:
            if 'h' in scbar:
                actor.addScalarBar(horizontal=True, vmin=vmin, vmax=vmax)
            else:
                actor.addScalarBar(horizontal=False, vmin=vmin, vmax=vmax)

        if 'warp' in mode or 'displac' in mode:
            if delta is None:
                delta = [u(p) for p in mesh.coordinates()]
            movedpts = mesh.coordinates() + delta
            actor.polydata(False).GetPoints().SetData(numpy_to_vtk(movedpts))
            actor.poly.GetPoints().Modified()
            actor.u_values = delta

        if warpZfactor:
            scals = actor.scalars(0)
            if len(scals):
                pts_act = actor.getPoints(copy=False)
                pts_act[:, 2] = scals*warpZfactor
        if warpYfactor:
            scals = actor.scalars(0)
            if len(scals):
                pts_act = actor.getPoints(copy=False)
                pts_act[:, 1] = scals*warpYfactor

        if len(isolns) > 0:
            ison = isolns.pop("n", 10)
            isocol = isolns.pop("c", 'black')
            isoalpha = isolns.pop("alpha", 1)
            isolw = isolns.pop("lw", 1)

            isos = isolines(actor, n=ison).color(isocol).lw(isolw).alpha(isoalpha)

            isoz = isolns.pop("z", None)
            if isoz is not None: # kind of hack to make isolines visible on flat meshes
                d = isoz
            else:
                d = actor.diagonalSize()/400
            isos.z(actor.z()+d)
            actors.append(isos)

        actors.append(actor)


    #################################################################
    if 'arrow' in mode or 'line' in mode:
        if 'arrow' in mode:
            arrs = MeshArrows(u, scale=scale)
        else:
            arrs = MeshLines(u, scale=scale)
        if legend and not 'mesh' in mode:
            arrs.legend(legend)
        if c:
            arrs.color(c)
            arrs.color(c)
        if alpha:
            arrs.alpha(alpha)
        actors.append(arrs)


    #################################################################
    if 'tensor' in mode:
        pass #todo


    #################################################################
    for ob in inputobj:
        inputtype = str(type(ob))
        if 'vtk' in inputtype:
           actors.append(ob)

    if text:
        textact = Text(text, font=font)
        actors.append(textact)

    if 'at' in options.keys() and 'interactive' not in options.keys():
        if settings.plotter_instance:
            N = settings.plotter_instance.shape[0]*settings.plotter_instance.shape[1]
            if options['at'] == N-1:
                options['interactive'] = True

    if settings.plotter_instance:
        for a2 in settings.collectable_actors:
            if isinstance(a2, vtk.vtkCornerAnnotation):
                if 0 in a2.renderedAt: # remove old message
                    settings.plotter_instance.remove(a2)
                    break

    return show(actors, **options)


###################################################################################
class MeshActor(Actor):
    """MeshActor, a vtkActor derived object for dolfin support."""

    def __init__(
        self, *inputobj, **options # c="gold", alpha=1, wire=True, bc=None, computeNormals=False
    ):

        c = options.pop("c", "gold")
        alpha = options.pop("alpha", 1)
        wire = options.pop("wire", True)
        bc = options.pop("bc", None)
        computeNormals = options.pop("computeNormals", False)

        mesh, u = _inputsort(inputobj)

        poly = vtkio.buildPolyData(mesh)

        Actor.__init__(
            self,
            poly,
            c=c,
            alpha=alpha,
            wire=wire,
            bc=bc,
            computeNormals=computeNormals,
        )

        self.mesh = mesh  # holds a dolfin Mesh obj
        self.u = u  # holds a dolfin function_data
        self.u_values = None  # holds the actual values of u on the mesh
        u_values = None

        if u:
            u_values = np.array([u(p) for p in self.mesh.coordinates()])
            #print(u_values)

        if u_values is not None:  # colorize if a dolfin function is passed
            if len(u_values.shape) == 2:
                if u_values.shape[1] in [2, 3]:  # u_values is 2D or 3D
                    self.u_values = u_values
                    dispsizes = utils.mag(u_values)
            else:  # u_values is 1D
                dispsizes = u_values

            self.addPointScalars(dispsizes, "u_values")#.mapPointsToCells()

def MeshPoints(*inputobj, **options):
    """
    Build a point ``Actor`` for a list of points.

    :param float r: point radius.
    :param c: color name, number, or list of [R,G,B] colors of same length as plist.
    :type c: int, str, list
    :param float alpha: transparency in range [0,1].
    """
    r = options.pop("r", 5)
    c = options.pop("c", "gray")
    alpha = options.pop("alpha", 1)

    mesh, u = _inputsort(inputobj)
    plist = mesh.coordinates()
    if u:
        u_values = np.array([u(p) for p in plist])
    if len(plist[0]) == 2:  # coords are 2d.. not good..
        plist = np.insert(plist, 2, 0, axis=1)  # make it 3d
    if len(plist[0]) == 1:  # coords are 1d.. not good..
        plist = np.insert(plist, 1, 0, axis=1)  # make it 3d
        plist = np.insert(plist, 2, 0, axis=1)

    actor = shapes.Points(plist, r=r, c=c, alpha=alpha)

    actor.mesh = mesh
    if u:
        actor.u = u
        if len(u_values.shape) == 2:
            if u_values.shape[1] in [2, 3]:  # u_values is 2D or 3D
                actor.u_values = u_values
                dispsizes = utils.mag(u_values)
        else:  # u_values is 1D
            dispsizes = u_values
        actor.addPointScalars(dispsizes, "u_values")
    return actor


def MeshLines(*inputobj, **options):
    """
    Build the line segments between two lists of points `startPoints` and `endPoints`.
    `startPoints` can be also passed in the form ``[[point1, point2], ...]``.

    A dolfin ``Mesh`` that was deformed/modified by a function can be
    passed together as inputs.

    :param float scale: apply a rescaling factor to the length
    """
    scale = options.pop("scale", 1)
    lw = options.pop("lw", 1)
    c = options.pop("c", None)
    alpha = options.pop("alpha", 1)

    mesh, u = _inputsort(inputobj)
    startPoints = mesh.coordinates()
    u_values = np.array([u(p) for p in mesh.coordinates()])
    if not utils.isSequence(u_values[0]):
        printc("~times Error: cannot show Lines for 1D scalar values!", c=1)
        raise RuntimeError()
    endPoints = mesh.coordinates() + u_values
    if u_values.shape[1] == 2:  # u_values is 2D
        u_values = np.insert(u_values, 2, 0, axis=1)  # make it 3d
        startPoints = np.insert(startPoints, 2, 0, axis=1)  # make it 3d
        endPoints = np.insert(endPoints, 2, 0, axis=1)  # make it 3d

    actor = shapes.Lines(
        startPoints, endPoints, scale=scale, lw=lw, c=c, alpha=alpha
    )

    actor.mesh = mesh
    actor.u = u
    actor.u_values = u_values
    return actor


def MeshArrows(*inputobj, **options):
    """
    Build arrows representing displacements.

    :param float s: cross-section size of the arrow
    :param float rescale: apply a rescaling factor to the length
    """
    s = options.pop("s", None)
    scale = options.pop("scale", 1)
    c = options.pop("c", "gray")
    alpha = options.pop("alpha", 1)
    res = options.pop("res", 12)

    mesh, u = _inputsort(inputobj)
    startPoints = mesh.coordinates()
    u_values = np.array([u(p) for p in mesh.coordinates()])
    if not utils.isSequence(u_values[0]):
        printc("~times Error: cannot show Arrows for 1D scalar values!", c=1)
        raise RuntimeError()
    endPoints = mesh.coordinates() + u_values
    if u_values.shape[1] == 2:  # u_values is 2D
        u_values = np.insert(u_values, 2, 0, axis=1)  # make it 3d
        startPoints = np.insert(startPoints, 2, 0, axis=1)  # make it 3d
        endPoints = np.insert(endPoints, 2, 0, axis=1)  # make it 3d

    actor = shapes.Arrows(
        startPoints, endPoints, s=s, scale=scale, c=c, alpha=alpha, res=res
    )
    actor.mesh = mesh
    actor.u = u
    actor.u_values = u_values
    return actor



#def make_mapping(sub_space, super_space):
#    from scipy.spatial import cKDTree
#    super_dof_coor = super_space.tabulate_dof_coordinates()
#    sub_dof_coor = sub_space.tabulate_dof_coordinates()
#
#    tree = cKDTree(super_dof_coor)
#    _,mapping = tree.query(sub_dof_coor, k=1)
#    return mapping




#from dolfin import *
#mesh = UnitCubeMesh(10, 10, 10)        # this will be the "grandparent" mesh
#mesh.coordinates()[:,0] -= .5          # shift x-coords
#mesh.coordinates()[:,1] -= .5          # shift y-coords
#
#bmesh  = BoundaryMesh(mesh, "exterior")   # surface boundary mesh
#
## mark the cells on the bottom of the bmesh surface by iterating over bmesh cells.
## Look up ccorresponding facet in mesh and mark if facet normal points in -z direction
#cellmap = bmesh.entity_map(2)
#vertmap = bmesh.entity_map(0)
#pb = MeshFunction("size_t", bmesh, dim =1)
#for c in cells(bmesh):
#  if Facet(mesh, cellmap[c.index()]).normal().z() < 0:
#    pb[c] = 1
#
## use the marked bottom of the bmesh to create a Submesh
#submesh = SubMesh(bmesh, pb, 1)           # bottom of boundary mesh
#
## FunctionSpaces on main mesh, bmesh, submesh
#V   = FunctionSpace(mesh, "CG", 1)        # mesh function space
#Vb  = FunctionSpace(bmesh,   "CG", 1)     # surface function space
#Vs  = FunctionSpace(submesh, "CG", 1)     # submesh function space
#
## mappings we may need:
#m    = vertex_to_dof_map(V)
#b    = vertex_to_dof_map(Vb)
#s    = vertex_to_dof_map(Vs)
#
#mi   = dof_to_vertex_map(V)
#bi   = dof_to_vertex_map(Vb)
#si   = dof_to_vertex_map(Vs)
#
#t = submesh.data().array('parent_vertex_indices', 0) # mapping from submesh back to bmesh
#
## Functions on main mesh, bmesh, and submesh
#u   = Function(V)
#ub  = Function(Vb)                        # boundary function
#us  = Function(Vs)                        # surface function
#
## Interpolate the following expr onto u, ub, us
#expr  = Expression('sqrt(pow(x[0],2) + pow(x[1], 2))', degree=2)
#u.interpolate(expr)
#ub.interpolate(expr)
#us.interpolate(expr)
#
## Some empty function to test the mappings
#ub_test = Function(Vb)  # empty bmesh function
#u_test  = Function(V)   # empty mesh function
#
## Mapping from submesh to bmesh (works)... Any way to avoid calls to get_local/set_local??
#us_a  = us.vector().get_local()              # origin array
#ub_test_a = ub_test.vector().get_local()     # destination array
#ub_test_a[b[t]]  = us_a[s]                   # transfer
#ub_test.vector().set_local(ub_test_a)        # set destination function values
#
## Mapping from submesh to grandparent mesh (less than satisfying solution to question in fenics forum link)
## Bonus points for getting this kind of map composition to work:
## u_test_a = u_test.vector().get_local()    # destination array
## u_test_a[m[b[t]]] = us_a[s]               # transfer ( not working )
## u_test.vector().set_local(u_test_a)
#for Vs_dof, val in enumerate(us.vector().get_local()):
#    submesh_vertex = si[Vs_dof]
#    boundary_vertex = t[submesh_vertex]
#    mesh_vertex = vertmap[int(boundary_vertex)] # np.uint not accepted
#    V_dof = m[mesh_vertex]
#    u_test.vector()[V_dof] = val
#
#u.rename('u','u')
#ub_test.rename('ub_test','ub_test')
#u_test.rename('u_test','u_test')
#us.rename('us','us')







#
#import dolfin
#import dolfin.cpp as cpp
#import ufl
#import numpy as np
#
#__all__ = ['plot']
#
#_meshfunction_types = (cpp.mesh.MeshFunctionBool,
#                       cpp.mesh.MeshFunctionInt,
#                       cpp.mesh.MeshFunctionDouble,
#                       cpp.mesh.MeshFunctionSizet)
#_matplotlib_plottable_types = (cpp.function.Function,
#                               cpp.function.Expression, cpp.mesh.Mesh,
#                               cpp.fem.DirichletBC) + _meshfunction_types
#_x3dom_plottable_types = (cpp.function.Function, cpp.mesh.Mesh)
#_all_plottable_types = tuple(set.union(set(_matplotlib_plottable_types),
#                                       set(_x3dom_plottable_types)))
#
#
#def _has_matplotlib():
#    try:
#        import matplotlib  # noqa
#    except ImportError:
#        return False
#    return True
#
#
#def mesh2triang(mesh):
#    import matplotlib.tri as tri
#    xy = mesh.coordinates()
#    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())
#
#
#def mplot_mesh(ax, mesh, **kwargs):
#    tdim = mesh.topology().dim()
#    gdim = mesh.geometry().dim()
#    if gdim == 2 and tdim == 2:
#        color = kwargs.pop("color", '#808080')
#        return ax.triplot(mesh2triang(mesh), color=color, **kwargs)
#    elif gdim == 3 and tdim == 3:
#        bmesh = cpp.mesh.BoundaryMesh(mesh, "exterior", order=False)
#        mplot_mesh(ax, bmesh, **kwargs)
#    elif gdim == 3 and tdim == 2:
#        xy = mesh.coordinates()
#        return ax.plot_trisurf(*[xy[:, i] for i in range(gdim)],
#                               triangles=mesh.cells(), **kwargs)
#    elif tdim == 1:
#        x = [mesh.coordinates()[:, i] for i in range(gdim)]
#        if gdim == 1:
#            x.append(np.zeros_like(x[0]))
#            ax.set_yticks([])
#        marker = kwargs.pop('marker', 'o')
#        return ax.plot(*x, marker=marker, **kwargs)
#    else:
#        assert False, "this code should not be reached"
#
#
## TODO: This is duplicated somewhere else
#def create_cg1_function_space(mesh, sh):
#    r = len(sh)
#    if r == 0:
#        V = dolfin.FunctionSpace(mesh, "CG", 1)
#    elif r == 1:
#        V = dolfin.VectorFunctionSpace(mesh, "CG", 1, dim=sh[0])
#    else:
#        V = dolfin.TensorFunctionSpace(mesh, "CG", 1, shape=sh)
#    return V
#
#
#def mplot_expression(ax, f, mesh, **kwargs):
#    # TODO: Can probably avoid creating the function space here by
#    # restructuring mplot_function a bit so it can handle Expression
#    # natively
#    V = create_cg1_function_space(mesh, f.value_shape)
#    g = dolfin.interpolate(f, V)
#    return mplot_function(ax, g, **kwargs)
#
#
#def mplot_function(ax, f, **kwargs):
#    mesh = f.function_space().mesh()
#    gdim = mesh.geometry().dim()
#    tdim = mesh.topology().dim()
#
#    # Extract the function vector in a way that also works for
#    # subfunctions
#    try:
#        fvec = f.vector()
#    except RuntimeError:
#        fspace = f.function_space()
#        try:
#            fspace = fspace.collapse()
#        except RuntimeError:
#            return
#        fvec = dolfin.interpolate(f, fspace).vector()
#
#    if fvec.size() == mesh.num_cells():
#        # DG0 cellwise function
#        C = fvec.get_local()  # NB! Assuming here dof ordering matching cell numbering
#        if gdim == 2 and tdim == 2:
#            return ax.tripcolor(mesh2triang(mesh), C, **kwargs)
#        elif gdim == 3 and tdim == 2:  # surface in 3d
#            # FIXME: Not tested, probably broken
#            xy = mesh.coordinates()
#            shade = kwargs.pop("shade", True)
#            return ax.plot_trisurf(mesh2triang(mesh), xy[:, 2], C, shade=shade,
#                                   **kwargs)
#        elif gdim == 1 and tdim == 1:
#            x = mesh.coordinates()[:, 0]
#            nv = len(x)
#            # Insert duplicate points to get piecewise constant plot
#            xp = np.zeros(2 * nv - 2)
#            xp[0] = x[0]
#            xp[-1] = x[-1]
#            xp[1:2 * nv - 3:2] = x[1:-1]
#            xp[2:2 * nv - 2:2] = x[1:-1]
#            Cp = np.zeros(len(xp))
#            Cp[0:len(Cp) - 1:2] = C
#            Cp[1:len(Cp):2] = C
#            return ax.plot(xp, Cp, *kwargs)
#        # elif tdim == 1:  # FIXME: Plot embedded line
#        else:
#            raise AttributeError('Matplotlib plotting backend only supports 2D mesh for scalar functions.')
#
#    elif f.value_rank() == 0:
#        # Scalar function, interpolated to vertices
#        # TODO: Handle DG1?
#        C = f.compute_vertex_values(mesh)
#        if gdim == 2 and tdim == 2:
#            mode = kwargs.pop("mode", "contourf")
#            if mode == "contourf":
#                levels = kwargs.pop("levels", 40)
#                return ax.tricontourf(mesh2triang(mesh), C, levels, **kwargs)
#            elif mode == "color":
#                shading = kwargs.pop("shading", "gouraud")
#                return ax.tripcolor(mesh2triang(mesh), C, shading=shading,
#                                    **kwargs)
#            elif mode == "warp":
#                from matplotlib import cm
#                cmap = kwargs.pop("cmap", cm.jet)
#                linewidths = kwargs.pop("linewidths", 0)
#                return ax.plot_trisurf(mesh2triang(mesh), C, cmap=cmap,
#                                       linewidths=linewidths, **kwargs)
#            elif mode == "wireframe":
#                return ax.triplot(mesh2triang(mesh), **kwargs)
#            elif mode == "contour":
#                return ax.tricontour(mesh2triang(mesh), C, **kwargs)
#        elif gdim == 3 and tdim == 2:  # surface in 3d
#            # FIXME: Not tested
#            from matplotlib import cm
#            cmap = kwargs.pop("cmap", cm.jet)
#            return ax.plot_trisurf(mesh2triang(mesh), C, cmap=cmap, **kwargs)
#        elif gdim == 3 and tdim == 3:
#            # Volume
#            # TODO: Isosurfaces?
#            # Vertex point cloud
#            X = [mesh.coordinates()[:, i] for i in range(gdim)]
#            return ax.scatter(*X, c=C, **kwargs)
#        elif gdim == 1 and tdim == 1:
#            x = mesh.coordinates()[:, 0]
#            ax.set_aspect('auto')
#
#            p = ax.plot(x, C, **kwargs)
#
#            # Setting limits for Line2D objects
#            # Must be done after generating plot to avoid ignoring function
#            # range if no vmin/vmax are supplied
#            vmin = kwargs.pop("vmin", None)
#            vmax = kwargs.pop("vmax", None)
#            ax.set_ylim([vmin, vmax])
#
#            return p
#        # elif tdim == 1: # FIXME: Plot embedded line
#        else:
#            raise AttributeError('Matplotlib plotting backend only supports 2D mesh for scalar functions.')
#
#    elif f.value_rank() == 1:
#        # Vector function, interpolated to vertices
#        w0 = f.compute_vertex_values(mesh)
#        nv = mesh.num_vertices()
#        if len(w0) != gdim * nv:
#            raise AttributeError('Vector length must match geometric dimension.')
#        X = mesh.coordinates()
#        X = [X[:, i] for i in range(gdim)]
#        U = [w0[i * nv: (i + 1) * nv] for i in range(gdim)]
#
#        # Compute magnitude
#        C = U[0]**2
#        for i in range(1, gdim):
#            C += U[i]**2
#        C = np.sqrt(C)
#
#        mode = kwargs.pop("mode", "glyphs")
#        if mode == "glyphs":
#            args = X + U + [C]
#            if gdim == 3:
#                length = kwargs.pop("length", 0.1)
#                return ax.quiver(*args, length=length, **kwargs)
#            else:
#                return ax.quiver(*args, **kwargs)
#        elif mode == "displacement":
#            Xdef = [X[i] + U[i] for i in range(gdim)]
#            import matplotlib.tri as tri
#            if gdim == 2 and tdim == 2:
#                # FIXME: Not tested
#                triang = tri.Triangulation(Xdef[0], Xdef[1], mesh.cells())
#                shading = kwargs.pop("shading", "flat")
#                return ax.tripcolor(triang, C, shading=shading, **kwargs)
#            else:
#                # Return gracefully to make regression test pass without vtk
#                cpp.warning('Matplotlib plotting backend does not support '
#                            'displacement for %d in %d. Continuing without '
#                            'plotting...' % (tdim, gdim))
#                return
#
#
#def mplot_meshfunction(ax, obj, **kwargs):
#    mesh = obj.mesh()
#    tdim = mesh.topology().dim()
#    d = obj.dim()
#    if tdim == 2 and d == 2:
#        C = obj.array()
#        triang = mesh2triang(mesh)
#        assert not kwargs.pop("facecolors", None), "Not expecting 'facecolors' in kwargs"
#        return ax.tripcolor(triang, facecolors=C, **kwargs)
#    else:
#        # Return gracefully to make regression test pass without vtk
#        cpp.warning('Matplotlib plotting backend does not support mesh '
#                    'function of dim %d. Continuing without plotting...' % d)
#        return
#
#
#def mplot_dirichletbc(ax, obj, **kwargs):
#    raise AttributeError("Matplotlib plotting backend doesn't handle DirichletBC.")
#
#
#def _plot_matplotlib(obj, mesh, kwargs):
#    if not isinstance(obj, _matplotlib_plottable_types):
#        print("Don't know how to plot type %s." % type(obj))
#        return
#
#    # Plotting is not working with all ufl cells
#    if mesh.ufl_cell().cellname() not in ['interval', 'triangle', 'tetrahedron']:
#        raise AttributeError(("Matplotlib plotting backend doesn't handle %s mesh.\n"
#                              "Possible options are saving the output to XDMF file "
#                              "or using 'x3dom' backend.") % mesh.ufl_cell().cellname())
#
#    # Avoid importing pyplot until used
#    try:
#        import matplotlib.pyplot as plt
#    except Exception:
#        cpp.warning("matplotlib.pyplot not available, cannot plot.")
#        return
#
#    gdim = mesh.geometry().dim()
#    if gdim == 3 or kwargs.get("mode") in ("warp",):
#        # Importing this toolkit has side effects enabling 3d support
#        from mpl_toolkits.mplot3d import axes3d  # noqa
#        # Enabling the 3d toolbox requires some additional arguments
#        ax = plt.gca(projection='3d')
#    else:
#        ax = plt.gca()
#    ax.set_aspect('equal')
#
#    title = kwargs.pop("title", None)
#    if title is not None:
#        ax.set_title(title)
#
#    # Translate range_min/max kwargs supported by VTKPlotter
#    vmin = kwargs.pop("range_min", None)
#    vmax = kwargs.pop("range_max", None)
#    if vmin and "vmin" not in kwargs:
#        kwargs["vmin"] = vmin
#    if vmax and "vmax" not in kwargs:
#        kwargs["vmax"] = vmax
#
#    # Drop unsupported kwargs and inform user
#    _unsupported_kwargs = ["rescale", "wireframe"]
#    for kw in _unsupported_kwargs:
#        if kwargs.pop(kw, None):
#            cpp.warning("Matplotlib backend does not support '%s' kwarg yet. "
#                        "Ignoring it..." % kw)
#
#    if isinstance(obj, cpp.function.Function):
#        return mplot_function(ax, obj, **kwargs)
#    elif isinstance(obj, cpp.function.Expression):
#        return mplot_expression(ax, obj, mesh, **kwargs)
#    elif isinstance(obj, cpp.mesh.Mesh):
#        return mplot_mesh(ax, obj, **kwargs)
#    elif isinstance(obj, cpp.fem.DirichletBC):
#        return mplot_dirichletbc(ax, obj, **kwargs)
#    elif isinstance(obj, _meshfunction_types):
#        return mplot_meshfunction(ax, obj, **kwargs)
#    else:
#        raise AttributeError('Failed to plot %s' % type(obj))
#
#
#def _plot_x3dom(obj, kwargs):
#    if not isinstance(obj, _x3dom_plottable_types):
#        cpp.warning("Don't know how to plot type %s." % type(obj))
#        return
#
#    x3dom = dolfin.X3DOM()
#    out = x3dom.html(obj)
#
#    return out
#
#
#def plot(object, *args, **kwargs):
#    """
#    Plot given object.
#
#    *Arguments*
#        object
#            a :py:class:`Mesh <dolfin.cpp.Mesh>`, a :py:class:`MeshFunction
#            <dolfin.cpp.MeshFunction>`, a :py:class:`Function
#            <dolfin.functions.function.Function>`, a :py:class:`Expression`
#            <dolfin.cpp.Expression>, a :py:class:`DirichletBC`
#            <dolfin.cpp.DirichletBC>, a :py:class:`FiniteElement
#            <ufl.FiniteElement>`.
#
#    *Examples of usage*
#        In the simplest case, to plot only e.g. a mesh, simply use
#
#        .. code-block:: python
#
#            mesh = UnitSquare(4, 4)
#            plot(mesh)
#
#        Use the ``title`` argument to specify title of the plot
#
#        .. code-block:: python
#
#            plot(mesh, tite="Finite element mesh")
#
#        It is also possible to plot an element
#
#        .. code-block:: python
#
#            element = FiniteElement("BDM", tetrahedron, 3)
#            plot(element)
#
#        Vector valued functions can be visualized with an alternative mode
#
#        .. code-block:: python
#
#            plot(u, mode = "glyphs")
#
#        A more advanced example
#
#        .. code-block:: python
#
#            plot(u,
#                 wireframe = True,              # use wireframe rendering
#                 interactive = False,           # do not hold plot on screen
#                 scalarbar = False,             # hide the color mapping bar
#                 hardcopy_prefix = "myplot",    # default plotfile name
#                 scale = 2.0,                   # scale the warping/glyphs
#                 title = "Fancy plot",          # set your own title
#                 )
#
#    """
#
#    # Return if plotting is disables
#    if os.environ.get("DOLFIN_NOPLOT", "0") != "0":
#        return
#
#    # Return if Matplotlib is not available
#    if not _has_matplotlib():
#        cpp.log.info("Matplotlib is required to plot from Python.")
#        return
#
#    # Plot element
#    if isinstance(object, ufl.FiniteElementBase):
#        import ffc
#        return ffc.plot(object, *args, **kwargs)
#
#    # For dolfin.function.Function, extract cpp_object
#    if hasattr(object, "cpp_object"):
#        object = object.cpp_object()
#
#    # Get mesh from explicit mesh kwarg, only positional arg, or via
#    # object
#    mesh = kwargs.pop('mesh', None)
#    if isinstance(object, cpp.mesh.Mesh):
#        if mesh is not None and mesh.id() != object.id():
#            raise RuntimeError("Got different mesh in plot object and keyword argument")
#        mesh = object
#    if mesh is None:
#        if isinstance(object, cpp.function.Function):
#            mesh = object.function_space().mesh()
#        elif hasattr(object, "mesh"):
#            mesh = object.mesh()
#
#    # Expressions do not carry their own mesh
#    if isinstance(object, cpp.function.Expression) and mesh is None:
#        raise RuntimeError("Expecting a mesh as keyword argument")
#
#    backend = kwargs.pop("backend", "matplotlib")
#    if backend not in ("matplotlib", "x3dom"):
#        raise RuntimeError("Plotting backend %s not recognised" % backend)
#
#    # Try to project if object is not a standard plottable type
#    if not isinstance(object, _all_plottable_types):
#        from dolfin.fem.projection import project
#        try:
#            cpp.log.info("Object cannot be plotted directly, projecting to "
#                         "piecewise linears.")
#            object = project(object, mesh=mesh)
#            mesh = object.function_space().mesh()
#            object = object._cpp_object
#        except Exception as e:
#            msg = "Don't know how to plot given object:\n  %s\n" \
#                  "and projection failed:\n  %s" % (str(object), str(e))
#            raise RuntimeError(msg)
#
#    # Plot
#    if backend == "matplotlib":
#        return _plot_matplotlib(object, mesh, kwargs)
#    elif backend == "x3dom":
#        return _plot_x3dom(object, kwargs)
#    else:
#        assert False, "This code should not be reached."
