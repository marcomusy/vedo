import numpy as np
import vedo
import vedo.shapes as shapes
import vedo.utils as utils
import vtk
from vedo.colors import printc
from vedo.io import download
from vedo.io import exportWindow
from vedo.io import load
from vedo.io import screenshot
from vedo.io import Video
from vedo.mesh import Mesh
from vedo.plotter import clear
from vedo.plotter import interactive
from vedo.plotter import Plotter
from vedo.plotter import show
from vedo.shapes import Text2D
from vedo.shapes import Text3D
from vedo.utils import ProgressBar

__doc__ = """
Submodule for support of the [FEniCS/Dolfin](https://fenicsproject.org) library.

Example:
    .. code-block:: python

        import dolfin
        from vedo.dolfin import dataurl, download, plot
        fname = download(dataurl+"dolfin_fine.xml")
        mesh = dolfin.Mesh(fname)
        plot(mesh)

    .. image:: https://user-images.githubusercontent.com/32848391/53026243-d2d31900-3462-11e9-9dde-518218c241b6.jpg

Find many more examples in
[vedo/examples/dolfin](https://github.com/marcomusy/vedo/blob/master/vedo/examples/other/dolfin).

.. image:: https://user-images.githubusercontent.com/32848391/58368591-8b3fab80-7eef-11e9-882f-8b8eaef43567.gif
"""


__all__ = [
    "plot",
    "load",
    "download",
    "show",
    "clear",
    "printc",
    "Plotter",
    "ProgressBar",
    "Text3D",
    "Text2D",
    "screenshot",
    "Video",
    "exportWindow",
    "interactive",
]


##########################################################################
def _inputsort(obj):
# def _inputsort_dolfin(obj):
    import dolfin

    u = None
    mesh = None
    if not utils.isSequence(obj):
        obj = [obj]

    for ob in obj:
        inputtype = str(type(ob))

        # printc('inputtype is', inputtype, c=2)

        if "vedo" in inputtype: # skip vtk objects, will be added later
            continue

        if "dolfin" in inputtype or "ufl" in inputtype:
            if "MeshFunction" in inputtype:
                mesh = ob.mesh()

                if ob.dim()>0:
                    printc('MeshFunction of dim>0 not supported.', c='r')
                    printc('Try e.g.:  MeshFunction("size_t", mesh, 0)', c='r', italic=1)
                    printc('instead of MeshFunction("size_t", mesh, 1)', c='r', strike=1)
                else:
                    #printc(ob.dim(), mesh.num_cells(), len(mesh.coordinates()), len(ob.array()))
                    V = dolfin.FunctionSpace(mesh, "CG", 1)
                    u = dolfin.Function(V)
                    v2d = dolfin.vertex_to_dof_map(V)
                    u.vector()[v2d] = ob.array()
            elif "Function" in inputtype or "Expression" in inputtype:
                u = ob
            elif "ufl.mathfunctions" in inputtype: # not working
                u = ob
            elif "Mesh" in inputtype:
                mesh = ob
            elif "algebra" in inputtype:
                mesh = ob.ufl_domain()
                #print('algebra', ob.ufl_domain())

    if u and not mesh and hasattr(u, "function_space"):
        V = u.function_space()
        if V:
            mesh = V.mesh()
    if u and not mesh and hasattr(u, "mesh"):
        mesh = u.mesh()

    #printc('------------------------------------')
    #printc('mesh.topology dim=', mesh.topology().dim())
    #printc('mesh.geometry dim=', mesh.geometry().dim())
    #if u: printc('u.value_rank()', u.value_rank())
    #if u and u.value_rank(): printc('u.value_dimension()', u.value_dimension(0)) # axis=0
    ##if u: printc('u.value_shape()', u.value_shape())
    return (mesh, u)


def _compute_uvalues(u, mesh):
    # the whole purpose of this function is
    # to have a scalar (or vector) for each point of the mesh

    if not u: return
#    print('u',u)

    if hasattr(u, 'compute_vertex_values'): # old dolfin, works fine
        u_values = u.compute_vertex_values(mesh)


        if u.value_rank() and u.value_dimension(0)>1:
            l = u_values.shape[0]
            u_values = u_values.reshape(u.value_dimension(0), int(l/u.value_dimension(0))).T

    elif hasattr(u, 'compute_point_values'): # dolfinx
        u_values = u.compute_point_values()

        try:
            from dolfin import fem
            fvec = u.vector
        except RuntimeError:
            fspace = u.function_space
            try:
                fspace = fspace.collapse()
            except RuntimeError:
                return []

        fvec = fem.interpolate(u, fspace).vector

        tdim = mesh.topology.dim

        #print('fvec.getSize', fvec.getSize(), mesh.num_entities(tdim))
        if fvec.getSize() == mesh.num_entities(tdim):
            # DG0 cellwise function
            C = fvec.get_local()
            if (C.dtype.type is np.complex128):
                print("Plotting real part of complex data")
                C = np.real(C)

        u_values = C

    else:
        u_values = []

    if hasattr(mesh, "coordinates"):
        coords = mesh.coordinates()
    else:
        coords = mesh.geometry.points

    if u_values.shape[0] != coords.shape[0]:
        vedo.logger.warning("mismatch in vedo.dolfin._compute_uvalues")
        u_values = np.array([u(p) for p in coords])
    return u_values


def plot(*inputobj, **options):
    """
    Plot the object(s) provided.

    Input can be any combination of: ``Mesh``, ``Volume``, ``dolfin.Mesh``,
    ``dolfin.MeshFunction``, ``dolfin.Expression`` or ``dolfin.Function``.

    Return the current ``Plotter`` class instance.

    Parameters
    ----------
    mode : str
        one or more of the following can be combined in any order

        - `mesh`/`color`, will plot the mesh, by default colored with a scalar if available
        - `displacement` show displaced mesh by solution
        - `arrows`, mesh displacements are plotted as scaled arrows.
        - `lines`, mesh displacements are plotted as scaled lines.
        - `tensors`, to be implemented

    add : bool
        add the input objects without clearing the already plotted ones

    density : float
        show only a subset of lines or arrows [0-1]

    wire[frame] : bool
        visualize mesh as wireframe [False]

    c[olor] : color
        set mesh color [None]

    exterior : bool
        only show the outer surface of the mesh [False]

    alpha : float
        set object's transparency [1]

    lw : int
        line width of the mesh (set to zero to hide mesh) [0.1]

    ps :  int
        set point size of mesh vertices [None]

    z : float
        add a constant to z-coordinate (useful to show 2D slices as function of time)

    legend : str
        add a legend to the top-right of window [None]

    scalarbar : bool
        add a scalarbar to the window ['vertical']

    vmin : float
        set the minimum for the range of the scalar [None]

    vmax : float
        set the maximum for the range of the scalar [None]

    scale : float
        add a scaling factor to arrows and lines sizes [1]

    cmap : str
        choose a color map for scalars

    shading : str
        mesh shading ['flat', 'phong']

    text : str
        add a gray text comment to the top-left of the window [None]

    isolines : dict
        dictionary of isolines properties

        - n, (int) - add this number of isolines to the mesh
        - c, - isoline color
        - lw, (float) - isoline width
        - z, (float) - add to the isoline z coordinate to make them more visible


    streamlines : dict
        dictionary of streamlines properties

        - probes, (list, None) - custom list of points to use as seeds
        - tol, (float) - tolerance to reduce the number of seed points used in mesh
        - lw, (float) - line width of the streamline
        - direction, (str) - direction of integration ('forward', 'backward' or 'both')
        - maxPropagation, (float) - max propagation of the streamline
        - scalarRange, (list) - scalar range of coloring


    warpZfactor : float
        elevate z-axis by scalar value (useful for 2D geometries)

    warpYfactor : float
        elevate z-axis by scalar value (useful for 1D geometries)

    scaleMeshFactors : list
        rescale mesh by these factors [1,1,1]

    new : bool
        spawn a new instance of Plotter class, pops up a new window

    at : int
        renderer number to plot to

    shape : list
        subdvide window in (n,m) rows and columns

    N : int
        automatically subdvide window in N renderers

    pos : list
        (x,y) coordinates of the window position on screen

    size : list
        window size (x,y)

    title : str
        window title

    bg : color
        background color name of window

    bg2 : color
        second background color name to create a color gradient

    style : int
        choose a predefined style [0-4]

          - 0, `vedo`, style (blackboard background, rainbow color map)
          - 1, `matplotlib`, style (white background, viridis color map)
          - 2, `paraview`, style
          - 3, `meshlab`, style
          - 4, `bw`, black and white style.

    axes : int
        Axes type number.
        Axes type-1 can be fully customized by passing a dictionary ``axes=dict()``.

           - 0,  no axes,
           - 1,  draw customizable grid axes (see below).
           - 2,  show cartesian axes from (0,0,0)
           - 3,  show positive range of cartesian axes from (0,0,0)
           - 4,  show a triad at bottom left
           - 5,  show a cube at bottom left
           - 6,  mark the corners of the bounding box
           - 7,  draw a simple ruler at the bottom of the window
           - 8,  show the `vtkCubeAxesActor` object,
           - 9,  show the bounding box outLine,
           - 10, show three circles representing the maximum bounding box,
           - 11, show a large grid on the x-y plane (use with zoom=8)
           - 12, show polar axes.

    infinity : bool
        if True fugue point is set at infinity (no perspective effects)

    sharecam : bool
        if False each renderer will have an independent vtkCamera

    interactive : bool
        if True will stop after show() to allow interaction w/ window

    offscreen : bool
        if True will not show the rendering window

    zoom : float
        camera zooming factor

    viewup : list, str
        camera view-up direction ['x','y','z', or a vector direction]

    azimuth : float
        add azimuth rotation of the scene, in degrees

    elevation : float
        add elevation rotation of the scene, in degrees

    roll : float
        add roll-type rotation of the scene, in degrees

    camera : dict
        Camera parameters can further be specified with a dictionary
        assigned to the ``camera`` keyword:
        (E.g. `show(camera={'pos':(1,2,3), 'thickness':1000,})`)

        - `pos`, `(list)`,
            the position of the camera in world coordinates

        - `focalPoint`, `(list)`,
            the focal point of the camera in world coordinates

        - `viewup`, `(list)`,
            the view up direction for the camera

        - `distance`, `(float)`,
            set the focal point to the specified distance from the camera position.

        - `clippingRange`, `(float)`,
            distance of the near and far clipping planes along the direction of projection.

        - `parallelScale`, `(float)`,
            scaling used for a parallel projection, i.e. the height of the viewport
            in world-coordinate distances. The default is 1. Note that the "scale" parameter works as
            an "inverse scale", larger numbers produce smaller images.
            This method has no effect in perspective projection mode.

        - `thickness`, `(float)`,
            set the distance between clipping planes. This method adjusts the far clipping
            plane to be set a distance 'thickness' beyond the near clipping plane.

        - `viewAngle`, `(float)`,
            the camera view angle, which is the angular height of the camera view
            measured in degrees. The default angle is 30 degrees.
            This method has no effect in parallel projection mode.
            The formula for setting the angle up for perfect perspective viewing is:
            angle = 2*atan((h/2)/d) where h is the height of the RenderWindow
            (measured by holding a ruler up to your screen) and d is the distance
            from your eyes to the screen.

    interactorStyle : int
        change the style of muose interaction of the scene
    """
    if len(inputobj)==0:
        interactive()
        return

    if 'numpy' in str(type(inputobj[0])):
        from vedo.pyplot import plot as pyplot_plot
        return pyplot_plot(*inputobj, **options)

    mesh, u = _inputsort(inputobj)

    mode = options.pop("mode", 'mesh')
    ttime = options.pop("z", None)

    add = options.pop("add", False)

    wire = options.pop("wireframe", None)

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
    scale = options.pop("scale", 1)
    scaleMeshFactors = options.pop("scaleMeshFactors", [1,1,1])
    shading = options.pop("shading", 'phong')
    text = options.pop("text", None)
    style = options.pop("style", 'vtk')
    isolns = options.pop("isolines", dict())
    streamlines = options.pop("streamlines", dict())
    warpZfactor = options.pop("warpZfactor", None)
    warpYfactor = options.pop("warpYfactor", None)
    lighting = options.pop("lighting", None)
    exterior = options.pop("exterior", False)
    fast = options.pop("fast", False)
    returnActorsNoShow = options.pop("returnActorsNoShow", False)
    at = options.pop("at", 0)

    # refresh axes titles for axes type = 8 (vtkCubeAxesActor)
    xtitle = options.pop("xtitle", 'x')
    ytitle = options.pop("ytitle", 'y')
    ztitle = options.pop("ztitle", 'z')
    if vedo.plotter_instance:
        if xtitle!='x':
            aet = vedo.plotter_instance.axes_instances
            if len(aet)>at and isinstance(aet[at], vtk.vtkCubeAxesActor):
                aet[at].SetXTitle(xtitle)
        if ytitle!='y':
            aet = vedo.plotter_instance.axes_instances
            if len(aet)>at and isinstance(aet[at], vtk.vtkCubeAxesActor):
                aet[at].SetYTitle(ytitle)
        if ztitle!='z':
            aet = vedo.plotter_instance.axes_instances
            if len(aet)>at and isinstance(aet[at], vtk.vtkCubeAxesActor):
                aet[at].SetZTitle(ztitle)


    # change some default to emulate standard behaviours
    if  style == 0 or style == 'vtk':
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
        bg = options.pop('bg', None)
        if bg is None:
            options['bg'] = (82, 87, 110)
        else:
            options['bg'] = bg
        if cmap is None:
            cmap = 'coolwarm'
    elif style == 3 or style == 'meshlab':
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
    if vedo.plotter_instance:
        if add:
            actors = vedo.plotter_instance.actors
        elif at==0: # just remove scalarbars
            for sb in vedo.plotter_instance.scalarbars:
                vedo.plotter_instance.renderer.RemoveActor(sb)

    if mesh and ('mesh' in mode or 'color' in mode or 'displace' in mode):

        actor = MeshActor(u, mesh, exterior=exterior, fast=fast)

        actor.wireframe(wire)
        actor.scale(scaleMeshFactors)
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

        if 'displace' in mode: actor.move(u)

        if cmap and (actor.u_values is not None) and c is None:
            if u.value_rank() > 0: # will show the size of the vector
                actor.cmap(cmap, utils.mag(actor.u_values), vmin=vmin, vmax=vmax)
            else:
                actor.cmap(cmap, actor.u_values, vmin=vmin, vmax=vmax)

        if warpYfactor:
            scals = actor.pointdata[0]
            if len(scals):
                pts_act = actor.points()
                pts_act[:, 1] = scals*warpYfactor*scaleMeshFactors[1]
        if warpZfactor:
            scals = actor.pointdata[0]
            if len(scals):
                pts_act = actor.points()
                pts_act[:, 2] = scals*warpZfactor*scaleMeshFactors[2]
        if warpYfactor or warpZfactor:
            actor.points(pts_act)
            if vmin is not None and vmax is not None:
                actor._mapper.SetScalarRange(vmin, vmax)

        if scbar and c is None:
            if '3d' in scbar:
                actor.addScalarBar3D()
            elif 'h' in scbar:
                actor.addScalarBar(horizontal=True)
            else:
                actor.addScalarBar(horizontal=False)

        if len(isolns) > 0:
            ison = isolns.pop("n", 10)
            isocol = isolns.pop("c", 'black')
            isoalpha = isolns.pop("alpha", 1)
            isolw = isolns.pop("lw", 1)

            isos = actor.isolines(n=ison).color(isocol).lw(isolw).alpha(isoalpha)

            isoz = isolns.pop("z", None)
            if isoz is not None: # kind of hack to make isolines visible on flat meshes
                d = isoz
            else:
                d = actor.diagonalSize()/400
            isos.z(actor.z()+d)
            actors.append(isos)

        actors.append(actor)


    #################################################################
    if 'streamline' in mode:
        mode = mode.replace('streamline','')
        str_act = MeshStreamLines(u, **streamlines)
        actors.append(str_act)

    #################################################################
    if 'arrow' in mode or 'line' in mode:
        if 'arrow' in mode:
            arrs = MeshArrows(u, scale=scale)
        else:
            arrs = MeshLines(u, scale=scale)

        if arrs:
            if legend and 'mesh' not in mode:
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
        if 'vedo' in inputtype:
           actors.append(ob)

    if text:
        # textact = Text2D(text, font=font)
        actors.append(text)

    if 'at' in options.keys() and 'interactive' not in options.keys():
        if vedo.plotter_instance:
            N = vedo.plotter_instance.shape[0]*vedo.plotter_instance.shape[1]
            if options['at'] == N-1:
                options['interactive'] = True

    # if vedo.plotter_instance:
    #     for a2 in vedo.collectable_actors:
    #         if isinstance(a2, vtk.vtkCornerAnnotation):
    #             if 0 in a2.renderedAt: # remove old message
    #                 vedo.plotter_instance.remove(a2)
    #                 break

    if len(actors)==0:
         print('Warning: no objects to show, check mode in plot(mode="...")')

    if returnActorsNoShow:
        return actors

    return show(actors, **options)


###################################################################################
class MeshActor(Mesh):
    """MeshActor, a vedo.Mesh derived object for dolfin support."""

    def __init__(self, *inputobj, **options):

        c = options.pop("c", None)
        alpha = options.pop("alpha", 1)
        exterior = options.pop("exterior", False)
        fast = options.pop("fast", False)
        computeNormals = options.pop("computeNormals", False)

        mesh, u = _inputsort(inputobj)
        if not mesh:
            return

        if exterior:
            import dolfin
            meshc = dolfin.BoundaryMesh(mesh, 'exterior')
        else:
            meshc = mesh

        if hasattr(mesh, "coordinates"):
            coords = mesh.coordinates()
        else:
            coords = mesh.geometry.points

        poly = utils.buildPolyData(coords, meshc.cells(), fast=fast, tetras=True)

        Mesh.__init__(self,
            poly,
            c=c,
            alpha=alpha,
        )
        if computeNormals:
            self.computeNormals()

        self.mesh = mesh  # holds a dolfin Mesh obj
        self.u = u  # holds a dolfin function_data
        # holds the actual values of u on the mesh
        self.u_values = _compute_uvalues(u, mesh)
        #self.addPointArray(self.u_values, "u_values")


    def move(self, u=None, deltas=None):
        """Move mesh according to solution `u` or from calculated vertex displacements `deltas`."""
        if u is None:
            u = self.u
        if deltas is None:
            if self.u_values is not None:
                deltas = self.u_values
            else:
                deltas = _compute_uvalues(u, self.mesh)
                self.u_values = deltas

        if hasattr(self.mesh, "coordinates"):
            coords = self.mesh.coordinates()
        else:
            coords = self.mesh.geometry.points

        if coords.shape != deltas.shape:
            vedo.logger.error(f"Try to move mesh with wrong solution type shape {coords.shape} vs {deltas.shape}")
            vedo.logger.error("Mesh is not moved. Try mode='color' in plot().")
            return

        movedpts = coords + deltas
        if movedpts.shape[1] == 2: #2d
            movedpts = np.c_[movedpts, np.zeros(movedpts.shape[0])]
        self.polydata(False).GetPoints().SetData(utils.numpy2vtk(movedpts, dtype=float))
        self.polydata(False).GetPoints().Modified()


def MeshPoints(*inputobj, **options):
    """Build a point object of type ``Mesh`` for a list of points."""
    r = options.pop("r", 5)
    c = options.pop("c", "gray")
    alpha = options.pop("alpha", 1)

    mesh, u = _inputsort(inputobj)
    if not mesh:
        return None

    if hasattr(mesh, "coordinates"):
        plist = mesh.coordinates()
    else:
        plist = mesh.geometry.points

    u_values = _compute_uvalues(u,mesh)

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
        actor.addPointArray(dispsizes, "u_values")
    return actor


def MeshLines(*inputobj, **options):
    """
    Build the line segments between two lists of points `startPoints` and `endPoints`.
    `startPoints` can be also passed in the form ``[[point1, point2], ...]``.

    A dolfin ``Mesh`` that was deformed/modified by a function can be
    passed together as inputs.

    Use ``scale`` to apply a rescaling factor to the length
    """
    scale = options.pop("scale", 1)
    lw = options.pop("lw", 1)
    c = options.pop("c", 'grey')
    alpha = options.pop("alpha", 1)

    mesh, u = _inputsort(inputobj)
    if not mesh:
        return None

    if hasattr(mesh, "coordinates"):
        startPoints = mesh.coordinates()
    else:
        startPoints = mesh.geometry.points

    u_values = _compute_uvalues(u,mesh)
    if not utils.isSequence(u_values[0]):
        vedo.logger.error("cannot show Lines for 1D scalar values")
        raise RuntimeError()

    endPoints = startPoints + u_values
    if u_values.shape[1] == 2:  # u_values is 2D
        u_values = np.insert(u_values, 2, 0, axis=1)  # make it 3d
        startPoints = np.insert(startPoints, 2, 0, axis=1)  # make it 3d
        endPoints = np.insert(endPoints, 2, 0, axis=1)  # make it 3d

    actor = shapes.Lines(startPoints, endPoints, scale=scale, lw=lw, c=c, alpha=alpha)

    actor.mesh = mesh
    actor.u = u
    actor.u_values = u_values
    return actor


def MeshArrows(*inputobj, **options):
    """Build arrows representing displacements."""
    s = options.pop("s", None)
    c = options.pop("c", "gray")
    scale = options.pop("scale", 1)
    alpha = options.pop("alpha", 1)
    res = options.pop("res", 12)

    mesh, u = _inputsort(inputobj)
    if not mesh:
        return None

    if hasattr(mesh, "coordinates"):
        startPoints = mesh.coordinates()
    else:
        startPoints = mesh.geometry.points

    u_values = _compute_uvalues(u,mesh)
    if not utils.isSequence(u_values[0]):
        vedo.logger.error("cannot show Arrows for 1D scalar values")
        raise RuntimeError()

    endPoints = startPoints + u_values *scale
    if u_values.shape[1] == 2:  # u_values is 2D
        u_values = np.insert(u_values, 2, 0, axis=1)  # make it 3d
        startPoints = np.insert(startPoints, 2, 0, axis=1)  # make it 3d
        endPoints = np.insert(endPoints, 2, 0, axis=1)  # make it 3d

    actor = shapes.Arrows(startPoints, endPoints,
                          s=s, alpha=alpha, res=res)
    actor.color(c)
    actor.mesh = mesh
    actor.u = u
    actor.u_values = u_values
    return actor


def MeshStreamLines(*inputobj, **options):
    """Build a streamplot."""
    from vedo.base import streamLines

    print('Building streamlines...')

    tol = options.pop('tol', 0.02)
    lw = options.pop('lw', 2)
    direction = options.pop('direction', 'forward')
    maxPropagation = options.pop('maxPropagation', None)
    scalarRange = options.pop('scalarRange', None)
    probes = options.pop('probes', None)

    tubes = options.pop('tubes', dict()) # todo
    maxRadiusFactor = options.pop('maxRadiusFactor', 1)
    varyRadius = options.pop('varyRadius', 1)

    mesh, u = _inputsort(inputobj)
    if not mesh:
        return None

    u_values = _compute_uvalues(u, mesh)
    if not utils.isSequence(u_values[0]):
        vedo.logger.error("cannot show Arrows for 1D scalar values")
        raise RuntimeError()
    if u_values.shape[1] == 2:  # u_values is 2D
        u_values = np.insert(u_values, 2, 0, axis=1)  # make it 3d

    meshact = MeshActor(u)
    meshact.addPointArray(u_values, 'u_values')

    if utils.isSequence(probes):
        pass # it's already it
    elif tol:
        print('decimating mesh points to use them as seeds...')
        probes = meshact.clone().subsample(tol).points()
    else:
        probes = meshact.points()
    if len(probes)>500:
        printc('Probing domain with n =', len(probes), 'points')
        printc(' ..this may take time (or choose a larger tol value)')

    if lw:
        tubes = dict()
    else:
        tubes['varyRadius'] = varyRadius
        tubes['maxRadiusFactor'] = maxRadiusFactor

    str_lns = streamLines(meshact, probes,
                          direction=direction,
                          maxPropagation=maxPropagation,
                          tubes=tubes,
                          scalarRange=scalarRange,
                          activeVectors='u_values')

    if lw:
        str_lns.lw(lw)

    return str_lns
