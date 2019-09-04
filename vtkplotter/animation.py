from __future__ import division, print_function
import numpy as np
import vtkplotter.utils as utils
from vtkplotter.utils import ProgressBar, linInterpolate
from vtkplotter.colors import printc, getColor
from vtkplotter.vtkio import Video
from vtkplotter.plotter import Plotter

__doc__ = (
    """
Animation module which allows to animate simultaneously various objects
by specifying event times and durations of different visual effects.

See examples
`here <https://github.com/marcomusy/vtkplotter/blob/master/examples/other>`_.

N.B.: this is still an experimental feature at the moment.
"""
)

__all__ = ['Animation']


class Animation(Plotter):
    """
    A ``Plotter`` derived class that allows to animate simultaneously various objects
    by specifying event times and durations of different visual effects.

    :param float totalDuration: expand or shrink the total duration of video to this value
    :param float timeResolution: in secconds, save a frame at this rate
    :param bool showProgressBar: show the progressbar
    :param str videoFileName: output file name of the video
    :param int videoFPS: desired value of the nr of frames per second.
    """

    def __init__(self, totalDuration=None, timeResolution=0.02, showProgressBar=True,
                 videoFileName='animation.mp4', videoFPS=12):
        Plotter.__init__(self)
        self.verbose = False
        self.resetcam = False

        self.events = []
        self.timeResolution = timeResolution
        self.totalDuration = totalDuration
        self.showProgressBar = showProgressBar
        self.videoFileName = videoFileName
        self.videoFPS = videoFPS
        self.bookingMode = True
        self._inputvalues = []
        self._performers = []
        self._lastT = None
        self._lastDuration = None
        self._lastActs = None
        self.eps = 0.00001


    def _parse(self, objs, t, duration):
        if t is None:
            if self._lastT:
                t = self._lastT
            else:
                t = 0.0
        if duration is None:
            if self._lastDuration:
                duration = self._lastDuration
            else:
                duration = 0.0
        if objs is None:
            if self._lastActs:
                objs = self._lastActs
            else:
                printc('Need to specify actors!', c=1)
                raise RuntimeError

        objs2 = objs

        if utils.isSequence(objs):
            objs2 = objs
        else:
            objs2 = [objs]

        #quantize time steps and duration
        t = int(t/self.timeResolution+0.5)*self.timeResolution
        nsteps =   int(duration/self.timeResolution+0.5)
        duration = nsteps*self.timeResolution

        rng = np.linspace(t, t+duration, nsteps+1)

        self._lastT = t
        self._lastDuration = duration
        self._lastActs = objs2

        for a in objs2:
            if a not in self.actors:
                self.actors.append(a)

        return objs2, t, duration, rng


    def switchOn(self, acts=None, t=None, duration=None):
        """Switch on the input list of actors."""
        return self.fadeIn(acts, t, 0)

    def switchOff(self, acts=None, t=None, duration=None):
        """Switch off the input list of actors."""
        return self.fadeOut(acts, t, 0)


    def fadeIn(self, acts=None, t=None, duration=None):
        """Gradually switch on the input list of actors by increasing opacity."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                alpha = linInterpolate(tt, [t,t+duration], [0,1])
                self.events.append((tt, self.fadeIn, acts, alpha))
        else:
            for a in self._performers:
                if a.alpha() >= self._inputvalues:
                    continue
                a.alpha(self._inputvalues)
        return self

    def fadeOut(self, acts=None, t=None, duration=None):
        """Gradually switch off the input list of actors by increasing transparency."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                alpha = linInterpolate(tt, [t,t+duration], [1,0])
                self.events.append((tt, self.fadeOut, acts, alpha))
        else:
            for a in self._performers:
                if a.alpha() <= self._inputvalues:
                    continue
                a.alpha(self._inputvalues)
        return self


    def changeAlphaBetween(self, alpha1, alpha2, acts=None, t=None, duration=None):
        """Gradually change transparency for the input list of actors."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                alpha = linInterpolate(tt, [t,t+duration], [alpha1, alpha2])
                self.events.append((tt, self.fadeOut, acts, alpha))
        else:
            for a in self._performers:
                a.alpha(self._inputvalues)
        return self


    def changeColor(self,  c, acts=None, t=None, duration=None):
        """Gradually change color for the input list of actors."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)

            col2 = getColor(c)
            for tt in rng:
                inputvalues = []
                for a in acts:
                    col1 = a.color()
                    r = linInterpolate(tt, [t,t+duration], [col1[0], col2[0]])
                    g = linInterpolate(tt, [t,t+duration], [col1[1], col2[1]])
                    b = linInterpolate(tt, [t,t+duration], [col1[2], col2[2]])
                    inputvalues.append((r,g,b))
                self.events.append((tt, self.changeColor, acts, inputvalues))
        else:
            for i,a in enumerate(self._performers):
                a.color(self._inputvalues[i])
        return self


    def changeBackColor(self, c, acts=None, t=None, duration=None):
        """Gradually change backface color for the input list of actors.
        An initial backface color should be set in advance."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)

            col2 = getColor(c)
            for tt in rng:
                inputvalues = []
                for a in acts:
                    if a.GetBackfaceProperty():
                        col1 = a.backColor()
                        r = linInterpolate(tt, [t,t+duration], [col1[0], col2[0]])
                        g = linInterpolate(tt, [t,t+duration], [col1[1], col2[1]])
                        b = linInterpolate(tt, [t,t+duration], [col1[2], col2[2]])
                        inputvalues.append((r,g,b))
                    else:
                        inputvalues.append(None)
                self.events.append((tt, self.changeBackColor, acts, inputvalues))
        else:
            for i,a in enumerate(self._performers):
                a.backColor(self._inputvalues[i])
        return self


    def changeToWireframe(self, acts=None, t=None):
        """Switch representation to wireframe for the input list of actors at time `t`."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, None)
            self.events.append((t, self.changeToWireframe, acts, True))
        else:
            for a in self._performers:
                a.wireframe(self._inputvalues)
        return self

    def changeToSurface(self, acts=None, t=None):
        """Switch representation to surface for the input list of actors at time `t`."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, None)
            self.events.append((t, self.changeToSurface, acts, False))
        else:
            for a in self._performers:
                a.wireframe(self._inputvalues)
        return self


    def changeLineWidth(self, lw, acts=None, t=None, duration=None):
        """Gradually change line width of the mesh edges for the input list of actors."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                inputvalues = []
                for a in acts:
                    newlw = linInterpolate(tt, [t,t+duration], [a.lw(), lw])
                    inputvalues.append(newlw)
                self.events.append((tt, self.changeLineWidth, acts, inputvalues))
        else:
            for i,a in enumerate(self._performers):
                a.lw(self._inputvalues[i])
        return self


    def changeLineColor(self, c, acts=None, t=None, duration=None):
        """Gradually change line color of the mesh edges for the input list of actors."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            col2 = getColor(c)
            for tt in rng:
                inputvalues = []
                for a in acts:
                    col1 = a.lineColor()
                    r = linInterpolate(tt, [t,t+duration], [col1[0], col2[0]])
                    g = linInterpolate(tt, [t,t+duration], [col1[1], col2[1]])
                    b = linInterpolate(tt, [t,t+duration], [col1[2], col2[2]])
                    inputvalues.append((r,g,b))
                self.events.append((tt, self.changeLineColor, acts, inputvalues))
        else:
            for i,a in enumerate(self._performers):
                a.lineColor(self._inputvalues[i])
        return self


    def changeLighting(self, style, acts=None, t=None, duration=None):
        """Gradually change the lighting style for the input list of actors.

        Allowed styles are: [metallic, plastic, shiny, glossy, default].
        """
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)

            c = (1,1,0.99)
            if   style=='metallic': pars = [0.1, 0.3, 1.0, 10, c]
            elif style=='plastic' : pars = [0.3, 0.4, 0.3,  5, c]
            elif style=='shiny'   : pars = [0.2, 0.6, 0.8, 50, c]
            elif style=='glossy'  : pars = [0.1, 0.7, 0.9, 90, c]
            elif style=='default' : pars = [0.1, 1.0, 0.05, 5, c]
            else:
                printc('Unknown lighting style:', [style], c=1)

            for tt in rng:
                inputvalues = []
                for a in acts:
                    pr = a.GetProperty()
                    aa = pr.GetAmbient()
                    ad = pr.GetDiffuse()
                    asp = pr.GetSpecular()
                    aspp = pr.GetSpecularPower()
                    naa  = linInterpolate(tt, [t,t+duration], [aa,  pars[0]])
                    nad  = linInterpolate(tt, [t,t+duration], [ad,  pars[1]])
                    nasp = linInterpolate(tt, [t,t+duration], [asp, pars[2]])
                    naspp= linInterpolate(tt, [t,t+duration], [aspp,pars[3]])
                    inputvalues.append((naa, nad, nasp, naspp))
                self.events.append((tt, self.changeLighting, acts, inputvalues))
        else:
            for i,a in enumerate(self._performers):
                pr = a.GetProperty()
                vals = self._inputvalues[i]
                pr.SetAmbient(vals[0])
                pr.SetDiffuse(vals[1])
                pr.SetSpecular(vals[2])
                pr.SetSpecularPower(vals[3])
        return self


    def move(self, act=None, pt=(0,0,0), t=None, duration=None, style='linear'):
        """Smoothly change the position of a specific object to a new point in space."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(act, t, duration)
            if len(acts) != 1:
                printc('Error in move(), can move only one object.', c=1)
            cpos = acts[0].pos()
            pt = np.array(pt)
            dv = (pt - cpos)/len(rng)
            for j,tt in enumerate(rng):
                i = j+1
                if 'quad' in style:
                    x = i/len(rng)
                    y = x*x
                    #print(x,y)
                    self.events.append((tt, self.move, acts, cpos+dv*i*y))
                else:
                    self.events.append((tt, self.move, acts, cpos+dv*i))
        else:
            self._performers[0].pos(self._inputvalues)
        return self


    def rotate(self, act=None, axis=(1,0,0), angle=0, t=None, duration=None):
        """Smoothly rotate a specific object by a specified angle and axis."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(act, t, duration)
            if len(acts) != 1:
                printc('Error in rotate(), can move only one object.', c=1)
            for tt in rng:
                ang  = angle/len(rng)
                self.events.append((tt, self.rotate, acts, (axis, ang)))
        else:
            ax = self._inputvalues[0]
            if   ax == 'x':
                self._performers[0].rotateX(self._inputvalues[1])
            elif ax == 'y':
                self._performers[0].rotateY(self._inputvalues[1])
            elif ax == 'z':
                self._performers[0].rotateZ(self._inputvalues[1])
        return self


    def scale(self, acts=None, factor=1, t=None, duration=None):
        """Smoothly scale a specific object to a specified scale factor."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                fac = linInterpolate(tt, [t,t+duration], [1, factor])
                self.events.append((tt, self.scale, acts, fac))
        else:
            for a in self._performers:
                a.scale(self._inputvalues)
        return self


    def meshErode(self, act=None, corner=6, t=None, duration=None):
        """Erode a mesh by removing cells that are close to one of the 6 corners
        of the bounding box.
        """
        if self.bookingMode:
            acts, t, duration, rng = self._parse(act, t, duration)
            if len(acts) != 1:
                printc('Error in meshErode(), can erode only one object.', c=1)
            diag = acts[0].diagonalSize()
            x0,x1, y0,y1, z0,z1 = acts[0].GetBounds()
            corners = [ (x0,y0,z0), (x1,y0,z0), (x1,y1,z0), (x0,y1,z0),
                        (x0,y0,z1), (x1,y0,z1), (x1,y1,z1), (x0,y1,z1) ]
            pcl = acts[0].closestPoint(corners[corner])
            dmin = np.linalg.norm(pcl - corners[corner])
            for tt in rng:
                d = linInterpolate(tt, [t,t+duration], [dmin, diag*1.01])
                if d>0:
                    ids = acts[0].closestPoint(corners[corner],
                                               radius=d, returnIds=True)
                    if len(ids) <= acts[0].N():
                        self.events.append((tt, self.meshErode, acts, ids))
        else:
            self._performers[0].deletePoints(self._inputvalues)
        return self


    def moveCamera(self, camstart=None, camstop=None, t=None, duration=None):
        """
        Smoothly move camera between two ``vtkCamera`` objects.
        """
        if self.bookingMode:
            if camstart is None:
                if not self.camera:
                    printc("Error in moveCamera(), no camera exists.")
                    return
                camstart = self.camera
            acts, t, duration, rng = self._parse(None, t, duration)
            p1 = np.array(camstart.GetPosition())
            f1 = np.array(camstart.GetFocalPoint())
            v1 = np.array(camstart.GetViewUp())
            c1 = np.array(camstart.GetClippingRange())
            s1 = camstart.GetDistance()

            p2 = np.array(camstop.GetPosition())
            f2 = np.array(camstop.GetFocalPoint())
            v2 = np.array(camstop.GetViewUp())
            c2 = np.array(camstop.GetClippingRange())
            s2 = camstop.GetDistance()
            for tt in rng:
                np1 = linInterpolate(tt, [t,t+duration], [p1,p2])
                nf1 = linInterpolate(tt, [t,t+duration], [f1,f2])
                nv1 = linInterpolate(tt, [t,t+duration], [v1,v2])
                nc1 = linInterpolate(tt, [t,t+duration], [c1,c2])
                ns1 = linInterpolate(tt, [t,t+duration], [s1,s2])
                inps = (np1, nf1, nv1, nc1, ns1)
                self.events.append((tt, self.moveCamera, acts, inps))
        else:
            if not self.camera:
                return
            np1, nf1, nv1, nc1, ns1 = self._inputvalues
            self.camera.SetPosition(np1)
            self.camera.SetFocalPoint(nf1)
            self.camera.SetViewUp(nv1)
            self.camera.SetClippingRange(nc1)
            self.camera.SetDistance(ns1)


    def play(self):
        """Play the internal list of events and save a video."""

        self.events = sorted(self.events, key=lambda x: x[0])
        self.bookingMode = False

        for a in self.actors: a.alpha(0)

        if self.showProgressBar:
            pb = ProgressBar(0, len(self.events), c='g')

        if self.totalDuration is None:
            self.totalDuration = self.events[-1][0] - self.events[0][0]
        vd = Video(self.videoFileName, fps=self.videoFPS, duration=self.totalDuration)

        ttlast=0
        for e in self.events:

            tt, action, self._performers, self._inputvalues = e
            action(0,0)

            dt = tt-ttlast
            if dt > self.eps:
                self.show(interactive=False, resetcam=self.resetcam)
                vd.addFrame()

                if dt > self.timeResolution+self.eps:
                    vd.pause(dt)

            ttlast = tt

            if self.showProgressBar:
                pb.print('t='+str(int(tt*100)/100)+'s,  '+action.__name__)

        self.show(interactive=False, resetcam=self.resetcam)
        vd.addFrame()

        vd.close()
        self.show(interactive=True, resetcam=self.resetcam)
        self.bookingMode = True




################################################################# to do
#    def morph(self, act1=None, act2=None, t=None, duration=None):
#        pass
#
#    def stretch(self, acts=None, p1=None, p2=None, t=None, duration=None):
#        pass
#
#    def crop(self, acts=None,
#             top=None, bottom=None, right=None, left=None, front=None, back=None,
#             t=None, duration=None):
#        pass
#
#    def meshBuild(self):
#        pass
# test with formulas, assemblies








