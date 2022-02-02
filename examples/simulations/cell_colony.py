"""Simulation of bacteria types that divide at a given rate
As they divide they occupy more and more space"""
print(__doc__)
from vedo import Plotter, ProgressBar, Points, Line
import numpy as np


##############################################################
class Colony:
    def __init__(self, cells=(), c="b", cellSize=5):
        self.cells = cells
        self.cellSize = cellSize
        self.color = c


##############################################################
class Cell:
    def __init__(self, pos=(0, 0, 0), tdiv=10):

        self.pos = pos  # position in space

        self.tdiv = tdiv  # after this time cell will divide on average
        self.tdiv_spread = 2  # gaussian spread to determine division time
        self.lag = 2  # cell division cannot happen before this age

        self.split_dist = 0.2  # split at this initial distance
        self.split_dist_spread = 0.05  # gauss split initial distance spread

        self.apoptosis = 90  # after this time cell dies
        self.apoptosis_spread = 1

        self.t = 0  # current absolute clock time
        self.timeOfBirth = 0  # time of life since birth
        self.rndwalk = 0.001  # random walk step

        # decide random life and death times for this specific cell
        self.celltdiv = self.tdiv + np.random.randn() * self.tdiv_spread
        self.celltdiv = max(self.lag, self.celltdiv)
        self.apoptosis += np.random.randn() * self.apoptosis_spread
        self.apoptosis = max(self.lag, self.apoptosis)

    def divideAt(self, t):
        self.t = t
        self.pos += np.random.normal(size=3) * self.rndwalk
        if t > self.timeOfBirth + self.celltdiv:
            return True
        else:
            return False

    def dieAt(self, t):
        if t > self.timeOfBirth + self.apoptosis:
            return True
        else:
            return False

    def split(self):
        c = Cell(self.pos, tdiv=self.tdiv)
        # this generates a random point uniformly on a sphere
        p = np.random.normal(size=3)
        radius = self.split_dist + np.random.normal() * self.split_dist_spread
        ps = np.linalg.norm(p) * 2 / radius
        d = p / ps
        c.pos = self.pos + d
        self.pos = self.pos - d
        self.timeOfBirth = self.t
        c.timeOfBirth = self.t
        return c

    def dist(self, c):
        v = self.pos - np.array(c.pos)
        s = np.linalg.norm(v)
        return v, s


##############################################################################
plt = Plotter(interactive=False, axes=3)

# Let's start with creating 3 colonies of 1 cell each
# of types: red, green and blue, in different positions in space
# and with 3 different rates of division (tdiv in hours)
c1 = Colony([Cell([1, 0, 0], tdiv=8)], c="b")
c2 = Colony([Cell([0, 1, 0], tdiv=9)], c="g")
c3 = Colony([Cell([0, 0, 1], tdiv=10)], c="r")
colonies = [c1, c2, c3]

# time goes from 0 to 90 hours
pb = ProgressBar(0, 50, step=0.1, c=1)
for t in pb.range():
    msg = "[Nb,Ng,Nr,t] = "
    plt.actors = []  # clean up the list of actors

    for colony in colonies:

        newcells = []
        for cell in colony.cells:

            if cell.dieAt(t):
                continue
            if cell.divideAt(t):
                newc = cell.split()  # make daughter cell
                plt += Line(cell.pos, newc.pos, c="k", lw=3, alpha=0.5)
                newcells.append(newc)
            newcells.append(cell)
        colony.cells = newcells

        pts = [c.pos for c in newcells]  # draw all points at once
        plt += Points(pts, c=colony.color, r=5, alpha=0.80)   # nucleus
        plt += Points(pts, c=colony.color, r=15, alpha=0.05)  # halo
        msg += str(len(colony.cells)) + ","

    pb.print(msg + str(int(t)))
    plt.show(resetcam=not t)
    if plt.escaped:
        exit(0)  # if ESC is hit during the loop
