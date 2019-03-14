from __future__ import division, print_function
import numpy as np

# np.random.seed(0)


##############################################################
class Colony:
    def __init__(self, cells=[], c="b", cellSize=5):
        self.cells = cells
        self.cellSize = cellSize
        self.color = c


##############################################################
class Cell:
    def __init__(self, pos=[0, 0, 0], tdiv=10):

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
