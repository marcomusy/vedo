"""Simulation of a flock of boids in a 3D box"""
import vedo
import numpy as np


########################################################################
class Boid:
    def __init__(self, pos, vel=(1,0,0), acc=(0,0,0), c='black'):
        self.xlim = (-3,3)
        self.ylim = (-3,3)
        self.zlim = (-3,3)
        self.max_speed = 0.075
        self.position = np.array(pos)
        self.acceleration = np.array(acc)
        self.velocity = np.array(vel) / np.linalg.norm(vel)
        self.color = c

    def update(self):
        x,y,z = self.position
        if x < self.xlim[0]: self.position[0] = self.xlim[1]
        if x > self.xlim[1]: self.position[0] = self.xlim[0]
        if y < self.ylim[0]: self.position[1] = self.ylim[1]
        if y > self.ylim[1]: self.position[1] = self.ylim[0]
        if z < self.zlim[0]: self.position[2] = self.zlim[1]
        if z > self.zlim[1]: self.position[2] = self.zlim[0]

        self.position += self.velocity * self.max_speed
        self.velocity += self.acceleration
        self.velocity = self.velocity / np.linalg.norm(self.velocity)

########################################################################
class Flock:
    def __init__(self, boids=()):
        self.neighbors  = 20
        self.cohesion   = 0.5
        self.separation = 0.3

        self.boids = list(boids)
        self.actor = None
        self.colors = [vedo.getColor(b.color) for b in boids]

        self.actor = vedo.Points([b.position for b in self.boids], r=8, c=self.colors)

    def positions(self):
        return np.array([b.position for b in self.boids])

    def velocities(self):
        return np.array([b.velocity for b in self.boids])

    def move(self):
        velos = self.velocities()
        coords = self.positions()
        for i,b in enumerate(self.boids):
            ids = self.actor.closestPoint(b.position, N=self.neighbors, returnPointId=True)[1:]

            # alignment: steer boid towards the average heading of local flockmates
            desired_vel = np.mean(velos[ids],  axis=0)
            b.acceleration = desired_vel/np.linalg.norm(desired_vel) - b.velocity

            # cohesion: steer boid to move toward the average position of local flockmates
            desired_pos = np.mean(coords[ids], axis=0)
            b.acceleration += (desired_pos - b.position) * self.cohesion

            # separation: steer boid to avoid crowding local flockmates
            dists = np.linalg.norm(coords[ids] - b.position, axis=1)
            idmin = np.argmin(dists)  # index of min distances in the list
            idpt = ids[idmin]         # index of the point
            b.acceleration += (b.position- coords[idpt]) * self.separation

            b.update()

        self.actor.points(self.positions()) # update all positions
        return self

################################################################################
if __name__=="__main__":

    vedo.settings.allowInteraction = True

    np.random.seed(6)
    boids = []
    for i in range(500):
        c = 'black' if i % 50 else 'red'
        boids.append( Boid(np.random.randn(3), np.random.randn(3), c=c) )
    flock = Flock(boids)

    plt = vedo.Plotter(bg2='lb', interactive=False)
    axes = vedo.Axes(xrange=(-3,3), yrange=(-3,3), zrange=(-3,3), yzGrid=True, zxGrid2=True)
    plt += [__doc__, flock.actor, axes]

    pb = vedo.ProgressBar(0, 100)
    for i in pb.range():
        flock.move()
        plt.show(resetcam=False, viewup='z')
        pb.print()


