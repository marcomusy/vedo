"""
Particle Simulator

By Tommy Vandermolen, 3 August 2018

Simulates interacting charged particles in 3D space.
Can also be imported as an independent module.
"""
from __future__ import print_function, division
from vtkplotter import Plotter, ProgressBar, mag, mag2, norm, vector
import itertools
import numpy as np
K_COULOMB = 8987551787.3681764  # N*m^2/C^2


def main():
    """ 
    An example simulation of N particles scattering on a charged target
    See e.g. https://en.wikipedia.org/wiki/Rutherford_scattering
    """

    N = 60
    sim = ParticleSim(dt=1e-6, box_length=2)
    sim.add_particle((0,0,0), charge=15e-6, diameter=0.05, fixed=True)
    positions = np.random.randn(N,3)/25 + vector(-1,0,0)
    for ap in positions:
        sim.add_particle(ap, charge=0.02e-6, mass=0.1e-6, diameter=0.02, vel=(200, 0,0))
 
    sim.simulate()


class ParticleSim:
    def __init__(self, dt=0.01, box_length=0):
        """ Creates a new particle simulator
                dt: time step, time between successive calculations of particle motion
                box_length: the size of the box to be drawn around the simulation, in meters. If 0, no box is drawn """
        self._dt = dt
        self._box_length = box_length
        self._particles = []

        # vtkplotter rendering engine
        self._vp = Plotter(title='Particle Simulator', bg='black', interactive=False, axes=2)
        self._vp.ytitle = ''
        self._vp.ztitle = ''

    def add_particle(self, pos=(0, 0, 0), charge=1e-6, mass=1e-3, diameter=0.03, color=None, vel=(0, 0, 0), fixed=False):
        """ Adds a new particle with specified properties (in SI units)
                pos: the XYZ starting position of the particle, in meters
                charge: the charge of the particle, in Coulombs
                mass: the mass of the particle, in kg
                diameter: the diameter of the particle, in meters. Purely for appearance, no effect on simulation
                color: the color of the particle. If None, a default color will be chosen
                vel: the initial velocity vector, in m/s
                fixed: if True, particle will remain fixed in place """

        color = color or len(self._particles) # or default color number
        
        p = Particle(self._vp, pos, charge, mass, diameter, color, vel, fixed)
        self._particles.append(p)

    def simulate(self):
        """ Runs the particle simulation """

        # Initial camera position
        self._vp.camera.Elevation(20)
        self._vp.camera.Azimuth(40)

        # Wire frame cube
        if self._box_length:
            self._vp.cube((0, 0, 0), length=self._box_length, wire=True, c='white')
        self._vp.show(interactive=False)

        # Main simulation loop
        pb = ProgressBar(0, 200)
        for i in pb.range():
            self._update()
            self._vp.render()
            self._vp.camera.Azimuth(0.1) # Rotate camera
            pb.print()
        self._vp.show(interactive=True, resetcam=False)

    def _update(self):
        """ Simulates one time step, dt, of the particle motion
            Calculates the force between each pair of particles and updates the particles' motion accordingly """

        for a, b in itertools.combinations(self._particles, 2):
            displacement = b.pos - a.pos
            dist_squared = mag2(displacement)
            direction = norm(displacement)

            f = direction * (K_COULOMB * a.charge * b.charge) / dist_squared
            a.update(-f, self._dt)
            b.update(f, self._dt)


class Particle:
    def __init__(self, vp, pos, charge, mass, diameter, color, vel, fixed):
        """ Creates a new particle with specified properties (in SI units)
                vp: the vtkplotter rendering engine for the simulation
                pos: the XYZ starting position of the particle, in meters
                charge: the charge of the particle, in Coulombs
                mass: the mass of the particle, in kg
                diameter: the diameter of the particle, in meters. Purely for appearance, no effect on simulation
                color: the color of the particle. If None, a default color will be chosen
                vel: the initial velocity vector, in m/s
                fixed: if True, particle will remain fixed in place 
        """
        self.pos = np.array(pos, dtype=np.float64)
        self.diameter = diameter
        self.charge = charge
        self.mass = mass
        self.vel = np.array(vel, dtype=np.float64)
        self.fixed = fixed
        self.color = color
        self._vp = vp
        self._old_pos = self.pos.copy()

        # The sphere representing the particle
        self.vtk_actor = vp.sphere(pos, r=diameter/2, c=color) 

        # The trail behind the particle, a list of lines
        # When the last line in the list is used, loop back to the beginning
        def make_trail_iter(length):
            lines = []
            for i in range(length):
                line = self._vp.line((0, 0, 0), (1, 1, 1), c=color, lw=0.2, alpha=0.5)
                vp.addActor(line)
                lines.append(line)
                yield line
            while True:
                for line in lines:
                    yield line
        self._trail_iter = make_trail_iter(100)

    def update(self, force, dt):
        """ Calculates the particle's new position based on the given force and time step """
        if not self.fixed:
            accel = force / self.mass
            self.vel += accel * dt
            self.pos += self.vel * dt

            # Add new segments to the particle's trail once it travelled a significant distance
            if mag(self.pos - self._old_pos) > 0.02:
                line = next(self._trail_iter)
                line.stretch(self._old_pos, self.pos)
                self._old_pos = self.pos.copy()
            self.vtk_actor.pos(self.pos)


if __name__ == '__main__':
    main()
