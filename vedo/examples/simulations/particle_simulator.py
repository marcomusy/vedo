"""
Particle Simulator

Simulates interacting charged particles in 3D space.
Can also be imported as an independent module.
"""
# By Tommy Vandermolen, 3 August 2018
print(__doc__)
from vedo import Plotter, Cube, Sphere, mag2, versor, vector
import numpy as np

K_COULOMB = 8987551787.3681764  # N*m^2/C^2
vp = None  # so that it can be also used without visualization


class ParticleSim:
    def __init__(self, dt, iterations):
        """ Creates a new particle simulator
                dt: time step, time between successive calculations of particle motion
        """
        self.dt = dt
        self.particles = []
        self.iterations = iterations

    def add_particle(
        self,
        pos=(0, 0, 0),
        charge=1e-6,
        mass=1e-3,
        radius=0.005,
        color=None,
        vel=(0, 0, 0),
        fixed=False,
        negligible=False,
    ):
        """ Adds a new particle with specified properties (in SI units) """
        color = color or len(self.particles)  # assigned or default color number

        p = Particle(pos, charge, mass, radius, color, vel, fixed, negligible)
        self.particles.append(p)

    def simulate(self):
        """ Runs the particle simulation. Simulates one time step, dt, of the particle motion.
            Calculates the force between each pair of particles and updates particles' motion accordingly
        """
        # Main simulation loop
        for i in range(self.iterations):
            for a in self.particles:
                if a.fixed:
                    continue
                ftot = vector(0, 0, 0)  # total force acting on particle a
                for b in self.particles:
                    if a.negligible and b.negligible or a == b:
                        continue
                    ab = a.pos - b.pos
                    ftot += ((K_COULOMB * a.charge * b.charge) / mag2(ab)) * versor(ab)
                a.vel += ftot / a.mass * self.dt  # update velocity and position of a
                a.pos += a.vel * self.dt
                a.vsphere.pos(a.pos)
            if vp:
                vp.show(zoom=1.2)
                vp.camera.Azimuth(0.1)  # rotate camera


class Particle:
    def __init__(self, pos, charge, mass, radius, color, vel, fixed, negligible):
        """ Creates a new particle with specified properties (in SI units)
                pos: XYZ starting position of the particle, in meters
                charge: charge of the particle, in Coulombs
                mass: mass of the particle, in kg
                radius: radius of the particle, in meters. No effect on simulation
                color: color of the particle. If None, a default color will be chosen
                vel: initial velocity vector, in m/s
                fixed: if True, particle will remain fixed in place
                negligible: assume charge is small wrt other charges to speed up calculation
        """
        self.pos = vector(pos)
        self.radius = radius
        self.charge = charge
        self.mass = mass
        self.vel = vector(vel)
        self.fixed = fixed
        self.negligible = negligible
        self.color = color
        if vp:
            self.vsphere = vp.add(Sphere(pos, r=radius, c=color))  # Sphere representing the particle
            # vp.addTrail(alpha=0.4, maxlength=1, n=50)
            # Add a trail behind the particle
            self.vsphere.addTrail(alpha=0.4, maxlength=1, n=50)


#####################################################################################################
if __name__ == "__main__":

    # An example simulation of N particles scattering on a charged target.
    # See e.g. https://en.wikipedia.org/wiki/Rutherford_scattering

    vp = Plotter(title="Particle Simulator", bg="black", axes=0, interactive=False)

    vp += Cube(c="white").wireframe(1) # a wireframe cube

    sim = ParticleSim(dt=5e-6, iterations=200)
    sim.add_particle((-0.4, 0, 0), color="w", charge=3e-6, radius=0.01, fixed=True)  # the target

    positions = np.random.randn(500, 3) / 60  # generate a beam of 500 particles
    for p in positions:
        p[0] = -0.5  # Fix x position. Their charge are small/negligible compared to target:
        sim.add_particle(p, charge=0.01e-6, mass=0.1e-6, vel=(1000, 0, 0), negligible=True)

    sim.simulate()
    vp.show(interactive=True, resetcam=False)
