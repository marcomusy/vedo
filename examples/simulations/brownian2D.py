"""Simple demo to illustrate the motion of a Big brownian
particle in a swarm of small particles in 2D motion.
The spheres collide elastically with themselves and
with the walls of the box. The masses of the spheres
are proportional to their radius**3 (as in 3D)"""
# Adapted by M. Musy from E. Velasco (2009)
print(__doc__)

from vedo import Plotter, ProgressBar, dot, Grid, Sphere, Point, settings
import random, numpy as np

screen_w = 800
screen_h = 800
settings.allowInteraction = True

plt = Plotter(size=(screen_w, screen_h), axes=0, interactive=0)

# Constants and time step
Nsp = 200  # Number of small spheres
Rb = screen_w / 32  # Radius of the big sphere
Rs = Rb * 0.43  # Radius of small spheres
Ms = (Rs / Rb) ** 3  # Mass of the small spheres (Mbig=1)
Dt = 0.03  # Time step

LBox = (screen_w / 2, screen_h / 2)  # Size of the box = 2LBox[0].2LBox[1]
Lb0 = LBox[0] - Rb
Lb1 = LBox[1] - Rb
Ls0 = LBox[0] - Rs
Ls1 = LBox[1] - Rs

# Create the arrays with the initial positions of the spheres.
# Start with the big sphere at the center, then put the small
# spheres at random selected from a grid of possible positions.
ListPos = [(0, 0)]
PossiblePos = [
    (x, y)
    for x in np.arange(-LBox[0] + 2 * Rs, LBox[0] - 2 * Rs, 2.2 * Rs)
    for y in np.arange(-LBox[1] + 2 * Rs, LBox[1] - 2 * Rs, 2.2 * Rs)
    if x * x + y * y > Rb + Rs
]

if Nsp > len(PossiblePos) + 1:
    Nsp = len(PossiblePos) + 1

for s in range(Nsp - 1):
    n = random.randint(0, len(PossiblePos) - 1)
    ListPos.append(PossiblePos[n])
    del PossiblePos[n]
Pos = np.array(ListPos)

# Create an array with all the radius and a list with all the masses
Radius = np.concatenate((np.array([Rb]), np.array([Rs] * (Nsp - 1))))
Mass = [1.0] + [Ms] * (Nsp - 1)

# Create the initial array of velocities at random with big sphere at rest
ListVel = [(0.0, 0.0)]
for s in range(1, Nsp):
    ListVel.append((Rb * random.uniform(-1, 1), Rb * random.uniform(-1, 1)))
Vel = np.array(ListVel)

# Create the spheres
Spheres = [Sphere(pos=(Pos[0][0], Pos[0][1], 0), r=Radius[0], c="red", res=12).phong()]
for s in range(1, Nsp):
    a = Sphere(pos=(Pos[s][0], Pos[s][1], 0), r=Radius[s], c="blue", res=6).phong()
    Spheres.append(a)
#    plt += a
plt += Spheres
plt += Grid(s=[screen_w,screen_w])

# Auxiliary variables
Id = np.identity(Nsp)
Dij = (Radius + Radius[:, np.newaxis]) ** 2  # Matrix Dij=(Ri+Rj)**2

# The main loop
pb = ProgressBar(0, 2000, c="r")
for i in pb.range():
    # Update all positions
    np.add(Pos, Vel * Dt, Pos)  # Fast version of Pos = Pos + Vel*Dt

    # Impose the bouncing at the walls
    if Pos[0, 0] <= -Lb0:
        Pos[0, 0] = -Lb0
        Vel[0, 0] = -Vel[0, 0]
    elif Pos[0, 0] >= Lb0:
        Pos[0, 0] = Lb0
        Vel[0, 0] = -Vel[0, 0]
    elif Pos[0, 1] <= -Lb1:
        Pos[0, 1] = -Lb1
        Vel[0, 1] = -Vel[0, 1]
    elif Pos[0, 1] >= Lb1:
        Pos[0, 1] = Lb1
        Vel[0, 1] = -Vel[0, 1]
    for s in range(1, Nsp):
        if Pos[s, 0] <= -Ls0:
            Pos[s, 0] = -Ls0
            Vel[s, 0] = -Vel[s, 0]
        elif Pos[s, 0] >= Ls0:
            Pos[s, 0] = Ls0
            Vel[s, 0] = -Vel[s, 0]
        elif Pos[s, 1] <= -Ls1:
            Pos[s, 1] = -Ls1
            Vel[s, 1] = -Vel[s, 1]
        elif Pos[s, 1] >= Ls1:
            Pos[s, 1] = Ls1
            Vel[s, 1] = -Vel[s, 1]

    # Create the set of all pairs and the list the colliding spheres
    Rij = Pos - Pos[:, np.newaxis]
    Mag2ij = np.add.reduce(Rij * Rij, -1)  # sphere-to-sphere distances**2
    colliding = np.less_equal(Mag2ij, Dij) - Id
    hitlist = np.sort(np.nonzero(colliding.flat)[0]).tolist()

    # Check to see if the spheres are colliding
    for ij in hitlist:
        s1, s2 = divmod(ij, Nsp)  # decode the spheres pair (s1,s2) colliding
        hitlist.remove(s2 * Nsp + s1)  # remove symmetric (s2,s1) pair from list
        R12 = Pos[s2] - Pos[s1]
        nR12 = np.linalg.norm(R12)
        d12 = Radius[s1] + Radius[s2] - nR12
        tau = R12 / nR12
        DR0 = d12 * tau
        x1 = Mass[s1] / (Mass[s1] + Mass[s2])
        x2 = 1 - x1  # x2 = Mass[s2]/(Mass[s1]+Mass[s2])
        Pos[s1] -= x2 * DR0
        Pos[s2] += x1 * DR0
        DV0 = 2 * dot(Vel[s2] - Vel[s1], tau) * tau
        Vel[s1] += x2 * DV0
        Vel[s2] -= x1 * DV0

    # Update the location of the spheres
    for s in range(Nsp):
        Spheres[s].pos([Pos[s][0], Pos[s][1], 0])

    if not int(i) % 10:  # every ten steps:
        rsp = [Pos[0][0], Pos[0][1], 0]
        rsv = [Vel[0][0], Vel[0][1], 0]
        plt += Point(rsp, c="r", r=5, alpha=0.1)  # leave a point trace
        plt.show()  # render scene
        if plt.escaped: break
    pb.print()

plt.interactive().close()
