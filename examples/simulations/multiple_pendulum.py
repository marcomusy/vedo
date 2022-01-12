from vedo import Plotter, printc, mag, versor, vector, settings
from vedo import Cylinder, Spring, Box, Sphere
import numpy as np

############## Constants
N = 5  # number of bobs
R = 0.3  # radius of bob (separation between bobs=1)
Ks = 50  # k of springs (masses=1)
g = 9.81  # gravity acceleration
gamma = 0.1  # some friction
Dt = 0.03  # time step

# Create the initial positions and velocitites (0,0) of the bobs
bob_x = [0]
bob_y = [0]
x_dot = np.zeros(N+1)  # velocities
y_dot = np.zeros(N+1)

# Create the bobs
for k in range(1, N + 1):
    alpha = np.pi / 5 * k / 10
    bob_x.append(bob_x[k - 1] + np.cos(alpha) + np.random.normal(0, 0.1))
    bob_y.append(bob_y[k - 1] + np.sin(alpha) + np.random.normal(0, 0.1))

settings.allowInteraction = True

plt = Plotter(title="Multiple Pendulum", axes=0, interactive=0, bg2='ly')
plt += Box(pos=(0, -5, 0), length=12, width=12, height=0.7, c="k").wireframe(1)
sph = Sphere(pos=(bob_x[0], bob_y[0], 0), r=R / 2, c="gray")
plt += sph
bob = [sph]
for k in range(1, N + 1):
    c = Cylinder(pos=(bob_x[k], bob_y[k], 0), r=R, height=0.3, c=k)
    plt += c
    bob.append(c)

# Create the springs out of N links
link = [None] * N
for k in range(N):
    p0 = bob[k].pos()
    p1 = bob[k + 1].pos()
    link[k] = Spring(p0, p1, thickness=0.015, r=R / 3, c="gray")
    plt += link[k]

# Create some auxiliary variables
x_dot_m = np.zeros(N+1)
y_dot_m = np.zeros(N+1)
dij     = np.zeros(N+1) # array with distances to previous bob
dij_m   = np.zeros(N+1)
for k in range(1, N + 1):
    dij[k] = mag([bob_x[k] - bob_x[k - 1], bob_y[k] - bob_y[k - 1]])

fctr = lambda x: (x - 1) / x
Dt *= np.sqrt(1 / g)
Dt2 = Dt / 2  # Midpoint time step
DiaSq = (2 * R) ** 2  # Diameter of bob squared

printc("Press ESC to exit.", c="red", invert=1)

while True:
    bob_x_m = list(map((lambda x, dx: x + Dt2 * dx), bob_x, x_dot))  # midpoint variables
    bob_y_m = list(map((lambda y, dy: y + Dt2 * dy), bob_y, y_dot))

    for k in range(1, N + 1):
        factor = fctr(dij[k])
        x_dot_m[k] = x_dot[k] - Dt2 * (Ks * (bob_x[k] - bob_x[k - 1]) * factor + gamma * x_dot[k])
        y_dot_m[k] = y_dot[k] - Dt2 * (
            Ks * (bob_y[k] - bob_y[k - 1]) * factor + gamma * y_dot[k] + g
        )

    for k in range(1, N):
        factor = fctr(dij[k + 1])
        x_dot_m[k] -= Dt2 * Ks * (bob_x[k] - bob_x[k + 1]) * factor
        y_dot_m[k] -= Dt2 * Ks * (bob_y[k] - bob_y[k + 1]) * factor

    # Compute the full step variables
    bob_x = list(map((lambda x, dx: x + Dt * dx), bob_x, x_dot_m))
    bob_y = list(map((lambda y, dy: y + Dt * dy), bob_y, y_dot_m))

    for k in range(1, N + 1):
        dij[k] = mag([bob_x[k] - bob_x[k - 1], bob_y[k] - bob_y[k - 1]])
        dij_m[k] = mag([bob_x_m[k] - bob_x_m[k - 1], bob_y_m[k] - bob_y_m[k - 1]])
        factor = fctr(dij_m[k])
        x_dot[k] -= Dt * (Ks * (bob_x_m[k] - bob_x_m[k - 1]) * factor + gamma * x_dot_m[k])
        y_dot[k] -= Dt * (Ks * (bob_y_m[k] - bob_y_m[k - 1]) * factor + gamma * y_dot_m[k] + g)

    for k in range(1, N):
        factor = fctr(dij_m[k + 1])
        x_dot[k] -= Dt * Ks * (bob_x_m[k] - bob_x_m[k + 1]) * factor
        y_dot[k] -= Dt * Ks * (bob_y_m[k] - bob_y_m[k + 1]) * factor

    # Check to see if they are colliding
    for i in range(1, N):
        for j in range(i + 1, N + 1):
            dist2 = (bob_x[i] - bob_x[j]) ** 2 + (bob_y[i] - bob_y[j]) ** 2
            if dist2 < DiaSq:  # are colliding
                Ddist = np.sqrt(dist2) - 2 * R
                tau = versor([bob_x[j] - bob_x[i], bob_y[j] - bob_y[i], 0])
                DR = Ddist / 2 * tau
                bob_x[i] += DR[0]  # DR.x
                bob_y[i] += DR[1]  # DR.y
                bob_x[j] -= DR[0]  # DR.x
                bob_y[j] -= DR[1]  # DR.y
                Vji = vector(x_dot[j] - x_dot[i], y_dot[j] - y_dot[i])
                DV = np.dot(Vji, tau) * tau
                x_dot[i] += DV[0]  # DV.x
                y_dot[i] += DV[1]  # DV.y
                x_dot[j] -= DV[0]  # DV.x
                y_dot[j] -= DV[1]  # DV.y

    # Update the loations of the bobs and the stretching of the springs
    for k in range(1, N + 1):
        bob[k].pos([bob_x[k], bob_y[k], 0])
        link[k - 1].stretch(bob[k - 1].pos(), bob[k].pos())

    plt.show()
    if plt.escaped: break  # if ESC is hit during the loop

plt.close()
