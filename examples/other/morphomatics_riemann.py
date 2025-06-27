"""Compute the mean of two Bézier splines on a sphere using the Riemannian mean"""
# https://morphomatics.github.io/tutorials/tutorial_bezierfold/
import jax
import jax.numpy as jnp
import numpy as np
from morphomatics.geom import BezierSpline
from morphomatics.manifold import Bezierfold
from morphomatics.manifold import Sphere
import vedo


M = Sphere()
B = Bezierfold(M, 2, 2)

North = jnp.array([0.0, 0.0, 1.0])
South = jnp.array([0.0, 0.0, -1.0])

p1 = jnp.array([1.0, 0.0, 0.0])
o1 = jnp.array([1 / jnp.sqrt(2), 1 / jnp.sqrt(2), 0.0])
om1 = M.connec.exp(o1, jnp.array([0, 0, -0.25]))
op1 = M.connec.exp(o1, jnp.array([0, 0, 0.25]))
q1 = jnp.array([0, 1, 0.0])

B1 = BezierSpline(M, [jnp.stack((p1, om1, o1)), jnp.stack((o1, op1, q1))])

z = M.connec.geopoint(o1, North, 0.5)

p2 = jnp.array([1.0, 0.0, 0.0])
o2 = M.connec.geopoint(p1, z, 0.5)
om2 = M.connec.geopoint(p1, z, 0.4)
op2 = M.connec.geopoint(p1, z, 0.6)
q2 = z
B2 = BezierSpline(M, [jnp.stack((p2, om2, o2)), jnp.stack((o2, op2, q2))])

data = jnp.array([B.to_coords(B1), B.to_coords(B2)])
mean = Bezierfold.FunctionalBasedStructure.mean(B, data)[0]
mean = B.from_coords(mean)

time = jnp.linspace(0.0, 2.0, num=100)
pts1 = np.asarray(jax.vmap(B1.eval)(time))
pts2 = np.asarray(jax.vmap(B2.eval)(time))
mean_pts = np.asarray(jax.vmap(mean.eval)(time))

sphere = vedo.Sphere().c("yellow9")
line1 = vedo.Line(pts1, lw=3).cmap("Blues", time)
line2 = vedo.Line(pts2, lw=3).cmap("Blues", time).add_scalarbar("Time")
line_mean = vedo.Line(mean_pts, c="red5", lw=4)

vedo.show(sphere, line1, line2, line_mean, __doc__, axes=1).close()
