"""A quaternion is a compact way to represent a 3D rotation.
It is useful because:

1. it avoids the gimbal-lock issues of Euler angles,
2. it converts cleanly to a rotation matrix,
3. it can be interpolated smoothly with SLERP.

Left panel:
- the gray object is the original local frame;
- the colored object is the same geometry rotated by a quaternion;
- the dashed line is the rotation axis.

Right panel:
- copies of the same object are generated with
  Quaternion().slerp(t, q_target)
  for different values of t between 0 and 1.
"""
import numpy as np
from vedo import Plotter, Quaternion, Sphere, Cube, merge, Arrow, DashedLine, Text3D, settings


settings.default_font = "Calco"


def make_frame(origin=(0, 0, 0), quaternion=None, scale=1.0, alpha=1.0):
    origin = np.asarray(origin, dtype=float)
    basis = np.eye(3) * scale
    if quaternion is not None:
        basis = np.array([quaternion.rotate(v) for v in basis])
    colors = ("red5", "green5", "blue5")
    return [
        Arrow(origin, origin + vec, s=0.003).c(color).alpha(alpha)
        for vec, color in zip(basis, colors)
    ]


axis = np.array([1.0, 1.0, 0.4])
axis /= np.linalg.norm(axis)
angle = 120
q = Quaternion.from_axis_angle(angle, axis)

print("\nQuaternions")
print("axis =", np.array2string(axis, precision=3, suppress_small=True))
print("angle =", angle, "deg")
print("wxyz =", np.array2string(q.wxyz, precision=4, suppress_small=True))
print("matrix3x3 =")
print(np.array2string(q.matrix3x3, precision=3, suppress_small=True))
print("Rotate [1, 0, 0] ->", np.array2string(q.rotate([1, 0, 0]), precision=3, suppress_small=True))
print(q)

# An asymmetric little object makes rotations easier to see.
body = Cube().scale([0.75, 0.22, 0.18]).x(0.35)
nose = Sphere(r=0.11).pos(0.82, 0, 0)
shape = merge(body, nose).lighting("glossy")

left_msg = (
    "Quaternion = one rigid 3D rotation\n"
    f"Axis-angle input: {angle} deg around {np.array2string(axis, precision=2, suppress_small=True)}\n"
    f"wxyz = {np.array2string(q.wxyz, precision=3, suppress_small=True)}\n"
    "The same quaternion can rotate vectors, geometry,\n"
    "or be turned into a matrix."
)

right_msg = (
    "SLERP = spherical linear interpolation\n"
    "This is why quaternions are popular for cameras, keyframes and robotics.\n"
    "Each copy below uses Quaternion().slerp(t, q_target)."
)

left_objs = [
    Sphere(r=1.15).wireframe().c("gray7").alpha(0.15),
    DashedLine(-1.4 * axis, 1.4 * axis).c("black"),
    shape.clone().c("gray6").alpha(0.15),
    shape.clone().apply_transform(q.to_transform()).c("orange5").alpha(0.55),
]
left_objs += make_frame(scale=1.0, alpha=0.45)
left_objs += make_frame(quaternion=q, scale=1.0, alpha=1.0)

slerp_objs = []
palette = ["blue5", "cyan5", "green5", "yellow5", "orange5", "red4", "red5"]
for x, t, color in zip(np.linspace(-3.9, 3.9, 7), np.linspace(0, 1, 7), palette):
    qi = Quaternion().slerp(t, q)
    ang_i, _ = qi.angle_axis()

    obj = shape.clone().apply_transform(qi.to_transform()).shift(x, 0, 0)
    obj.c(color).alpha(0.85)
    slerp_objs.append(obj)
    slerp_objs += make_frame(origin=(x, 0, 0), quaternion=qi, scale=0.55, alpha=0.8)
    slerp_objs.append(Text3D(f"{ang_i:3.0f}^o", s=0.08, c='r2').pos(x, -0.78, 0))

plt = Plotter(N=2, size=(1800, 850), axes=1, sharecam=False)
plt.at(0).show(left_msg, *left_objs, viewup="z")
plt.at(1).show(right_msg, *slerp_objs, viewup="z")
plt.interactive().close()
