"""Select points inside a closed mesh (https://github.com/marcomusy/vedo/issues/1308).

Run: python issue_1308.py
Check without opening a window: python issue_1308.py --no-show
"""

import argparse

import numpy as np
from vedo import Points, Torus, show


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    # A torus tests a concave surface: points in its central hole are outside.
    ring_radius, tube_radius = 1.0, 0.35
    mesh = Torus(r1=ring_radius, r2=tube_radius, res=48, quads=True)
    mesh.triangulate().clean()
    if not mesh.is_closed() or not mesh.is_manifold():
        raise ValueError("Containment requires a closed, manifold surface")

    rng = np.random.default_rng(1308)
    xyz = rng.uniform([-1.5, -1.5, -0.6], [1.5, 1.5, 0.6], (5000, 3))
    points = Points(xyz)

    # This is the recommended operation. Pass your own mesh and points here.
    ids = mesh.inside_points(points, return_ids=True)
    mask = points.pointdata["IsInside"].astype(bool)
    inside = Points(xyz[ids], c="green4", r=5)
    outside = Points(xyz[~mask], c="red4", r=2).alpha(0.12)
    # Alternatively: inside = mesh.inside_points(points)

    # Independent check against the exact torus equation. Exclude points very
    # near the ideal surface because the triangular mesh only approximates it.
    radial = np.hypot(xyz[:, 0], xyz[:, 1])
    exact_distance = np.hypot(radial - ring_radius, xyz[:, 2]) - tube_radius
    away = np.abs(exact_distance) > 0.01
    errors = np.count_nonzero(mask[away] != (exact_distance[away] < 0))
    print(f"Inside: {len(ids)} / {len(xyz)}; analytic mismatches: {errors}")
    assert errors == 0, "Containment disagrees with the analytic reference"
    assert not mesh.contains((0, 0, 0)), "The central hole must be outside"

    # For consistently outward-oriented faces, signed distances are negative
    # inside. Reversing face orientation flips the sign, but not containment.
    distance = points.distance_to(mesh, signed=True)
    reversed_mesh = mesh.clone().reverse()
    reversed_distance = points.distance_to(reversed_mesh, signed=True)
    reversed_ids = reversed_mesh.inside_points(points, return_ids=True)
    assert np.array_equal(ids, reversed_ids)
    assert np.allclose(distance, -reversed_distance)
    print("Reversing faces preserves containment and flips distance signs.")

    if not args.no_show:
        show(
            mesh.c("lightblue").alpha(0.25), inside, outside,
            "Green: inside | Red: outside (including the central hole)",
            axes=1, viewup="z", size=(1000, 750),
        ).close()


if __name__ == "__main__":
    main()
