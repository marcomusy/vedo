"""
Demo to show how to solve the Tower of Hanoi
"""
from __future__ import division, print_function
from vedo import Plotter, makePalette, Cylinder, Box, ProgressBar, printc


class Hanoi:
    """
    Class to solve the Hanoi problem. It is taken from Geert Jan Bex's website:

    https://github.com/gjbex/training-material/blob/master/Misc/Notebooks/hanoi.ipynb

    with Creative Commons Zero v1.0 Universal licence
    """

    def __init__(self, nr_disks):
        self._nr_disks = nr_disks
        self._towers = [list(range(nr_disks, 0, -1)), list(), list()]

    @property
    def nr_disks(self):
        return self._nr_disks

    @property
    def nr_moves(self):
        return 2 ** self.nr_disks - 1

    @property
    def towers(self):
        from copy import deepcopy

        return deepcopy(self._towers)

    def tower(self, n):
        return self._towers[n].copy()

    def move_disk(self, from_tower, to_tower):
        disk = self._towers[from_tower].pop()
        self._towers[to_tower].append(disk)
        return disk, from_tower, to_tower

    def move_disks(self, n, from_tower, to_tower):
        if n == 1:
            yield self.move_disk(from_tower, to_tower)
        else:
            helper = 3 - from_tower - to_tower
            yield from self.move_disks(n - 1, from_tower, helper)
            yield self.move_disk(from_tower, to_tower)
            yield from self.move_disks(n - 1, helper, to_tower)

    def solve(self):
        for move in self.moves():
            print("disk {0} from {1} to {2}".format(*move))

    def moves(self):
        yield from self.move_disks(self.nr_disks, 0, 1)


def demo3d_hanoi(**kwargs):
    nr_disks = kwargs.get("nr_disks", 5)
    interactive = kwargs.get("interactive", 1)

    hanoi = Hanoi(nr_disks)
    tower_states = list([hanoi.towers])
    for _ in hanoi.moves():
        tower_states.append(hanoi.towers)

    vp = Plotter(axes=0, interactive=0, bg="w", size=(800, 600))
    vp.camera.SetPosition([18.5, -20.7, 7.93])
    vp.camera.SetFocalPoint([3.0, 0.0, 2.5])
    vp.camera.SetViewUp([-0.1, +0.17, 0.977])

    cols = makePalette("red", "blue", hanoi.nr_disks+1, hsv=True)
    disks = {
        hanoi.nr_disks
        -i: Cylinder(pos=[0,0,0], r=0.2 * (hanoi.nr_disks-i+1), c=cols[i])
        for i in range(hanoi.nr_disks)
    }
    for k in disks:
        vp += disks[k]
    vp += Box(pos=(3.0, 0, -.5), length=12.0, width=4.0, height=0.1)
    vp.show(zoom=1.2)

    printc("\n Press q to continue, Esc to exit. ", c="y", invert=1)
    pb = ProgressBar(0, len(tower_states), 1, c="b", ETA=False)
    for t in pb.range():
        pb.print()
        state = tower_states[t]
        for tower_nr in range(3):
            for i, disk in enumerate(state[tower_nr]):
                disks[disk].pos([3 * tower_nr, 0, i + 0.5])
        vp.show(resetcam=0, interactive=interactive, rate=10)
    vp.show(resetcam=0, interactive=1)


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description="Solve the Tower of Hanoi.")
    pa = parser.add_argument
    pa("-n", "--nr_disks", type=int, default=5, help="Number of disks")
    pa("-i", "--interactive", action="store_true",
        help="Request user to press keyboard to display next step",
    )
    args = parser.parse_args()
    demo3d_hanoi(
        nr_disks=args.nr_disks,
        interactive=args.interactive,
    )
