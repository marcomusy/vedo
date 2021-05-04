"""Demo to show how to solve the Tower of Hanoi"""

from vedo import Plotter, Cylinder, Box, ProgressBar, settings, interactive
from copy import deepcopy


class Hanoi:
    """
    Class to solve the Hanoi problem. It is taken from Geert Jan Bex's website:
    https://github.com/gjbex/training-material/blob/master/Misc/Notebooks/hanoi.ipynb
    Creative Commons Zero v1.0 Universal licence
    """
    def __init__(self, nr_disks):
        self._nr_disks = nr_disks
        self._towers = [list(range(nr_disks, 0, -1)), list(), list()]

    @property
    def nr_disks(self):
        return self._nr_disks

    @property
    def nr_moves(self):
        return 2**self.nr_disks - 1

    @property
    def towers(self):
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

    def moves(self):
        yield from self.move_disks(self.nr_disks, 0, 1)


nr_disks=5
hanoi = Hanoi(nr_disks)

tower_states = list([hanoi.towers])
for _ in hanoi.moves():
    tower_states.append(hanoi.towers)
disks = { hanoi.nr_disks - i : Cylinder(r=0.2*(hanoi.nr_disks-i+1), c=i)
          for i in range(hanoi.nr_disks) }

settings.useDepthPeeling = False
plt = Plotter(interactive=False, size=(800, 600), bg='wheat', bg2='lb')
plt.camera.SetPosition( [14.57, -14.772, 6.132] )
plt.camera.SetFocalPoint( [3.0, 0.6, 2.0] )
plt.camera.SetViewUp( [-0.098, 0.167, 0.981] )
plt += disks.values()
plt += Box(pos=(3,0,-0.5), size=(12,4,0.1))

pb = ProgressBar(0, len(tower_states), 1, c="y")
for t in pb.range():
    pb.print()
    state = tower_states[t]
    for tower_nr in range(3):
        for i, disk in enumerate(state[tower_nr]):
            disks[disk].pos([3 * tower_nr, 0, i+0.5])
    plt.show(resetcam=0, rate=10)
interactive().close()

