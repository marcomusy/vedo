"""
Demo to show how to solve the Tower of Hanoi
"""
from vtkplotter import *

class Hanoi:
    """
    Class to solve the Hanoi problem. It is taken from
    Geert Jan Bex's website

    https://github.com/gjbex/training-material/blob/master/Misc/Notebooks/hanoi.ipynb

    with licence

    Creative Commons Zero v1.0 Universal

    The Creative Commons CC0 Public Domain Dedication
    waives copyright interest in a work you've created
    and dedicates it to the world-wide public domain.
    Use CC0 to opt out of copyright entirely and ensure
    your work has the widest reach. As with the Unlicense
    and typical software licenses, CC0 disclaims
    warranties. CC0 is very similar to the Unlicense.
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
            print('disk {0} from {1} to {2}'.format(*move))

    def moves(self):
        yield from self.move_disks(self.nr_disks, 0, 1)


def demo3d_hanoi(**kwargs):
    nr_disks = kwargs.get("nr_disks", 5)
    create_png = kwargs.get("create_png", False)
    create_gif = kwargs.get("create_gif", False)
    display_rate = kwargs.get("display_rate", 1)
    interactive = kwargs.get("interactive", 1)
    if create_gif: create_png = True

    hanoi = Hanoi(nr_disks)
    tower_states = list([hanoi.towers])
    for _ in hanoi.moves():
        tower_states.append(hanoi.towers)
    vp = Plotter(axes=0, interactive=0, bg="w")
    cols = makePalette("red", "blue", hanoi.nr_disks + 1, hsv=True)
    disks = {hanoi.nr_disks-i : \
              Cylinder(pos=[0, 0, 0.0], \
                       r=0.2 * (hanoi.nr_disks - i + 1), \
                       c=cols[i]) \
                  for i in range(hanoi.nr_disks)}
    for k in disks:
        vp.add(disks[k])
    vp.add(Box(pos=(3.0, 0, -0.05), length=12.0, width=4.0, height=0.1))
    vp.camera.SetPosition([18.5, -20.7, 7.93] )
    vp.camera.SetFocalPoint([3.0, 0.0, 2.5] )
    vp.camera.SetViewUp([-0.1, +0.17, 0.977] )
    vp.camera.SetDistance( 26.0)
    vp.show()

    list_of_images = []
    pb = ProgressBar(0, len(tower_states), 1, c="b")
    for t in pb.range():
        pb.print()
        state = tower_states[t]
        for tower_nr in range(3):
            for i, disk in enumerate(state[tower_nr]):
                disks[disk].pos([3 * tower_nr, 0, i + 0.5])
        vp.show(resetcam=0, interactive=interactive, rate=display_rate)
        if create_png:
            vp.show(resetcam=0, interactive=0)
            output = 'Hanoi_{0:02d}.png'.format(t)
            screenshot(output)
            list_of_images.append(output)
    if create_gif:
        create_animated_gif(filename='Hanoi_{0:02d}.gif'.format(nr_disks),
                            pngs=list_of_images)


def create_animated_gif(filename='Hanoi.gif', **kwargs):
    import subprocess
    pngs = kwargs.get('pngs', None)
    if pngs is None:
        from glob import glob
        pngs = glob('*.png')
    cmd = 'convert -antialias -density 100 -delay 40 '
    cmd += ' '.join(pngs)
    cmd += ' ' + filename
    subprocess.check_output(cmd.split(' '))


def main(cli=None):
    import argparse
    parser = argparse.ArgumentParser(
        description='Create a 3D images showing how to solve the Tower of Hanoi.')
    pa = parser.add_argument
    pa("-n", "--nr_disks", type=int, default=5,
        help="Number of disks")
    pa("-i", "--interactive", action='store_true',
        help='Request user to press keyboard to display next step')
    pa("-r", "--display_rate", type=int, default=2,
        help="Change display rate to speed up or slow down animation")
    pa("-p", "--png", action='store_true',
        help='Create pngs')
    pa("-g", "--gif", action='store_true',
        help="Create an animated gif. Require convert program from imagemagick")
    args = parser.parse_args(cli)
    demo3d_hanoi(nr_disks=args.nr_disks,
                 display_rate=args.display_rate,
                 create_png=args.png,
                 create_gif=args.gif)


if __name__ == "__main__":
    main()
