# Available colors:
#  0-black, 1-red, 2-green, 3-yellow, 4-blue, 5-magenta, 6-cyan, 7-white
# Available modifiers:
#  c (foreground color), bc (background color)
#  hidden, bold, blink, underLine, dim, invert, box

from vtkplotter import printc

printc(" 1- Change the world~world by being yourself - Amy Poehler", c=1)
printc(" 2- Never regret anything that made you smile ~smile - Mark Twain", c="r", bold=0)
printc(" 3- Every moment is a fresh beginning - T.S Eliot~copyright", c="m", underline=1)
printc(" 4- Die with memories, not dreams - Unknown~registered", blink=1, bold=0)
printc(" 5- Aspire to inspire before we expire - Unknown", c=2, hidden=1)
printc(" 6- Everything you can imagine is real - Pablo Picasso~rocket", c=3)
printc(" 7- Simplicity is the ultimate sophistication - Leonardo da Vinci~idea", c=4)
printc(" 8- Whatever you do, do it well - ~rainbowWalt Disney", c=3, bc=1)
printc(" 9- What we think, we become - Buddha ~target", c=6, invert=1)
printc("10- All limitations are self-imposed ~sparks Oliver Wendell Holmes", c=7, dim=1)
printc("11- When words fail, music speaks - Shakespeare ~pin")
printc(
    "12- If you tell the truth you dont have to remember anything",
    "Mark Twain~diamomd",
    underline=1,
    invert=1,
    c=6,
    dim=1,
)

printc(299792.48, "km/s ~lightning", c=4, box="*")


import vtk
printc("Any string", True, 455.5, vtk.vtkActor, c="green", box="=", invert=1)


from vtkplotter.colors import emoji
for k in sorted(emoji.keys()):
    print(emoji[k] + " \t" + k)
