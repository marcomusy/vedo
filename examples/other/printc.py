# Available modifiers:
#  c (foreground color), bc (background color)
#  bold, blink, underLine, dim, invert, box
from vedo import printc

printc(":world: 1- Change the world by being yourself - Amy Poehler", c=1)
printc(":smile: 2- Never regret anything that made you smile - Mark Twain", c="r", bold=0)
printc(":construction: 3- Every moment is a fresh beginning - T.S Eliot", c="m", underline=1)
printc(":thumbup: 4- Die with memories, not dreams - Unknown", blink=1, bold=0)
printc(":pin: 5- When words fail, music speaks - Shakespeare")
printc(":rocket: 6- Everything you can imagine is real - Pablo Picasso", c=3)
printc(":idea: 7- Simplicity is the ultimate sophistication - Leonardo da Vinci", c=4)
printc(":rainbow: 8- Whatever you do, do it well - Walt Disney", c=3, bc=1)
printc(":target: 9- What we think, we become - Buddha", c=6, invert=1)
printc(":sparks:10- All limitations are self-imposed - Oliver Wendell Holmes", c=7, dim=1)
printc(":checked:11- If you tell the truth you don't have to remember anything - Mark Twain",
        underline=True,
        invert=True,
        c="indigo9",
)

from vedo.colors import emoji
for k in emoji.keys():
    print(emoji[k], "\t", k)
