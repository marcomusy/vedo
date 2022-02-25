# Available modifiers:
#  c (foreground color), bc (background color)
#  bold, blink, underLine, dim, invert, box
from vedo import printc

printc(" 1- Change the world by being yourself - Amy Poehler\world", c=1)
printc(" 2- Never regret anything that made you smile - Mark Twain\smile", c="r", bold=0)
printc(" 3- Every moment is a fresh beginning - T.S Eliot", c="m", underline=1)
printc(" 4- Die with memories, not dreams - Unknown\ethumbup", blink=1, bold=0)
printc(" 5- When words fail, music speaks - Shakespeare \pin")
printc(" 6- Everything you can imagine is real - Pablo Picasso\erocket", c=3)
printc(" 7- Simplicity is the ultimate sophistication - Leonardo da Vinci\idea", c=4)
printc(" 8- Whatever you do, do it well - Walt Disney\erainbow", c=3, bc=1)
printc(" 9- What we think, we become - Buddha \etarget", c=6, invert=1)
printc("10- All limitations are self-imposed - Oliver Wendell Holmes\sparks", c=7, dim=1)
printc("11- If you tell the truth you dont have to remember anything - Mark Twain\checked",
        underline=True,
        invert=True,
        dim=True,
        c=6,
)

from vedo.colors import emoji
for k in sorted(emoji.keys()):
    print(emoji[k] + " \t" + repr(k))
