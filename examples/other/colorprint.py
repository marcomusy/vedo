# Available colors: 
#  0-black, 1-red, 2-green, 3-yellow, 4-blue, 5-magenta, 6-cyan, 7-white
# Available modifiers:
#  c (foreground color), bc (background color)
#  hidden, bold, blink, underline, dim, invert, box

from vtkplotter import printc

printc(' 1- Change the world by being yourself - Amy Poehler', c=1)
printc(' 2- Never regret anything that made you smile - Mark Twain', c='r', bold=0)
printc(' 3- Every moment is a fresh beginning - T.S Eliot', c='m', underline=1)
printc(' 4- Die with memories, not dreams - Unknown', blink=1, bold=0)
printc(' 5- Aspire to inspire before we expire - Unknown', c=2, hidden=1)
printc(' 6- Everything you can imagine is real - Pablo Picasso', c=3)
printc(' 7- Simplicity is the ultimate sophistication - Leonardo da Vinci', c=4)
printc(' 8- Whatever you do, do it well - Walt Disney', c=3, bc=1)
printc(' 9- What we think, we become - Buddha', c=6, invert=1)
printc('10- All limitations are self-imposed - Oliver Wendell Holmes', c=7, dim=1)
printc('11- When words fail, music speaks - Shakespeare')
printc('12- If you tell the truth you dont have to remember anything',
       'Mark Twain', separator=' - ', underline=1, invert=1, c=6, dim=1)

printc(299792.48, c=4, box='*') 

import vtk
printc( 'Any string', True, 455.5, vtk.vtkActor,  c='green', box='=', invert=1)

