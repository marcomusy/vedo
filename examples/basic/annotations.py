'''
Show some text as a corner annotation.
Fonts: arial, courier, times.
'''
from vtkplotter import show, text

with open('basic/align1.py') as fname: t = fname.read()

actor2d = text(t, pos=3, s=1.5, bg='lb', font='courier')

show(actor2d, verbose=0)
