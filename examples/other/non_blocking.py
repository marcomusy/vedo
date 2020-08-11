"""Non blocking interactive rendering window,
python flow is not blocked
but displayed objects cannot be accessed"""
import time, os
from multiprocessing import Process
from vedo import Sphere, show, printc


printc("..starting main", c='g')

sphere = Sphere().alpha(0.1).lw(0.1)

# ------ instead of (typical):
#show(sphere, __doc__, axes=1)

# ------ spawn an independent subprocess:
def spawn(): show(sphere, __doc__, axes=1)
Process(target=spawn).start()

printc("..python flow is not blocked, wait 1 sec..", c='y')
time.sleep(1)

printc("..continuing in main", c='r')
os._exit(0) # this exits immediately with no cleanup or buffer flushing 
