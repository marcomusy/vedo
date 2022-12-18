
print("IMPORTING vtkclasses")
try:
    import vedo.vtkclasses
except:
    assert False
    exit(1)

print("importing vtkclasses success")

import numpy as np
print("NUMPY Version:", np.__version__)


