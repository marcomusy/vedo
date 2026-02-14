import numpy as np
from vedo.pyplot import histogram

data = np.random.randn(1000)
histogram(data).show().close()
