"""Minimal 1D histogram example."""
import numpy as np
from vedo.pyplot import histogram

# Sample from a standard normal distribution.
data = np.random.randn(1000)
histogram(data).show().close()
