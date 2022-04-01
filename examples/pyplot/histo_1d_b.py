"""Superimpose and compare histograms"""
import numpy as np
from vedo.pyplot import histogram

np.random.seed(0)
theory = np.random.randn(500).tolist()
backgr = ((np.random.rand(100)-0.5)*6).tolist()
data = np.random.randn(500).tolist() + backgr

# A first histogram:
histo = histogram(
    theory + backgr,
    ylim=(0,90),
    title=__doc__,
    xtitle='measured variable',
    c='red4',
    gap=0,      # no gap between bins
    padding=0,  # no extra spaces
)
# Extract the 11th bin and color it purple
histo.unpack(10).c('purple4')

# Add a second and a third histogram:
# format them like histo, so they can be superimposed
histo += histogram(backgr, like=histo)
histo += histogram(data, like=histo, marker='o', errors=True, fill=False)

histo.show(zoom='tight').close()
