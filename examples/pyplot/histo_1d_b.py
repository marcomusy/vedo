"""Superimpose and compare histograms"""
import numpy as np
from vedo.pyplot import histogram

np.random.seed(0)
theory = np.random.randn(500).tolist()
backgr = ((np.random.rand(100)-0.5)*6).tolist()
data = np.random.randn(500).tolist() + backgr

# A first histogram:
fig = histogram(
    theory + backgr,
    ylim=(0,90),
    title=__doc__,
    xtitle='measured variable',
    c='red4',
    gap=0,      # no gap between bins
    padding=0,  # no extra spaces
    label="theory",
)

# Extract the 11th bin and color it purple
fig.unpack(10).c('purple4')
fig.addLabel("special bin", marker='s', mc='purple4')

# Add a second histogram, format it like histo, so it can be superimposed
fig += histogram(backgr, like=fig, label='background')

# Add the data histogram with poissonian errors
fig += histogram(data, like=fig, marker='o', errors=True, fill=False, label='data')

fig.addLegend(s=0.8)
fig.show(zoom='tight').close()
