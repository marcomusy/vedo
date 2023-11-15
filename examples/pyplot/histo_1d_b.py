"""Superimpose and compare histograms"""
import numpy as np
from vedo.pyplot import histogram
from vedo import settings

settings.remember_last_figure_format = True

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
fig[10].c('purple4')
fig.add_label("special bin", marker='s', mc='purple4')

# Add a second histogram to be superimposed
fig += histogram(backgr, label='background')

# Add the data histogram with poissonian errors
fig += histogram(data, marker='o', errors=True, fill=False, label='data')

fig.add_legend(s=0.8)
fig.show(zoom='tight').close()
