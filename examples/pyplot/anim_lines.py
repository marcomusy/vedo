"""Animated plot showing multiple temporal data lines"""
# Copyright (c) 2021, Nicolas P. Rougier. License: BSD 2-Clause*
# Adapted for vedo by M. Musy, February 2021
import numpy as np
from vedo import settings, Line, show

settings.default_font = "Theemim"

# Generate random data
np.random.seed(1)
data = np.random.uniform(0, 1, (25, 100))
X = np.linspace(-1, 1, data.shape[-1])
G = 0.15 * np.exp(-4 * X**2) # use a  gaussian as a weight

# Generate line plots
lines = []
for i, d in enumerate(data):
    pts = np.c_[X, np.zeros_like(X)+i/10, G*d]
    lines.append(Line(pts, lw=3))

# Set up the first frame
axes = dict(xtitle=':Deltat /:mus', ytitle="source", ztitle="")
plt = show(lines, __doc__, axes=axes, elevation=-30, interactive=False, bg='k8')

for i in range(50):
    data[:, 1:] = data[:, :-1]                      # Shift data to the right
    data[:, 0] = np.random.uniform(0, 1, len(data)) # Fill-in new values
    for line, d in zip(lines, data):                    # Update data
        v = line.vertices
        v[:,2] = G * d
        line.vertices = v
        line.cmap('gist_heat_r', v[:,2])
    plt.render()

plt.interactive().close()




#############################################################################
# *BSD 2-Clause License
#
# Copyright (c) 2021, Nicolas P. Rougier
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Original version at: https://github.com/rougier/unknown-pleasures

