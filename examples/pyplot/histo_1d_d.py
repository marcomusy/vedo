"""Insert a Figure into another (note that the x-axes stay aligned)"""
from vedo import Marker, settings, show, np
from vedo.pyplot import histogram

settings.defaultFont = "Ubuntu"

data = np.random.normal(loc=100, size=1000) + 7

################## Create the first Figure
fig1 = histogram(
    data,
    bins=20,
    xlim=(95,111),
    aspect=16/9,
    xtitle="shifted gaussian",
    c='cyan3',
)
# let's add an asterix marker where the mean is
fig1 += Marker('a', [fig1.mean,150,0.1], s=8).c('orange5')

################## Create a second Figure
fig2 = histogram(
    data - 7,
    bins=60,
    aspect=4/3,
    density=True,
    outline=True,
    c='purple9',
    axes=dict(xyGrid=True, xyPlaneColor='grey2', xyAlpha=1, gridLineWidth=0),
)
# let's add an asterix marker where the mean is
fig2 += Marker('a', [fig2.mean,0.2,0.1], s=0.1).c('orange5')

# shift fig2 in vertical by 25, and in z by 0.1 (to make it show on top)
fig2.shift(0, 25, 0.1)

################## Insert fig2 into fig1
fig1.insert(fig2)

show(fig1, __doc__, zoom='tight', size=(1200,900)).close()

