from vedo import Volume, Text3D, show, dataurl, settings

settings.use_parallel_projection = True

vol = Volume(dataurl+"embryo.slc")
vol.mode(0).c('b9').alpha_unit(1)

t = Text3D("Sharpe\n~~~Lab", s=40, font="Spears", vspacing=1.4)
t.c('k9').rotate_x(90).pos(200,150,70)

cam = dict(
    position=(227.421, -911.244, 192.438),
    focal_point=(217.166, 126.841, 116.242),
    viewup=(0, 0, 1),
    parallel_scale=110,
    clipping_range=(754.990, 1403.38),
)

plt = show(vol, t, size=(700,400), camera=cam, bg='bb')
settings.screenshot_transparent_background = 1
plt.screenshot("logo.png")

