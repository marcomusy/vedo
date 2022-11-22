from vedo import VedoLogo, settings

settings.use_parallel_projection = 1
settings.screenshot_transparent_background = 0

vl = VedoLogo(frame=False, c='k')
vl.show(size=(340*3,115*3), zoom="tight")
# vl.screenshot("logo_vedo_simple.png")
