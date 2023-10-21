"""Modify a spline interactively.
- Drag points with mouse
- Add points by clicking on the line
- Remove them by selecting&pressing DEL
--- PRESS q TO PROCEED ---"""
from vedo import Circle, show

# Create a set of points in space
pts = Circle(res=8).extrude(zshift=0.5).ps(4)

# Visualize the points
plt = show(pts, __doc__, interactive=False, axes=1)

# Add the spline tool using the same points and interact with it
sptool = plt.add_spline_tool(pts, closed=True)

# Add a callback to print the center of mass of the spline
sptool.add_observer(
    "end of interaction", 
    lambda o, e: (
        print(f"Spline changed! CM = {sptool.spline().center_of_mass()}"),
        print(f"\tNumber of points: {sptool.spline().vertices.size}"),
    )
)

# Stay in the loop until the user presses q
plt.interactive()

# Switch off the tool
sptool.off()

# Extract and visualize the resulting spline
sp = sptool.spline().lw(4)
show(sp, "My spline is ready!", interactive=True, resetcam=False).close()
