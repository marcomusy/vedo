"""Add a square button with N possible internal states
to a rendering window that calls an external function"""
from vedo import Plotter, Mesh, dataurl, printc

# Define a function that toggles the transparency of a mesh
#  and changes the button state
def buttonfunc(obj, ename):
    mesh.alpha(1 - mesh.alpha())  # toggle mesh transparency
    bu.switch()                   # change to next status
    printc(bu.status(), box="_", dim=True)

# Load a mesh and set its color to violet
mesh = Mesh(dataurl+"magnolia.vtk").c("violet").flat()

# Create an instance of the Plotter class with axes style-11 enabled
plt = Plotter(axes=11)

# Add a button to the plotter with buttonfunc as the callback function
bu = plt.add_button(
    buttonfunc,
    pos=(0.7, 0.1),   # x,y fraction from bottom left corner
    states=["click to hide", "click to show"],  # text for each state
    c=["w", "w"],     # font color for each state
    bc=["dg", "dv"],  # background color for each state
    font="courier",   # font type
    size=30,          # font size
    bold=True,        # bold font
    italic=False,     # non-italic font style
)

# Show the mesh, docstring, and button in the plot
plt.show(mesh, __doc__).close()
