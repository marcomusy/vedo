 # @ Author: Giovanni Dalmasso
 # @ Create Time: 09-02-2024 17:02:07
 # @ Modified by: M. Musy
import numpy as np
from vedo import Arrow2D, Latex, Line, Mesh, Plotter, Text3D, dataurl


# Define the forces and initial angle
FB = 800    # Newton
FA = 750    # Newton
theta = 48  # Initial angle in degrees

# Calculate the initial components of FA and Fx
theta_rad = np.radians(theta)
FA_x = FA * np.sin(theta_rad)
FA_y = FA * np.cos(theta_rad)
Fx = FB * np.cos(np.radians(30)) + FA * np.sin(theta_rad)

# Initialize the text object for Fx with its initial value
arrow_Fx = Arrow2D([0, 0, 0], (Fx, 0, 0), c="green4")
label_Fx = Text3D(f"F_x : {Fx:.0f}~N", pos=(Fx, 0, 0), c="green4", s=80)

# Create arrows for forces
arrow_FA = Arrow2D([0, 0, 0], (FA_x, FA_y, 0), c="red4")
label_FA = Text3D(f"F_A : {FA:.0f}~N", (FA_x, FA_y, 0), c="red4", s=80)

arrow_FB = Arrow2D(
    [0, 0, 0],
    [FB * np.cos(np.radians(30)), -FB * np.sin(np.radians(30)), 0],
    c="blue4",
)
arrow_FB.name = "ArrowFixed"

label_FB = Text3D(
    f"F_B : {FB:.0f}~N",
    pos=(FB * np.cos(np.radians(30)), -FB * np.sin(np.radians(30)), 0),
    c="blue4",
    s=80,
)
label_FB.name = "LabelFixed"

# Vertical line to represent the reference direction for theta
vertical_line = Line([0, 0, 0], [0, FA_y, 0], c="black", lw=10)

theta_txt = Latex(r"\vartheta", s=400).pos([0, FA_y / 3, 0])
deg = Latex(r"30^\circ", s=400).pos([400, -300])

gio = Text3D("Giovanni D.", s=50, c="blue4", italic=True).pos(-600, -300, 0)

car = Mesh(dataurl + "porsche.ply").c("k8").lighting("metallic").phong()
car.scale(50).pos(-300, 0, 0).rotate(90)


def update_scene(widget, event):
    """Update the forces FA and Fx based on the angle theta."""

    # Recalculate components of FA and Fx
    theta = np.radians(widget.value)
    FA_x, FA_y = FA * np.sin(theta), FA * np.cos(theta)
    Fx = FB * np.cos(np.radians(30)) + FA * np.sin(theta)

    arrow_FA = Arrow2D([0, 0, 0], (FA_x, FA_y, 0), c="red4").z(0.1)
    arrow_Fx = Arrow2D([0, 0, 0], (Fx, 0, 0), c="green4")

    label_FA.pos(FA_x, FA_y, 0.1).text(f"F_A : {FA:.0f}~N")
    label_Fx.pos(Fx, 0, 0).text(f"F_x : {Fx:.0f}~N")

    plt.remove("Arrow2D").add([arrow_FA, arrow_Fx])


# Create the plotter and a slider to adjust the angle theta
plt = Plotter(bg2="lightblue", size=(1200, 800))
plt.add(car, vertical_line, theta_txt, deg, gio)
plt.add(arrow_FA, arrow_FB, arrow_Fx, label_FA, label_FB, label_Fx)
plt.add_slider(update_scene, 0, 90, value=theta, title="Theta Angle")
plt.show(zoom=1.3).close()
