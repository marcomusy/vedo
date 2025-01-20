"""Create an interactive plot that allows the user to control a parameter using a slider, 
The plot shows the solution to a system of equations for y given x and a constant C. 
The user can change the value of C using a slider, and the plot will update the y-range."""
import numpy as np
from scipy.optimize import fsolve
from vedo import Plotter, settings
from vedo.pyplot import plot


# Initial values
C_init = 0
xdata = np.linspace(-3, 3, 50)


# Function to solve for y given x and C (from your first script)
def solve_for_y(x, C):
    y_vals = []
    for sign in [1, -1]:  # Solve for positive and negative y

        def equation(y):
            return 0.5 * y**2 + np.log(np.abs(y)) - 0.5 * x**2 - C

        y_initial_guess = sign * np.exp(-0.5 * x**2 - C)

        root = fsolve(equation, y_initial_guess)[0]
        if equation(root) < 1e-5:  # Only accept the root if it's a valid solution
            y_vals.append(root)
    return y_vals


# Generate the y values for plotting (positive and negative y)
def generate_y_values(x_values, C):
    y_positive = []
    y_negative = []
    for x in x_values:
        y_vals = solve_for_y(x, C)
        if len(y_vals) > 0:
            y_positive.append(max(y_vals))  # Choose the largest root as positive
            y_negative.append(min(y_vals))  # Choose the smallest root as negative
        else:
            y_positive.append(np.nan)       # Use NaN for missing values
            y_negative.append(np.nan)
    return y_positive, y_negative


# Function to update the plot when the slider changes
def update_plot(widget=None, event=""):
    C_value = C_init if widget is None else widget.value
    y_positive, y_negative = generate_y_values(xdata, C_value)
    m = max(max(y_positive), abs(min(y_negative)))
    p  = plot(xdata, y_positive, c='red5',  lw=4, ylim=(-m, m))
    p += plot(xdata, y_negative, c='blue5', lw=4, like=p)
    plt.remove("PlotXY").add(p)


# Create Plotter and the slider to control the value of C
settings.default_font = "Brachium"

plt = Plotter(size=(1200, 760), title="Exercise")
slider = plt.add_slider(update_plot, -10.0, 10.0, value=C_init, title="C value", c="green3")
update_plot()  # Initial plot

plt.show(__doc__, mode="2d", zoom=1.35)
