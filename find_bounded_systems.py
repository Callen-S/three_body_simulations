import matplotlib
matplotlib.use('TkAgg')
import math
import pandas as pd
import numpy as np
import random
from simulation_tools import years_to_seconds, three_body_simulation, is_problem_bounded, get_simple_plot


def generate_initial_conditions():
    """
    Generate initial positions and velocities for a three-body problem.

    Returns:
        tuple: (masses, initial_positions, initial_velocities)
    """
    masses = [2e30 * random.uniform(0.5, 1.5), 2e30 * random.uniform(0.5, 1.5),
              2e30 * random.uniform(0.5, 1.5)]  # Example masses

    au_to_meters = 1.496e11
    R = au_to_meters * random.uniform(20, 200)
    initial_positions = np.array([
        [random.uniform(-R / 4, R / 4) + R, random.uniform(-R / 4, R / 4), random.uniform(-R / 4, R / 4)],
        [random.uniform(-R / 4, R / 4) + R / 2, random.uniform(-R / 4, R / 4) + R * math.sqrt(3) / 2,
         random.uniform(-R / 4, R / 4)],
        [random.uniform(-R / 4, R / 4) + R / 2, random.uniform(-R / 4, R / 4) - R * math.sqrt(3) / 2,
         random.uniform(-R / 4, R / 4)],
    ])
    scale = 1000 * random.uniform(0.5, 1.5)
    # Generate initial velocities such that the sum of the velocities is on average around 200 m/s
    initial_velocities = np.array([
        [random.uniform(-25, 25) * scale - 25 * scale, random.uniform(-25, 25) * scale, random.uniform(-25, 25)],
        [random.uniform(-50, 25) * scale - 10 * scale, random.uniform(-25, 25) * scale - 10 * scale,
         random.uniform(-25, 25) + 10],
        [random.uniform(-25, 25) * scale - 10 * scale, random.uniform(-25, 25) * scale + 10 * scale,
         random.uniform(-25, 25) + 10],
    ])

    return masses, initial_positions, initial_velocities


# Example usage
bounded_solutions = []
while len(bounded_solutions) < 100:
    t_span = (0, years_to_seconds(500))  # Time span for the simulation
    t_eval = np.linspace(0, t_span, 10000000)  # Time points to evaluate
    masses, initial_positions, initial_velocities = generate_initial_conditions()
    result = three_body_simulation(masses, initial_positions, initial_velocities, t_span)
    animate = False
    # Plotting the result in 3D

    if is_problem_bounded(result, masses):

        bounded_solutions.append(np.hstack((masses, initial_positions.flatten(), initial_velocities.flatten())))
        fig = get_simple_plot(result)

        fig.show()

df = pd.DataFrame(bounded_solutions,
                  columns=["m1", "m2", "m3", "x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "vx1", "vy1", "vz1", "vx2", "vy2",
                           "vz2", "vx3", "vy3", "vz3"])
df.to_csv("initial_conditions.csv", index=False)