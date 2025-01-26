import matplotlib
matplotlib.use('TkAgg')
import math
from scipy.integrate import RK45
import pandas as pd
import numpy as np

import random


def generate_initial_conditions():
    """
    Generate initial positions and velocities for a three-body problem.

    Returns:
        tuple: (masses, initial_positions, initial_velocities)
    """
    masses = [2e30*random.uniform(0.5, 1.5), 2e30*random.uniform(0.5, 1.5), 2e30*random.uniform(0.5, 1.5)]  # Example masses


    au_to_meters = 1.496e11
    R = au_to_meters*random.uniform(20,200)
    initial_positions = np.array([
        [random.uniform(-R/4, R/4) + R, random.uniform(-R/4, R/4),random.uniform(-R/4, R/4)],
        [random.uniform(-R/4, R/4) + R/2, random.uniform(-R/4, R/4) + R*math.sqrt(3)/2, random.uniform(-R/4, R/4)],
        [random.uniform(-R/4, R/4) + R/2, random.uniform(-R/4, R/4) - R*math.sqrt(3)/2, random.uniform(-R/4, R/4)],
    ])
    scale = 1000*random.uniform(0.5, 1.5)
    # Generate initial velocities such that the sum of the velocities is on average around 200 m/s
    initial_velocities = np.array([
        [random.uniform(-25, 25)*scale -25*scale, random.uniform(-25, 25)*scale, random.uniform(-25, 25)],
        [random.uniform(-50, 25)*scale -10*scale, random.uniform(-25, 25)*scale -10*scale, random.uniform(-25, 25) +10],
        [random.uniform(-25, 25)*scale -10*scale, random.uniform(-25, 25)*scale +10*scale, random.uniform(-25, 25) +10],
    ])

    return masses, initial_positions, initial_velocities


def years_to_seconds(years):
    """
    Convert years to seconds.

    Parameters:
    years (float): Number of years to convert.

    Returns:
    float: Equivalent number of seconds.
    """
    days_per_year = 365.25  # Average including leap years
    hours_per_day = 24
    minutes_per_hour = 60
    seconds_per_minute = 60

    seconds = years * days_per_year * hours_per_day * minutes_per_hour * seconds_per_minute
    return seconds

def gravitational_acceleration(masses, positions):
    """
    Calculate the gravitational acceleration for all objects in 3D space due to their mutual interactions.

    Parameters:
        masses (list or np.ndarray): List of masses (M) in kilograms.
        positions (list or np.ndarray): List of positions (x, y, z) of the masses in meters.

    Returns:
        np.ndarray: Array of gravitational acceleration vectors for each object in m/s^2.
    """
    G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
    positions = np.array(positions, dtype=float)
    n = len(masses)
    accelerations = np.zeros((n, 3), dtype=float)

    for i in range(n):
        for j in range(n):
            if i != j:
                r_vector = positions[j] - positions[i]
                r_magnitude = np.linalg.norm(r_vector)
                if r_magnitude != 0:  # Avoid division by zero
                    r_unit = r_vector / r_magnitude
                    accelerations[i] += G * masses[j] / r_magnitude**2 * r_unit

    return accelerations

def three_body_simulation(masses, initial_positions, initial_velocities, t_span, t_eval):
    """
    Simulate the three-body problem using RK45 integrator.

    Parameters:
        masses (list or np.ndarray): List of masses (M) in kilograms.
        initial_positions (list or np.ndarray): Initial positions (x, y, z) of the masses in meters.
        initial_velocities (list or np.ndarray): Initial velocities (vx, vy, vz) of the masses in m/s.
        t_span (tuple): The time interval for the simulation (start, end).
        t_eval (np.ndarray): Time points at which to store the computed solution.

    Returns:
        np.ndarray: Array of positions and velocities at each time point.
    """
    def derivatives(t, y):
        num_objects = len(masses)
        positions = y[:3*num_objects].reshape((num_objects, 3))
        velocities = y[3*num_objects:].reshape((num_objects, 3))
        accelerations = gravitational_acceleration(masses, positions)
        return np.concatenate((velocities.flatten(), accelerations.flatten()))

    initial_conditions = np.concatenate((initial_positions.flatten(), initial_velocities.flatten()))
    solution = RK45(derivatives, t_span[0], initial_conditions, t_span[1])

    result = []
    while solution.status == 'running':
        solution.step()
        result.append(solution.y)

    return np.array(result)
def is_problem_bounded(result, masses):
    """
    Check if the problem is still bounded by verifying if at any point in the last 1% of the simulation
    the acceleration is equal to 1% of the velocity for each axis.

    Parameters:
        result (np.ndarray): Array of positions and velocities at each time point.
        masses (list or np.ndarray): List of masses (M) in kilograms.

    Returns:
        bool: True if the problem is still bounded, False otherwise.
    """
    last_1_percent_index = int(0.99 * len(result))
    last_1_percent_data = result[last_1_percent_index:]

    for data in last_1_percent_data:
        num_objects = len(masses)
        positions = data[:3*num_objects].reshape((num_objects, 3))
        velocities = data[3*num_objects:].reshape((num_objects, 3))
        accelerations = gravitational_acceleration(masses, positions)

        for i in range(num_objects):
            for axis in range(3):
                if not (abs(accelerations[i][axis]) >= 0.000000001/100 * abs(velocities[i][axis])):
                    return False
    return True
# Example usage
i = 0
bounded_solutions = []
while(len(bounded_solutions) < 100):
    t_span = (0, years_to_seconds(500))  # Time span for the simulation
    t_eval = np.linspace(0, t_span, 10000000)  # Time points to evaluate
    masses, initial_positions, initial_velocities = generate_initial_conditions()
    result = three_body_simulation(masses, initial_positions, initial_velocities, t_span, t_eval)
    animate = False
    # Plotting the result in 3D
    if i%50 == 0:
        print(i)
    i+=1
    if is_problem_bounded(result,masses):
        print("Bounded")
        print(initial_positions)
        print(initial_velocities)
        bounded_solutions.append(np.hstack((initial_positions.flatten(), initial_velocities.flatten())))
        import plotly.graph_objects as go

        fig = go.Figure()

        au_to_meters = 1.496e11
        for i in range(3):
            fig.add_trace(go.Scatter3d(
                x=result[:, i*3] / au_to_meters,
                y=result[:, i*3+1] / au_to_meters,
                z=result[:, i*3+2] / au_to_meters,
                mode='lines',
                name=f'Body {i+1}'
            ))

        fig.update_layout(
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z'
            ),
            title='Three-Body Problem'
        )

        fig.show()



df = pd.DataFrame(bounded_solutions, columns=["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "vx1", "vy1", "vz1", "vx2", "vy2", "vz2", "vx3", "vy3", "vz3"])
df.to_csv("initial_conditions.csv", index=False)
bounded_solutions = []