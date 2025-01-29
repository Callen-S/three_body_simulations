import numpy as np
from scipy.integrate import RK45
import plotly.graph_objects as go

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

def three_body_simulation(masses, initial_positions, initial_velocities, t_span):
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
    solution = RK45(fun=derivatives, t0=t_span[0], y0=initial_conditions, t_bound=t_span[1], rtol=0.0000001)
    t_values = []
    result = []
    while solution.status == 'running':
        solution.step()
        t_values.append(solution.t)
        result.append(solution.y)

    return np.array(result), np.array(t_values)
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

def get_simple_plot(result):

    fig = go.Figure()

    au_to_meters = 1.496e11
    for i in range(3):
        fig.add_trace(go.Scatter3d(
            x=result[:, i * 3] / au_to_meters,
            y=result[:, i * 3 + 1] / au_to_meters,
            z=result[:, i * 3 + 2] / au_to_meters,
            mode='lines',
            name=f'Body {i + 1}'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z'
        ),
        title='Three-Body Problem'
    )
    return fig

def get_animated_plot(result):
    fig = go.Figure()

    au_to_meters = 1.496e11
    num_objects = result.shape[1] // 6

    for i in range(num_objects):
        fig.add_trace(go.Scatter3d(
            x=result[:, i * 3] / au_to_meters,
            y=result[:, i * 3 + 1] / au_to_meters,
            z=result[:, i * 3 + 2] / au_to_meters,
            mode='lines',
            name=f'Body {i + 1}'
        ))

    frames = [go.Frame(data=[go.Scatter3d(
        x=result[:k, i * 3] / au_to_meters,
        y=result[:k, i * 3 + 1] / au_to_meters,
        z=result[:k, i * 3 + 2] / au_to_meters
    ) for i in range(num_objects)]) for k in range(1, len(result))]

    fig.update(frames=frames)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='x', range=[result[:, 0].min() / au_to_meters, result[:, 0].max() / au_to_meters]),
            yaxis=dict(title='y', range=[result[:, 1].min() / au_to_meters, result[:, 1].max() / au_to_meters]),
            zaxis=dict(title='z', range=[result[:, 2].min() / au_to_meters, result[:, 2].max() / au_to_meters])
        ),
        title='Three-Body Problem',
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(label='Play',
                          method='animate',
                          args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)])]
        )]
    )
    return fig

def perturb(masses, positions, velocities):
    """
    Perturb the masses, positions, and velocities by small random amounts.

    Parameters:
        masses (list or np.ndarray): List of masses (M) in kilograms.
        positions (list or np.ndarray): List of positions (x, y, z) of the masses in meters.
        velocities (list or np.ndarray): List of velocities (vx, vy, vz) of the masses in m/s.

    Returns:
        tuple: Perturbed masses, positions, and velocities.
    """
    perturbed_masses = masses * (1 + np.random.uniform(-0.00005, 0.00005, size=len(masses)))
    perturbed_positions = positions * (1 + np.random.uniform(-0.012, 0.012, size=positions.shape))
    perturbed_velocities = velocities * (1 + np.random.uniform(-0.02, 0.02, size=velocities.shape))

    return perturbed_masses, perturbed_positions, perturbed_velocities
def read_initial_conditions(file_path):
    """
    Read initial conditions from a CSV file.

    Parameters:
        file_path (str): Path to the file containing initial conditions.

    Returns:
        tuple: masses, initial_positions, initial_velocities
    """
    import csv

    masses = []
    initial_positions = []
    initial_velocities = []

    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            masses.append([float(row['m1']), float(row['m2']), float(row['m3'])])
            initial_positions.append(np.array([
                [float(row['x1']), float(row['y1']), float(row['z1'])],
                [float(row['x2']), float(row['y2']), float(row['z2'])],
                [float(row['x3']), float(row['y3']), float(row['z3'])]
            ]))
            initial_velocities.append(np.array([
                [float(row['vx1']), float(row['vy1']), float(row['vz1'])],
                [float(row['vx2']), float(row['vy2']), float(row['vz2'])],
                [float(row['vx3']), float(row['vy3']), float(row['vz3'])]
            ]))

    return masses, initial_positions, initial_velocities

class nBodySystem:
    def __init__(self, file_path, system_number=0, t_span=(0, years_to_seconds(200))):
        masses, initial_positions, initial_velocities = read_initial_conditions(file_path)
        self.t_span = t_span
        self.masses = masses[system_number]
        self.initial_positions = initial_positions[system_number]
        self.initial_velocities = initial_velocities[system_number]

    def simulate(self):
        result, time = three_body_simulation(self.masses, self.initial_positions, self.initial_velocities, self.t_span)
        return result, time

    def plot(self, animated=False):
        result, _ = self.simulate()
        if animated:
            fig = get_animated_plot(result)
        else:
            fig = get_simple_plot(result)
        fig.show()