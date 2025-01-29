from simulation_tools import three_body_simulation, years_to_seconds, get_simple_plot, read_initial_conditions
import numpy as np

file_path = 'initial_conditions.csv'
masses, initial_positions, initial_velocities = read_initial_conditions(file_path)

for i in range(100):
    t_span = (0, years_to_seconds(200))
    result, time = three_body_simulation(masses[i],initial_positions[i],initial_velocities[i], t_span)
    fig = get_simple_plot(result)
    fig.show()


