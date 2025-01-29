from simulation_tools import three_body_simulation, years_to_seconds, read_initial_conditions, perturb
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
masses, initial_positions, initial_velocities = read_initial_conditions('initial_conditions.csv')
for i in range(0,100):
    t_span = (0, years_to_seconds(50))

    # Run the unperturbed simulation
    unperturbed_result, time = three_body_simulation(masses[i], initial_positions[i], initial_velocities[i], t_span)

    # Initialize an array to store the errors
    errors = []

    # Run 50 perturbed simulations and calculate the error
    for j in range(10):
        p_mass, p_initial_positions, p_initial_velocities = perturb(masses[i], initial_positions[i], initial_velocities[i])
        perturbed_result, perturbed_time = three_body_simulation(p_mass, p_initial_positions, p_initial_velocities, t_span)
        num_objects = len(masses[i])
        errors_per_body = []

        for k in range(num_objects):
            interped_x = np.interp(time, perturbed_time, perturbed_result[:, k * 3])
            interped_y = np.interp(time, perturbed_time, perturbed_result[:, k * 3 + 1])
            interped_z = np.interp(time, perturbed_time, perturbed_result[:, k * 3 + 2])

            error_x = np.abs(unperturbed_result[:, k * 3] - interped_x)
            error_y = np.abs(unperturbed_result[:, k * 3 + 1] - interped_y)
            error_z = np.abs(unperturbed_result[:, k * 3 + 2] - interped_z)

            error = np.sqrt(error_x**2 + error_y**2 + error_z**2)
            mag = np.sqrt(unperturbed_result[:, k * 3]**2 + unperturbed_result[:, k * 3 + 1]**2 + unperturbed_result[:, k * 3 + 2]**2)
            errors_per_body.append(error/mag * 100)

        errors.append(np.mean(errors_per_body, axis=0))


# Calculate the average error over all simulations, ignoring NaN values
average_error = np.mean(errors, axis=0)

# Plot the average error over time


plt.plot(time / years_to_seconds(1), average_error)
plt.xlabel('Normalized Time')
plt.ylabel('Average Error (%)')
plt.title('Time in Years vs. % Distance Errror')
plt.show()
