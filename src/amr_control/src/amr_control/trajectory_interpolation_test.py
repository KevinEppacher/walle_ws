import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time

def interpolate_trajectory(ref_traj, total_prediction_distance, N, robot_max_speed, scale_factor=1.1):
    """
    Interpolates or reduces the given trajectory to exactly N points, including orientation (Yaw).
    Only keeps waypoints that are within the total prediction distance.
    
    :param ref_traj: list of waypoints (x, y, yaw) as reference trajectory
    :param total_prediction_distance: total distance for MPC prediction
    :param N: number of prediction points in MPC
    :param robot_max_speed: maximum speed of the robot
    :param scale_factor: scale factor to adjust reference trajectory distance (default 1.1)
    :return: Interpolated or reduced reference trajectory as a list of waypoints with orientation
    """

    # Calculate the distances between points in the current reference trajectory
    ref_points = np.array(ref_traj)[:, :2]
    yaw_angles = np.array(ref_traj)[:, 2]
    distances = np.sqrt(np.sum(np.diff(ref_points, axis=0) ** 2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    # Keep only the waypoints within the total prediction distance
    within_distance_mask = cumulative_distances <= total_prediction_distance
    ref_points_within_distance = ref_points[within_distance_mask]
    yaw_within_distance = yaw_angles[within_distance_mask]
    cumulative_distances_within = cumulative_distances[within_distance_mask]

    # If the number of points is insufficient, add the last point
    if len(ref_points_within_distance) < 2:
        ref_points_within_distance = np.vstack([ref_points_within_distance, ref_points[-1]])
        yaw_within_distance = np.append(yaw_within_distance, yaw_angles[-1])
        cumulative_distances_within = np.append(cumulative_distances_within, total_prediction_distance)

    # Total distance of the reference trajectory
    total_ref_distance = cumulative_distances_within[-1]

    # Scaled distance between points
    scaled_total_distance = total_ref_distance * scale_factor

    # Calculate the time for each prediction step based on the maximum speed
    prediction_distance = total_prediction_distance / N
    T = prediction_distance / robot_max_speed

    # Interpolated positions based on the desired number of points N
    new_distances = np.linspace(0, total_ref_distance, N)

    # Interpolate x and y separately
    interp_x = interp1d(cumulative_distances_within, ref_points_within_distance[:, 0], kind='linear', fill_value="extrapolate")
    interp_y = interp1d(cumulative_distances_within, ref_points_within_distance[:, 1], kind='linear', fill_value="extrapolate")

    new_points_x = interp_x(new_distances)
    new_points_y = interp_y(new_distances)

    # Interpolate yaw angles
    interp_yaw = interp1d(cumulative_distances_within, yaw_within_distance, kind='linear', fill_value="extrapolate")
    new_yaw_angles = interp_yaw(new_distances)

    # Create a new trajectory with x, y, and yaw
    new_trajectory = np.column_stack((new_points_x, new_points_y, new_yaw_angles))

    return new_trajectory, T

# Example trajectory with waypoints (x, y, yaw)
ref_traj = np.array([
    [0, 0, 0],
    [5, 5, np.pi / 2],
    [7, 10, np.pi],
])

# Parameters for interpolation
total_prediction_distance = 15  # Total distance for MPC prediction
N = 10  # Number of prediction points for MPC
current_speed = 0.5  # Current speed of the robot
robot_max_speed = 1.0  # Maximum speed of the robot

# Start computation
start_time = time.time()

# Call interpolation function
new_trajectory, T = interpolate_trajectory(ref_traj, total_prediction_distance, N, current_speed, robot_max_speed)
print(new_trajectory)

# End computation
end_time = time.time()

# Increase font sizes
plt.rcParams.update({'font.size': 40, 'axes.titlesize': 40, 'axes.labelsize': 40, 'legend.fontsize': 14, 'xtick.labelsize': 40, 'ytick.labelsize': 40})

# Computation duration in seconds
calculation_time = end_time - start_time

# Plot original and interpolated trajectories
plt.figure(figsize=(8, 6))

# Original trajectory (blue points and lines) with larger markers
plt.plot(ref_traj[:, 0], ref_traj[:, 1], 'bo-', markersize=10, label='Original Trajectory')

# Interpolated trajectory (red points and dashed lines) with smaller markers on top
plt.plot(new_trajectory[:, 0], new_trajectory[:, 1], 'ro--', markersize=6, zorder=10, label=f'Interpolated Trajectory (N={N})')

# Plot orientation arrows for original trajectory
for i, (x, y, yaw) in enumerate(ref_traj):
    plt.arrow(x, y, np.cos(yaw), np.sin(yaw), head_width=0.1, color='blue')

# Plot orientation arrows for interpolated trajectory
for i, (x, y, yaw) in enumerate(new_trajectory):
    plt.arrow(x, y, np.cos(yaw), np.sin(yaw), head_width=0.1, color='red', zorder=11)

# Display axis labels and title
plt.xlabel('X Position [m]', fontsize=50)
plt.ylabel('Y Position [m]', fontsize=50)
plt.title('Original and Interpolated Trajectories', fontsize=50)

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.show()

# Print computation time in console
print(f"Interpolation computation took {calculation_time:.6f} seconds.")
print(f"Size of new trajectory: {new_trajectory.shape[0]}")
