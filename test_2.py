import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def dvdt(v, t, a_max, v_max, green_light, position, distance_from_lead):
    if green_light or position > -25:
        return a_max * (1 - (v / v_max))
    else:
        return -a_max * (1 - (v / v_max))  # Decelerate with the same rate after 30 seconds


def traffic_system(y, t, params):
    """
    Define the system of ODEs representing the motion of the cars.
   
    Parameters:
        y : numpy array
            The current state of the system: [position_1, velocity_1, position_2, velocity_2, ..., position_n, velocity_n]
        t : float
            The current timestep
        params : tuple
            Tuple containing the parameters: (n, L, a_max, v_max, r)
   
    Returns:
        dydt : array_like
            The derivatives of the state variables, y.
    """
    n, L, a_max, v_max, r, green_light = params
    dydt = np.zeros_like(y)
   
    # Unpack the positions and velocities of the cars
    positions = y[::2]      # Every second element starting from index 0
    velocities = y[1::2]    # Every second element starting from index 1
    # Lead car in this case is just the car infront of current car
    distance_from_lead = [1000] + [positions[i]-positions[i+1] for i in range(len(positions)-1)]
    if not green_light:
        car_at_light = np.argmin(positions[positions<0])
        
    else: car_at_light = 0
    # Loop through each car
    for i in range(n):
        # Calculate acceleration based on the distance to the car ahead
        if green_light:
            delta_d = distance_from_lead[i]
            if delta_d <= L:
                # If the distance is at most the desired distance, maintain velocity
                accel = 0
            elif delta_d > L and velocities[i] < v_max:
                # If the distance is greater than the desired distance, accelerate
                # accel = min(dvdt(velocities[i], t, a_max, v_max, positions[i]), (velocities[i-1] - velocities[i]) / r) # sort of works
                accel = dvdt(velocities[i],t,a_max,v_max,positions[i],delta_d,car_at_light)
                """
                The cars should accelerate until they reach the desired distance, then maintain that distance.
                Add code here to compensate for distance.
            #     """
        else:
            delta_d = distance_from_lead[i]
            if delta_d >= L:
                # If the distance is at most the desired distance, maintain velocity
                accel = 0
            elif delta_d < L and velocities[i] > v_max:
                # If the distance is greater than the desired distance, accelerate
                # accel = min(dvdt(velocities[i], t, a_max, v_max, positions[i]), (velocities[i-1] - velocities[i]) / r) # sort of works
                accel = dvdt(velocities[i],t,a_max,v_max,positions[i],delta_d,car_at_light)
                """
                The cars should accelerate until they reach the desired distance, then maintain that distance.
                Add code here to compensate for distance.
            #     """
            elif delta_d > L and velocities[i] > v_max:
                # If the distance is greater than the desired distance, accelerate
                # accel = min(dvdt(velocities[i], t, a_max, v_max, positions[i]), (velocities[i-1] - velocities[i]) / r) # sort of works
                accel = dvdt(velocities[i],t,a_max,v_max,positions[i],delta_d,car_at_light)

           
        # Update velocity and position in state vector
        dydt[2*i] = velocities[i]       # dx/dt = v
        dydt[2*i + 1] = accel           # dv/dt = a
   
    return dydt
# Main simulation function
def run():
    n = 10  # Number of cars
    L = 10  # Desired distance between cars
    D = 2  # Initial distance between cars
    a_max = 2  # Maximum acceleration
    v_max = 15  # Maximum velocity
    r = 2  # Reaction time per car
    t_max = 60  # Maximum time for simulation
    precision = 1000  # Number of points to simulate
    light_duration = 30  # Duration of one traffic light colour

    initial_conditions = np.zeros(2 * n)
    initial_conditions[::2] = np.arange(n) * -D
    initial_conditions[1::2] = 0

    params = [n, L, a_max, v_max, r, True] # Last param is True for green - False for red
    t1 = np.linspace(0, light_duration, precision)
    solution_g = odeint(traffic_system, initial_conditions, t1, args=(params,))
    positions, velocities = solution_g[:, ::2], solution_g[:, 1::2]
    params[-1] = False
    solution_r = odeint(traffic_system, initial_conditions, t1, args=(params,))
    positions = np.r_[positions,solution_r[:,::2]]
    velocities = np.r_[velocities,solution_r[:,1::2]]
    t1 = np.linspace(0,2*light_duration,precision*2)


    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    for car in range(n):
        axs[0].plot(t1, positions[:, car], label=f'Car {car+1}')
        axs[1].plot(t1, velocities[:, car], label=f'Car {car+1}')

    axs[0].set_title('Car Positions Over Time')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Position')
    axs[0].legend()

    axs[1].set_title('Car Velocities Over Time')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Velocity')
    axs[1].legend()

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    run()

