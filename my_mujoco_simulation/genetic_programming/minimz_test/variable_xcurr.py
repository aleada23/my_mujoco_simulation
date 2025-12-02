import numpy as np
import matplotlib.pyplot as plt

x_targ = 10
gain = 1
x_curr = 5
step_size = 1e-4   # small enough to avoid divergence
x_update_factor = 0.01  # factor to update x_curr

gain_buffer = []
cost_buffer = []
x_buffer = []

max_iters = int(1e6)
i = 0

while True:
    # Compute cost
    cost = (x_targ - gain * (x_curr + gain * x_update_factor))**2
    cost_buffer.append(cost)
    x_buffer.append(x_curr)
    
    # Gradient descent update for gain
    new_gain = gain - step_size * (-2 * (x_curr + gain * x_update_factor) * (x_targ - gain * (x_curr + gain * x_update_factor)))
    gain_buffer.append(new_gain)
    
    # Update x_curr
    #x_curr = x_curr + new_gain * x_update_factor
    
    # Stopping condition
    if np.linalg.norm(new_gain - gain) < 1e-6:
        break
    
    gain = new_gain
    i += 1
    if i >= max_iters:
        print("Stopped at max iterations")
        break

# ---- PLOT GAIN ----
plt.figure(figsize=(10,4))
plt.plot(gain_buffer)
plt.xlabel("Iteration")
plt.ylabel("Gain")
plt.title("Gain Evolution")
plt.grid(True)
plt.show()

# ---- PLOT COST ----
plt.figure(figsize=(10,4))
plt.plot(cost_buffer)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Evolution")
plt.grid(True)
plt.show()