import numpy as np
import matplotlib.pyplot as plt

# Parameters
x_targ = 10
gain1 = 1
gain2 = 0.5
x_curr = 5
step_size = 1e-4   # small enough to avoid divergence
x_update_factor = 0.01  # factor to update x_curr

# Buffers for plotting
gain1_buffer = []
gain2_buffer = []
cost_buffer = []
cost1_buffer = []
cost2_buffer = []
x_buffer = []

max_iters = int(1e6)
i = 0

while True:
    # Compute costs
    cost1 = (x_targ - gain1 * x_curr)**2
    cost2 = (x_targ - gain2 * x_curr)**2
    cost = 0.5*cost1 + 0.5*cost2
    
    # Save for plotting
    cost1_buffer.append(cost1)
    cost2_buffer.append(cost2)
    cost_buffer.append(cost)
    x_buffer.append(x_curr)
    
    # Gradient descent update for gains
    new_gain1 = gain1 - step_size * (-2 * x_curr * (x_targ - gain1 * x_curr))
    new_gain2 = gain2 - step_size * (-2 * x_curr * (x_targ - gain2 * x_curr))
    
    gain1_buffer.append(new_gain1)
    gain2_buffer.append(new_gain2)
    
    
    # Stopping condition (when both gains change very little)
    if np.linalg.norm(new_gain1 - gain1) < 1e-6 and np.linalg.norm(new_gain2 - gain2) < 1e-6:
        break
    
    gain1 = new_gain1
    gain2 = new_gain2
    
    i += 1
    if i >= max_iters:
        print("Stopped at max iterations")
        break

# ---- PLOT GAINS ----
plt.figure(figsize=(10,4))
plt.plot(gain1_buffer, label="Gain 1")
plt.plot(gain2_buffer, label="Gain 2")
plt.xlabel("Iteration")
plt.ylabel("Gain")
plt.title("Gain Evolution")
plt.legend()
plt.grid(True)
plt.show()

# ---- PLOT COSTS ----
plt.figure(figsize=(10,4))
plt.plot(cost_buffer, label="Total Cost")
plt.plot(cost1_buffer, label="Cost 1", linestyle='--')
plt.plot(cost2_buffer, label="Cost 2", linestyle=':')
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Evolution")
plt.legend()
plt.grid(True)
plt.show()

# Fixed current value
x_curr = 5
x_targ = 10

# Range of gains
gain_vals = np.linspace(0, 5, 900)

# Compute costs
cost1_vals = (x_targ - gain_vals * x_curr)**2
cost2_vals = (x_targ - gain_vals * x_curr)**2  # same shape, for demonstration
total_cost_vals = cost1_vals + cost2_vals

# Plot cost functions
plt.figure(figsize=(10,5))
plt.plot(gain_vals, cost1_vals, label="Cost 1")
plt.plot(gain_vals, cost2_vals, label="Cost 2", linestyle='--')
plt.plot(gain_vals, total_cost_vals, label="Total Cost", linestyle=':')
plt.xlabel("Gain")
plt.ylabel("Cost")
plt.title("Shape of Cost Functions")
plt.legend()
plt.grid(True)
plt.show()