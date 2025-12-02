import numpy as np
import matplotlib.pyplot as plt

# Parameters
x_targ = 10
gain1 = 1
gain2 = 1
x_curr = 5
step_size = 1e-4
x_update_factor = 0.01
max_iters = 10000

# Buffers
gain1_buffer = []
gain2_buffer = []
cost_buffer = []
cost1_buffer = []
cost2_buffer = []
x_buffer = []

# Gradient descent loop
for i in range(max_iters):
    # Costs
    cost1 = (x_targ - gain1 * x_curr)**2
    cost2 = (x_targ - gain2 * x_curr)**2
    cost = cost1 + cost2
    
    # Save for plotting
    gain1_buffer.append(gain1)
    gain2_buffer.append(gain2)
    cost1_buffer.append(cost1)
    cost2_buffer.append(cost2)
    cost_buffer.append(cost)
    x_buffer.append(x_curr)
    
    # Gradient descent updates (each gain independently)
    gain1_new = gain1 - step_size * (-2 * x_curr * (x_targ - gain1 * x_curr))
    gain2_new = gain2 - step_size * (-2 * x_curr * (x_targ - gain2 * x_curr))
    
    # Update x_curr based on the gains (simple linear combination)
    x_curr = x_curr + x_update_factor * (gain1_new + gain2_new)
    
    # Stopping condition
    if np.linalg.norm(gain1_new - gain1) < 1e-6 and np.linalg.norm(gain2_new - gain2) < 1e-6:
        break
    
    gain1 = gain1_new
    gain2 = gain2_new

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

# ---- PLOT x_curr ----
plt.figure(figsize=(10,4))
plt.plot(x_buffer)
plt.xlabel("Iteration")
plt.ylabel("x_curr")
plt.title("x_curr Evolution")
plt.grid(True)
plt.show()
