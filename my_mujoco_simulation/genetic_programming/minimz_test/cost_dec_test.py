import numpy as np
import matplotlib.pyplot as plt

x_targ = 10
gain = 0
x_curr = 5
step_size = 1e-4   # choose small enough to avoid divergence

gain_buffer = []
cost_buffer = []

max_iters = 1e6
i = 0

while True:
    # Compute cost
    cost = (x_targ - (1+gain) * x_curr)**2
    cost_buffer.append(cost)
    # Gradient descent update
    new_gain = gain - step_size * (-2 * x_curr * (x_targ - (1+gain) * x_curr))
    gain_buffer.append(new_gain)
    # stopping condition
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
