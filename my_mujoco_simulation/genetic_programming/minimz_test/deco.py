import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Parameters
x_targ = 10
x_curr_fixed = 5
"""new sample is g1*x_curr"""
# Define total cost as the sum of two independent costs
def total_cost(g1, g2, x_curr=x_curr_fixed):
    cost1 = (x_targ - (1+g1) * x_curr)**2
    cost2 = ((g2 + (1+g1)*x_curr) - 5)**2 #+ np.sin(g2)
    # Optional interaction term can be added if desired
    return cost1 + cost2

# Compute independent minima (each gain minimizes its own cost)
gain1_indep = 1 #1, 5/6, 50/55
gain2_indep = 0
cost_indep = total_cost(gain1_indep, gain2_indep)

# Grid to explore the total cost surface
g1_vals = np.linspace(0, 2, 300)
g2_vals = np.linspace(0, 2, 300)
G1, G2 = np.meshgrid(g1_vals, g2_vals)
C = total_cost(G1, G2)

# Find global minimum numerically
min_idx = np.unravel_index(np.argmin(C), C.shape)
g1_global = G1[min_idx]
g2_global = G2[min_idx]
cost_global = C[min_idx]

print(f"Independent minima: gain1={gain1_indep:.2f}, gain2={gain2_indep:.2f}, cost={cost_indep:.2f}")
print(f"Global minimum: gain1={g1_global:.2f}, gain2={g2_global:.2f}, cost={cost_global:.2f}")

# ---- PLOT TOTAL COST SURFACE ----
plt.figure(figsize=(8,6))
plt.contourf(G1, G2, C, levels=50, cmap='viridis')
plt.colorbar(label='Total Cost')
plt.scatter(gain1_indep, gain2_indep, color='red', label='Independent minima')
plt.scatter(g1_global, g2_global, color='gray', label='Global minimum')
plt.xlabel('Gain1')
plt.ylabel('Gain2')
plt.title('Total Cost Surface (Two Original Costs)')
plt.legend()
plt.show()

# ---- 3D SURFACE PLOT ----
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(G1, G2, C, cmap='viridis', alpha=0.8, edgecolor='none')

# Colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Total Cost')

# Mark independent minima
ax.scatter(gain1_indep, gain2_indep, total_cost(gain1_indep, gain2_indep), color='red', s=50, label='Independent minima')

# Mark global minimum
ax.scatter(g1_global, g2_global, total_cost(g1_global, g2_global), color='gray', s=50, label='Global minimum')

# Labels
ax.set_xlabel('Gain1')
ax.set_ylabel('Gain2')
ax.set_zlabel('Total Cost')
ax.set_title('Total Cost Surface (3D)')

ax.legend()
plt.show()
