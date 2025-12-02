import numpy as np
import matplotlib.pyplot as plt

# --- Existing cost function ---
x_targ = 10
x_curr_fixed = 5

def total_cost(K):
    #u1 = 0.001*np.cos(x_curr_fixed)
    #u1 = 0.001*x_curr_fixed
    g1, g2 = K
    #LINEAR UPDATE
    #cost1 = (x_targ - (1 + g1) * x_curr_fixed)**2
    #cost2 = ((g2 + (1 + g1) * x_curr_fixed) - 5)**2
    #NON LINEAR UPDATEs
    cost1 = (x_targ - (x_curr_fixed + g1*np.cos(x_curr_fixed)) )**2
    cost2 = (g2 + (x_curr_fixed + g1*np.cos(x_curr_fixed)) - 5)**2
    return cost1 + cost2

# --- Independent minima ---
gain1_indep = 1.0
gain2_indep = 0.0
cost_indep = total_cost([gain1_indep, gain2_indep])

# --- Adam-refined gains ---
def refine_gains(
    initial_K,
    lr=1e-3,
    max_iter=100000,
    tol=1e-6,
    gain_bounds=(0,2),
    eps_cost=1e-6,
    use_log_param=False,
    beta1=0.9,
    beta2=0.999
):
    K = initial_K.copy()
    if use_log_param:
        theta = np.log(K)

    m_adam = np.zeros_like(K)
    v_adam = np.zeros_like(K)
    t = 0

    for iteration in range(max_iter):
        if use_log_param:
            K_curr = np.exp(theta)
        else:
            K_curr = K.copy()

        # Numeric gradient w.r.t. K
        dC_dK = np.zeros_like(K_curr)
        for j in range(len(K_curr)):
            K_plus = K_curr.copy()
            K_minus = K_curr.copy()
            K_plus[j] += eps_cost
            K_minus[j] -= eps_cost
            C_plus = total_cost(K_plus)
            C_minus = total_cost(K_minus)
            dC_dK[j] = (C_plus - C_minus) / (2 * eps_cost)

        # Log-param chain rule
        if use_log_param:
            grad_theta = dC_dK * K_curr
            g = grad_theta
        else:
            g = dC_dK

        # Adam update
        t += 1
        m_adam = beta1 * m_adam + (1 - beta1) * g
        v_adam = beta2 * v_adam + (1 - beta2) * (g ** 2)
        m_hat = m_adam / (1 - beta1**t)
        v_hat = v_adam / (1 - beta2**t)

        if use_log_param:
            theta -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            theta = np.clip(theta, np.log(gain_bounds[0]), np.log(gain_bounds[1]))
        else:
            K -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            K = np.clip(K, gain_bounds[0], gain_bounds[1])

        if np.linalg.norm(g) < tol:
            break

    return np.exp(theta) if use_log_param else K

# Run Adam refinement
initial_K = np.array([0.5, 0.5])
K_refined = refine_gains(initial_K)
cost_refined = total_cost(K_refined)

# --- Grid for total cost surface ---
g1_vals = np.linspace(0, 2, 300)
g2_vals = np.linspace(0, 2, 300)
G1, G2 = np.meshgrid(g1_vals, g2_vals)
C = total_cost([G1, G2])

# Numerical global minimum
min_idx = np.unravel_index(np.argmin(C), C.shape)
g1_global = G1[min_idx]
g2_global = G2[min_idx]
cost_global = C[min_idx]

# --- Print summary ---
print(f"Independent minima: gain1={gain1_indep:.2f}, gain2={gain2_indep:.2f}, cost={cost_indep:.4f}")
print(f"Numerical global minimum: gain1={g1_global:.2f}, gain2={g2_global:.2f}, cost={cost_global:.4f}")
print(f"Adam-refined gains: gain1={K_refined[0]:.4f}, gain2={K_refined[1]:.4f}, cost={cost_refined:.4f}")

# --- 2D contour plot ---
plt.figure(figsize=(8,6))
plt.contourf(G1, G2, C, levels=50, cmap='viridis')
plt.colorbar(label='Total Cost')
plt.scatter(gain1_indep, gain2_indep, color='red', label='Independent minima')
plt.scatter(g1_global, g2_global, color='gray', label='Numerical global minimum')
plt.scatter(K_refined[0], K_refined[1], color='orange', label='Refined gains')
plt.xlabel('Gain1')
plt.ylabel('Gain2')
plt.title('Total Cost Surface with Different Minima')
plt.legend()
plt.show()

# --- 3D surface plot ---
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(G1, G2, C, cmap='viridis', alpha=0.8, edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Total Cost')
ax.scatter(gain1_indep, gain2_indep, cost_indep, color='red', s=50, label='Independent minima')
ax.scatter(g1_global, g2_global, cost_global, color='gray', s=50, label='Numerical global minimum')
ax.scatter(K_refined[0], K_refined[1], cost_refined, color='orange', s=50, label='Refined gains')
ax.set_xlabel('Gain1')
ax.set_ylabel('Gain2')
ax.set_zlabel('Total Cost')
ax.set_title('3D Total Cost Surface with Different Minima')
ax.legend()
plt.show()
