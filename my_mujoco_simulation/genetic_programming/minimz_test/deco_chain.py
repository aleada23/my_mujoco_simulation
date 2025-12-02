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

# --- Independent minima ---u1
gain1_indep = 1.0
gain2_indep = 0.0
cost_indep = total_cost([gain1_indep, gain2_indep])

# --- Adam-refined gains ---
def refine_gains_structured(initial_K, lr=1e-4, max_iter=10000000, tol=1e-6, alpha=1.0, gain_bounds=(0,2)):
    K = initial_K.copy()
    m_adam = np.zeros_like(K)
    v_adam = np.zeros_like(K)
    t = 0
    eps_cost = 1e-6

    for iteration in range(max_iter):
        # --- Step 1: compute dq_total from scaled tasks ---
        g1, g2 = K
        u1 = np.array([np.cos(x_curr_fixed)])  # unscaled task velocity
        scaled_tasks = [(g1*u1, 1.0), (g2*u1, 1.0)]  # J=1 for both
        dq_total = np.array(scaled_tasks[0][0])  # SOT output
        #dq_total = dq_total.flatten()  # shape (1,)

        # --- Step 2: numeric dC/dq ---
        dC_ddq = np.zeros(1)
        dq_plus = np.zeros(1)
        dq_minus = np.zeros(1)
        for j in range(len(dq_total)):
            #dq_plus = dq_total.copy()
            #dq_minus = dq_total.copy()
            dq_plus[j] = dq_total + eps_cost
            dq_minus[j] = dq_total -  eps_cost
            # temporary "post-motion" gains
            C_plus = total_cost(K + dq_plus)
            C_minus = total_cost(K + dq_minus)
            dC_ddq[j] = (C_plus - C_minus) / (2*eps_cost)

        # --- Step 3: Build A-matrix ---
        n = 1  # dq_total dimension (1D task)
        m = len(K)
        A = np.zeros((n, m))
        for j in range(m):
            # Perturb only gain j
            scaled_tasks_basis = []
            for i in range(m):
                u = np.array([1.0]) if i == j else np.array([0.0])
                J = 1.0
                scaled_tasks_basis.append((u, J))
            dq_col = sum(u for u,_ in scaled_tasks_basis).flatten()
            A[:, j] = dq_col

        # --- Step 4: Chain rule: grad_K = dC/dq * A ---
        grad_K = dC_ddq @ A  # shape (m,)
        # --- Adam update ---
        t += 1
        beta1, beta2 = 0.9, 0.9
        eps_adam = 1e-9
        m_adam = beta1*m_adam + (1-beta1)*grad_K
        v_adam = beta2*v_adam + (1-beta2)*(grad_K**2)
        m_hat = m_adam/(1-beta1**t)
        v_hat = v_adam/(1-beta2**t)
        K -= lr * m_hat / (np.sqrt(v_hat)+eps_adam)
        K = np.clip(K, gain_bounds[0], gain_bounds[1])

        if np.linalg.norm(grad_K) < tol:
            break

    return K

# Run Adam refinement
initial_K = np.array([1.0, 1.0])
K_refined = refine_gains_structured(initial_K)
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

# --- Compute starting cost ---
cost_start = total_cost(initial_K)

# --- Print summary including start ---
print(f"Start of descent: gain1={initial_K[0]:.4f}, gain2={initial_K[1]:.4f}, cost={cost_start:.4f}")
print(f"Numerical global minimum: gain1={g1_global:.4f}, gain2={g2_global:.4f}, cost={cost_global:.4f}")
print(f"Refined gains: gain1={K_refined[0]:.4f}, gain2={K_refined[1]:.4f}, cost={cost_refined:.4f}")


# --- Compute starting cost ---
cost_start = total_cost(initial_K)

# --- 2D contour plot ---
plt.figure(figsize=(8,6))
plt.contourf(G1, G2, C, levels=50, cmap='viridis')
plt.colorbar(label='Total Cost')
plt.scatter(g1_global, g2_global, color='gray', label='Numerical global minimum')
plt.scatter(K_refined[0], K_refined[1], color='orange', label='Refined gains')
plt.scatter(initial_K[0], initial_K[1], color='blue', label='Start of descent', marker='x', s=80)
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
ax.scatter(g1_global, g2_global, cost_global, color='gray', s=50, label='Numerical global minimum')
ax.scatter(K_refined[0], K_refined[1], cost_refined, color='orange', s=50, label='Refined gains')
ax.scatter(initial_K[0], initial_K[1], cost_start, color='blue', s=80, label='Start of descent', marker='x')
ax.set_xlabel('Gain1')
ax.set_ylabel('Gain2')
ax.set_zlabel('Total Cost')
ax.set_title('3D Total Cost Surface with Different Minima')
ax.legend()
plt.show()
