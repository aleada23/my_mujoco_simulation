import numpy as np
import matplotlib.pyplot as plt

# ------------------------
#   COST + MODEL
# ------------------------
x_target = 10
x0 = 2

# nonlinear secondary directions
n1 = lambda g1, g2: np.cos(g1)
n2 = lambda g1, g2: np.sin(g2)

alpha = 0.5  # step size in q-space

def forward(K):
    g1, g2 = K
    dq = 1.0 + g1*n1(g1,g2) + g2*n2(g1,g2)    
    x_new = x0 + alpha*dq
    manip = g2*np.sqrt(x_new)
    return x_new, manip

def total_cost(K):
    x_new, manip = forward(K)
    return (x_target - x_new)**2 


# ------------------------
#   NUMERICAL GLOBAL MINIMUM
# ------------------------
g1_vals = np.linspace(0, 5, 300)
g2_vals = np.linspace(0, 5, 300)
G1, G2 = np.meshgrid(g1_vals, g2_vals)
C = total_cost([G1, G2])

min_idx = np.unravel_index(np.argmin(C), C.shape)
g1_global = G1[min_idx]
g2_global = G2[min_idx]
cost_global = C[min_idx]


# ------------------------
#   NAIVE GRADIENT DESCENT
# ------------------------
def naive_GD(initial_K, lr=0.05, iters=10000):
    K = initial_K.copy()
    for _ in range(iters):
        grad = np.zeros(2)
        eps = 1e-5
        for j in range(2):
            Kp = K.copy(); Km = K.copy()
            Kp[j] += eps
            Km[j] -= eps
            grad[j] = (total_cost(Kp) - total_cost(Km)) / (2*eps)
        
        K -= lr * grad
        K = np.clip(K, 0, 5)  # <-- clip g1 and g2 between 0 and 5

    return K, total_cost(K)



# ------------------------
#   CHAIN-RULE GD (CORRECT)
# ------------------------
def chain_GD(initial_K, lr=0.001, iters=100000):
    K = initial_K.copy()
    for _ in range(iters):
        g1, g2 = K

        # STEP 1: compute dq
        dq = 1.0 + g1*n1(g1,g2) + g2*n2(g1,g2)

        # STEP 2: compute dC/dq
        x_new = x0 + alpha*dq
        dC_ddq = -2*(x_target - x_new)*alpha

        # STEP 3: dq/dg1 and dq/dg2
        ddq_dg1 = n1(g1,g2) + g1*(-np.sin(g1))
        ddq_dg2 = n2(g1,g2) + g2*(np.cos(g2))

        grad = np.array([
            dC_ddq * ddq_dg1,
            dC_ddq * ddq_dg2
        ])

        K -= lr * grad
        K = np.clip(K, 0, 5)  # <-- clip g1 and g2 between 0 and 5

    return K, total_cost(K)



# ------------------------
#   RUN BOTH OPTIMIZERS
# ------------------------
initial_K = np.array([4.0, 4.0])
cost_start = total_cost(initial_K)

K_naive, cost_naive = naive_GD(initial_K)
K_chain, cost_chain = chain_GD(initial_K)


# ------------------------
#   PRINT RESULTS
# ------------------------
print("\n--- RESULTS ---")
print(f"Start:         K=({initial_K[0]:.3f}, {initial_K[1]:.3f}), cost={cost_start:.4f}")
print(f"True minimum:  K=({g1_global:.3f}, {g2_global:.3f}), cost={cost_global:.4f}")
print(f"Naive GD:      K=({K_naive[0]:.3f}, {K_naive[1]:.3f}), cost={cost_naive:.4f}")
print(f"Chain-rule GD: K=({K_chain[0]:.3f}, {K_chain[1]:.3f}), cost={cost_chain:.4f}")
print("------------------------\n")


# ------------------------
#   PLOTS
# ------------------------
plt.figure(figsize=(8,6))
plt.contourf(G1, G2, C, levels=50, cmap='viridis')
plt.colorbar(label="Total Cost")

plt.scatter(initial_K[0], initial_K[1], color='blue', marker='x', s=80, label='Start')
plt.scatter(g1_global, g2_global, color='white', edgecolor='black', label='True minimum')
plt.scatter(K_naive[0], K_naive[1], color='red', label='Naive GD result')
plt.scatter(K_chain[0], K_chain[1], color='orange', label='Chain-rule GD result')

plt.xlabel("g1")
plt.ylabel("g2")
plt.title("Cost Landscape + Optimization Results")
plt.legend()
plt.show()


# 3D PLOT
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(G1, G2, C, cmap='viridis', alpha=0.85, edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5)

ax.scatter(initial_K[0], initial_K[1], cost_start, color='blue', marker='x', s=80)
ax.scatter(g1_global, g2_global, cost_global, color='white', edgecolor='black', s=60)
ax.scatter(K_naive[0], K_naive[1], cost_naive, color='red', s=60)
ax.scatter(K_chain[0], K_chain[1], cost_chain, color='orange', s=60)

ax.set_xlabel("g1")
ax.set_ylabel("g2")
ax.set_zlabel("Cost")
ax.set_title("3D Cost Landscape")
plt.show()
