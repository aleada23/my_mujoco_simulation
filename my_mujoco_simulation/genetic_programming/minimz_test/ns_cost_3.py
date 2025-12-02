import numpy as np

# ------------------------
# PROBLEM SETUP (5D gains example)
# ------------------------
x_target = 10
x0 = 2
alpha = 0.5
d = 5  # number of gains
np.random.seed(0)

# nonlinear coupling terms
def forward(K):
    # K is 5D: [g1, g2, g3, g4, g5]
    g1, g2, g3, g4, g5 = K
    dq = (
        1.0
        + g1*np.cos(g1) + g2*np.sin(g2)
        + g3*np.sin(5*g1)*np.cos(3*g2)
        + g4*g5**1.5
        + 0.1*np.exp(-0.5*np.sum(K**2))
    )
    x_new = x0 + alpha*dq
    manip = np.sum(K)*np.sqrt(np.abs(x_new)) + 0.05*np.sin(7*np.prod(K))
    return x_new, manip

def total_cost(K):
    x_new, manip = forward(K)
    # complex oscillatory cost
    return (x_target - x_new)**2 + 0.01*(np.sin(10*x_new) + np.cos(5*manip))

# ------------------------
# NAIVE GD (2*d evals per step)
# ------------------------
def naive_GD(initial_K, lr=0.001, iters=200):
    K = initial_K.copy()
    for _ in range(iters):
        grad = np.zeros(d)
        eps = 1e-5
        for j in range(d):
            Kp = K.copy(); Km = K.copy()
            Kp[j] += eps
            Km[j] -= eps
            grad[j] = (total_cost(Kp) - total_cost(Km)) / (2*eps)
        K -= lr * grad
        K = np.clip(K, 0, 5)
    return K, total_cost(K)

# ------------------------
# CHAIN-RULE GD (analytical gradients)
# ------------------------
def chain_GD(initial_K, lr=0.001, iters=200):
    K = initial_K.copy()
    for _ in range(iters):
        # forward pass
        x_new, manip = forward(K)
        
        # dC/dx
        dC_dx = 2*(x_new - x_target) + 0.01*10*np.cos(10*x_new)
        # dx/dK = partial derivatives of dq
        g1, g2, g3, g4, g5 = K
        ddq = np.zeros(d)
        ddq[0] = np.cos(g1) - g1*np.sin(g1) + 5*g3*np.cos(5*g1)*np.cos(3*g2)
        ddq[1] = np.sin(g2) + g2*np.cos(g2) - 3*g3*np.sin(5*g1)*np.sin(3*g2)
        ddq[2] = np.sin(5*g1)*np.cos(3*g2)
        ddq[3] = g5**1.5
        ddq[4] = 1.5*g4*g5**0.5
        
        grad = dC_dx * alpha * ddq
        K -= lr * grad
        K = np.clip(K, 0, 5)
    return K, total_cost(K)

# ------------------------
# RUN BOTH OPTIMIZERS
# ------------------------
initial_K = np.random.rand(d)*5
cost_start = total_cost(initial_K)

K_naive, cost_naive = naive_GD(initial_K)
K_chain, cost_chain = chain_GD(initial_K)

# ------------------------
# PRINT RESULTS
# ------------------------
print("\n--- RESULTS ---")
print(f"Start:         K={initial_K}, cost={cost_start:.4f}")
print(f"Naive GD:      K={K_naive}, cost={cost_naive:.4f}")
print(f"Chain-rule GD: K={K_chain}, cost={cost_chain:.4f}")
print("------------------------\n")
