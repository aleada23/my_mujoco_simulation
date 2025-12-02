import numpy as np

# ------------------------
# 10D COMPLEX COST + MODEL
# ------------------------
x_target = 10
x0 = 2
alpha = 0.5
d = 10  # 10 gains

np.random.seed(0)

def forward(K):
    # K is 10D: g1..g10
    dq = 1.0
    # sum of nonlinear couplings and oscillations
    for i in range(d):
        dq += K[i]*np.cos(K[i]) + 0.1*np.sin(5*K[i])
        for j in range(i+1, d):
            dq += 0.05*(K[i]*K[j])**1.2 + 0.02*np.sin(7*K[i]*K[j])
    
    x_new = x0 + alpha*dq
    manip = np.sum(K)*np.sqrt(np.abs(x_new)) + 0.05*np.sin(np.prod(K))
    return x_new, manip

def total_cost(K):
    x_new, manip = forward(K)
    return (x_target - x_new)**2 + 0.01*(np.sin(10*x_new) + np.cos(5*manip))

# ------------------------
# NAIVE GD
# ------------------------
def naive_GD(initial_K, lr=0.001, iters=1000):
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
# CHAIN-RULE GD
# ------------------------
def chain_GD(initial_K, lr=0.001, iters=1000):
    K = initial_K.copy()
    for _ in range(iters):
        x_new, manip = forward(K)
        # dC/dx_new
        dC_dx = 2*(x_new - x_target) + 0.01*10*np.cos(10*x_new)
        # dq derivatives
        ddq = np.zeros(d)
        for i in range(d):
            ddq[i] = np.cos(K[i]) - K[i]*np.sin(K[i]) + 0.1*5*np.cos(5*K[i])
            for j in range(d):
                if j > i:
                    ddq[i] += 0.05*1.2*(K[i]*K[j])**0.2*K[j] + 0.02*7*K[j]*np.cos(7*K[i]*K[j])
        grad = dC_dx * alpha * ddq
        K -= lr * grad
        K = np.clip(K, 0, 5)
    return K, total_cost(K)

# ------------------------
# RUN OPTIMIZERS
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
