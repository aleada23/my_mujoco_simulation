import sympy as sp

# --------------------------------------------------------------
# 1. Declare symbolic joint variables
# --------------------------------------------------------------
q = sp.symbols('q1:8')

# --------------------------------------------------------------
# 2. DH transform function
# --------------------------------------------------------------
def dh(a, d, alpha, theta):
    ca, sa = sp.cos(alpha), sp.sin(alpha)
    ct, st = sp.cos(theta), sp.sin(theta)
    return sp.Matrix([
        [ct, -st, 0, a],
        [st*ca, ct*ca, -sa, -sa*d],
        [st*sa, ct*sa, ca, ca*d],
        [0, 0, 0, 1]
    ], evaluate=False)

# --------------------------------------------------------------
# 3. Franka Panda DH parameters
# --------------------------------------------------------------
dh_params = [
    (0, 0.333, 0, q[0]),
    (0, 0, -sp.pi/2, q[1]),
    (0, 0.316, sp.pi/2, q[2]),
    (0.0825, 0, sp.pi/2, q[3]),
    (-0.0825, 0.384, -sp.pi/2, q[4]),
    (0, 0, sp.pi/2, q[5]),
    (0.088, 0, sp.pi/2, q[6]),
    (0, 0.107, 0, 0),
    (0, 0, 0, -sp.pi/4),
    (0, 0.1034, 0, 0)
]

# --------------------------------------------------------------
# 4. Forward kinematics
# --------------------------------------------------------------
T = sp.eye(4)
T_list = []

for a, d, alpha, theta in dh_params:
    T = T * dh(a, d, alpha, theta)
    T_list.append(T)

T_EE = T_list[-1]

# --------------------------------------------------------------
# 5. Symbolic Jacobian J(q)
# --------------------------------------------------------------
J = sp.zeros(6, 7)
p_EE = T_EE[:3, 3]

for i in range(7):
    Ti = T_list[i]
    z_i = Ti[:3, 2]
    p_i = Ti[:3, 3]

    J[:3, i] = z_i.cross(p_EE - p_i)  # linear part
    J[3:, i] = z_i                     # angular part

# --------------------------------------------------------------
# 6. Symbolic derivatives of Jacobian
# --------------------------------------------------------------
dJdq = [J.diff(qi) for qi in q]

# --------------------------------------------------------------
# 7. Save compact Jacobian and derivatives to file
# --------------------------------------------------------------
with open("panda_symbolic_jacobian_compact.txt", "w") as f:
    f.write("J(q) =\n")
    f.write(str(J) + "\n\n")  # write compact form
    
    for i, dJ in enumerate(dJdq):
        f.write(f"dJ/dq{i+1} =\n")
        f.write(str(dJ) + "\n\n")

print("\nCompact symbolic Jacobian and derivatives saved to 'panda_symbolic_jacobian_compact.txt'")
