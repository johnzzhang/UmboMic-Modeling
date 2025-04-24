# Physical parameters
L = 3e-3          # beam length [m]
b0 = 3e-3          # base width [m]
h_s = 150e-6        # substrate thickness [m]
E_s = 5e9           # substrate Young's modulus [Pa]
rho = 1800.0        # substrate density [kg/m3]

# Lumped TM+ossicular chain parameters
m_lump = 14e-6      # [kg]
k_lump = 900        # [N/m]
c_lump = 0.1        # [N·s/m]
A_tm = 65e-6        # tymp membrane area [m^2]

# Piezo-layer properties (PVDF example)
t_p = 50e-6         # piezo thickness [m]
E_p = 3e9           # piezo Young's modulus [Pa]
d31 = -23e-12       # piezoelectric coeff [C/N]  (PVDF negative)

# Simulation parameters
n_modes = 10         # Ritz functions (cantilever modes)
nq = 12             # quadrature points 

# Piezostack parameters
K_p = 1e8          # blocked stiffness of the stack  [N/m]
d33 = 3e-8         # piezoelectric coeff [m/V]
V_drive = 1
delta_f = d33 * V_drive   # free extension [m]

# # inside the frequency loop ---------------------------
# D = K - w**2 * M
# invD = np.linalg.inv(D)
# k_b  = 1.0 / (p.T @ invD @ p)        # dynamic stiffness of beam tip

# # modified boundary: (K_b + K_p) * W = K_p * delta_f
# W_tip = (K_p * delta_f) / (k_b + K_p)
