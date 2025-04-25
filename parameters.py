# Physical parameters
L = 3e-3          # beam length [m]
b0 = 3e-3          # base width [m]
h_s = 150e-6        # substrate thickness [m]
E_s = 3.3e9           # substrate Young's modulus [Pa]
rho_s = 1400.0        # substrate density [kg/m3]

# Lumped TM+ossicular chain parameters
m_lump = 14e-6      # [kg]
k_lump = 900        # [N/m]
c_lump = 0.1        # [N·s/m]
A_tm = 65e-6        # tymp membrane area [m^2]

# Piezo-layer properties (PVDF example)
h_p = 50e-6         # piezo thickness [m]
E_p = 3e9           # piezo Young's modulus [Pa]
d31 = -23e-12       # piezoelectric coeff [C/N]  (PVDF negative)
rho_p = 1800.0      # piezo density [kg/m3]

# Simulation parameters
n_modes = 10         # Ritz functions (cantilever modes)
nq = 12             # quadrature points 

# Piezostack parameters
K_p = 1e8          # blocked stiffness of the stack  [N/m]
d33 = 3e-8         # piezoelectric coeff [m/V]
V_drive = 1
delta_f = d33 * V_drive   # free extension [m]

t_tot = h_p * 2 + h_s

# ------------------------------------------------------------------
# helper: composite section props at given width b -----------------
# ------------------------------------------------------------------
def comp_props(b):
    """
    returns (EI, rhoA)  for width b  [N·m² , kg/m]
    NA is measured from laminate mid-plane (positive upwards).
    """
    # areas
    A_s = b * h_s
    A_p = b * h_p
    # axial stiffness contributions
    EA_s = E_s * A_s
    EA_p = E_p * A_p

    # neutral-axis position  ȳ   (0 = mid-plane of whole laminate)
    # because the stack is symmetric in geometry but not in modulus:
    ybar = (EA_p * (+t_tot/2) + EA_p * (-t_tot/2) + EA_s*0) \
           / (2*EA_p + EA_s)

    # second moment of each layer about its own centroid
    I_s0 = b * h_s**3 / 12
    I_p0 = b * h_p**3 / 12

    # parallel-axis shift distances
    d_s = abs(ybar)                     # substrate centroid at 0
    d_p = abs((t_tot/2) - ybar)         # upper  piezo → +t_tot/2
    # same for lower piezo

    # composite EI = Σ E_i ( I_i0 + A_i d_i² )
    EI = (E_s*(I_s0 + A_s*d_s**2)
          + E_p*(I_p0 + A_p*d_p**2) * 2)

    # distributed mass ρA  [kg/m]
    rhoA = rho_s*A_s + rho_p*A_p*2

    return EI, rhoA