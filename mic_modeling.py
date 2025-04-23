import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from scipy import signal
from amp_modeling import calculate_amplifier_tf

# === PARAMETERS (copied / extended) ===
L   = 3e-3          # beam length [m]
b0  = 3e-3          # base width [m]
h_s = 150e-6        # substrate thickness [m]
E_s = 5e9           # substrate Young's modulus [Pa]
rho = 1800.0        # substrate density [kg/m3]

# lumped TM+ossicular chain parameters
m_lump = 14e-6      # [kg]
k_lump = 900        # [N/m]
c_lump = 0.1        # [N·s/m]
A_tm   = 65e-6      # tymp membrane area [m^2]

# piezo-layer properties (PVDF example)
t_p   = 25e-6       # piezo thickness [m]
E_p   = 3e9         # piezo Young's modulus [Pa]
d31   = -23e-12     # piezoelectric coeff [C/N]  (PVDF negative)

n_modes = 5         # Ritz functions (cantilever modes)
nq = 12             # quadrature points

# === 1.  MODE SHAPES (uniform cantilever) ===
def generate_cantilever_modes(n, L):
    betaL_roots = [1.8751040687119611, 4.694091132974174,
                   7.854757438237612, 10.995540734875466,
                   14.13716839104647]  # extend if needed
    if n > len(betaL_roots):
        raise ValueError("extend betaL list")
    phi, d2phi = [], []
    for r in betaL_roots[:n]:
        beta = r/L
        C1 = np.cosh(r) + np.cos(r)
        C2 = np.sinh(r) + np.sin(r)
        a = C1/C2
        phi.append(lambda x, b=beta, a=a: np.cosh(b*x) - np.cos(b*x) - a*(np.sinh(b*x)-np.sin(b*x)))
        d2phi.append(lambda x, b=beta, a=a: b**2*(np.cosh(b*x)+np.cos(b*x) - a*(np.sinh(b*x)+np.sin(b*x))))
    return phi, d2phi

phi_list, d2phi_list = generate_cantilever_modes(n_modes, L)

# === 2.  GEOMETRY ===
def b(x): return b0*(1-x/L)               # linear taper
def A(x): return b(x)*h_s
def I(x): return b(x)*h_s**3/12
EI = lambda x: E_s*I(x)

# quadrature
xi_g, w_g = leggauss(nq)
def integrate(f):
    xg = 0.5*(xi_g+1)*L
    return 0.5*L*np.sum(w_g*f(xg))

# === 3.  ASSEMBLE M,K matrices ===
M = np.zeros((n_modes,n_modes))
K = np.zeros_like(M)
for i in range(n_modes):
    for j in range(n_modes):
        M[i,j] = integrate(lambda x, ii=i, jj=j: rho*A(x)*phi_list[ii](x)*phi_list[jj](x))
        K[i,j] = integrate(lambda x, ii=i, jj=j: EI(x)*d2phi_list[ii](x)*d2phi_list[jj](x))

# === 4.  Piezoelectric coupling vector Λ (charge per unit generalized curvature q_i) ===
z_eff = h_s/2 + t_p/2    # distance from neutral axis to piezo centre
def lambda_integrand(i):
    return lambda x: d31 * E_p * b(x) * z_eff * d2phi_list[i](x)

Lambda = np.array([integrate(lambda_integrand(i)) for i in range(n_modes)]).reshape((1,n_modes))  # row vector

# === 5.  Frequency sweep ===
f_Hz = np.logspace(0, 4, 1000)
w = 2*np.pi*f_Hz
Q_over_P = np.zeros_like(w, dtype=complex)   # charge per acoustic pressure
U_over_P = np.zeros_like(w, dtype=complex)   # tip displacement for reference
U_over_P_unloaded = np.zeros_like(w, dtype=complex)   # tip displacement for reference
p_vec = np.array([phi(L) for phi in phi_list]).reshape((n_modes,1))

# Get amplifier transfer function
H_amp, mag_amp, phase_amp = calculate_amplifier_tf(f_Hz)

for k, omega in enumerate(w):
    D = K - omega**2 * M
    invD = np.linalg.inv(D)
    k_b  = 1.0 / (p_vec.T @ invD @ p_vec)            # dynamic stiffness
    Z_msd = -omega**2 * m_lump - 1j*omega*c_lump + k_lump
    den   = k_b + Z_msd
    U_over_P[k] = A_tm / den
    # generalized coordinates for unit pressure: q = invD p * k_b * U
    q = (invD @ p_vec) * (k_b * U_over_P[k])
    Q_over_P[k] = Lambda @ q                        # scalar
    U_over_P_unloaded[k] = A_tm/Z_msd

# Calculate overall pressure-to-voltage response
V_over_P = Q_over_P * H_amp

# Print midband gains
mid_idx = np.argmin(np.abs(f_Hz - 1000))   # ~1 kHz
print("\nMidband gains:")
print(f"Pressure to Charge: {np.abs(Q_over_P[mid_idx])*1e12:.2e} fC/Pa")
print(f"Amplifier: {np.abs(H_amp[mid_idx]):.2e} V/C")
print(f"Overall Pressure to Voltage: {np.abs(V_over_P[mid_idx]):.2e} V/Pa")

# === 6.  Plot responses ===
plt.figure()
plt.semilogx(f_Hz, mag_amp, label='|H_amp|')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.title('Amplifier Transfer Function')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))

# Pressure to Charge
plt.subplot(2, 1, 1)
plt.loglog(f_Hz, np.abs(Q_over_P)*1e12, label='|Q/P|')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [fC / Pa]')
plt.title('Pressure to Charge Response')
plt.grid(True)
plt.legend()

# Pressure to Voltage
plt.subplot(2, 1, 2)
plt.loglog(f_Hz, np.abs(V_over_P), label='|V/P|')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [V / Pa]')
plt.title('Overall Pressure to Voltage Response')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# also plot displacement if desired
plt.figure()
plt.loglog(f_Hz, np.abs(U_over_P_unloaded)*1e9, label='unloaded')
plt.loglog(f_Hz, np.abs(U_over_P)*1e9, label='loaded')
plt.xlim(100, 10e3)
plt.ylim(1e-1, 1e3)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Tip Displacement [nm/Pa]')
plt.grid(True)
plt.legend()
plt.show()
