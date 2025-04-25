import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from scipy import signal
from amp_modeling import calculate_amplifier_tf
from parameters import *

# === 1.  MODE SHAPES (uniform cantilever) ===
def generate_cantilever_modes(n, L):
    betaL_roots = [1.8751040687119611, 4.694091132974174,
                   7.854757438237612, 10.995540734875466,
                   14.13716839104647, 17.27875953208823,
                   20.42035224562606, 23.56194490184045,
                   26.70353755550819, 29.84513020910325]  # first 10 roots
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

# === 2.  GEOMETRY ===
# choose the taper ratio you want ------------------------------
r_tip = 0.8          # 5 % of base width
k_exp = -np.log(r_tip)

def b_exp(x, L): return b0 * np.exp(-k_exp * x / L)     # exponential taper
def b_linear(x, L): return b0*(1-x/L)                   # linear taper
def b_rect(x,L): return b0

def EI(x, L):
    EI, _ = comp_props(b_rect(x, L))
    return EI

def rhoA(x, L):
    rhoA, _ = comp_props(b_rect(x, L))
    return rhoA

# quadrature
xi_g, w_g = leggauss(nq)
def integrate(f):
    xg = 0.5*(xi_g+1)*L
    return 0.5*L*np.sum(w_g*f(xg))

def run_simulation():
    # Generate mode shapes
    phi_list, d2phi_list = generate_cantilever_modes(n_modes, L)

    # === 3.  ASSEMBLE M,K matrices ===
    M = np.zeros((n_modes,n_modes))
    K = np.zeros_like(M)
    for i in range(n_modes):
        for j in range(n_modes):
            M[i,j] = integrate(lambda x, ii=i, jj=j: rhoA(x, L)*phi_list[ii](x)*phi_list[jj](x))
            K[i,j] = integrate(lambda x, ii=i, jj=j: EI(x, L)*d2phi_list[ii](x)*d2phi_list[jj](x))

    # === 4.  Piezoelectric coupling vector Λ (charge per unit generalized curvature q_i) ===
    z_eff = h_s/2    # distance from neutral axis to piezo inner surface
    def lambda_integrand(i):
        return lambda x: d31 * E_p * b_rect(x, L) * z_eff * d2phi_list[i](x)

    Lambda = np.array([integrate(lambda_integrand(i)) for i in range(n_modes)]).reshape((1,n_modes))  # row vector

    # === 5.  Frequency sweep ===
    f_Hz = np.logspace(0, np.log10(48e3), 1000)
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
    print(f"Pressure to Charge: {np.abs(Q_over_P[mid_idx])*1e12:.2e} pC/Pa")
    print(f"Amplifier: {np.abs(H_amp[mid_idx]):.2e} V/C")
    print(f"Overall Pressure to Voltage: {np.abs(V_over_P[mid_idx]):.2e} V/Pa")

    return f_Hz, Q_over_P, V_over_P, U_over_P, U_over_P_unloaded, H_amp, mag_amp

def plot_results(f_Hz, Q_over_P, V_over_P, U_over_P, U_over_P_unloaded, H_amp, mag_amp):
    # Plot amplifier transfer function
    plt.figure()
    plt.semilogx(f_Hz, mag_amp, label='|H_amp|')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('Amplifier Transfer Function')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot pressure responses
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
    plt.title('Pressure to Voltage Response')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot displacement
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

def run_simulation_for_length(L_value):
    """Run simulation for a specific beam length"""
    # Generate mode shapes
    phi_list, d2phi_list = generate_cantilever_modes(n_modes, L_value)

    # === 3.  ASSEMBLE M,K matrices ===
    M = np.zeros((n_modes,n_modes))
    K = np.zeros_like(M)
    for i in range(n_modes):
        for j in range(n_modes):
            M[i,j] = integrate(lambda x, ii=i, jj=j: rhoA(x, L_value)*phi_list[ii](x)*phi_list[jj](x))
            K[i,j] = integrate(lambda x, ii=i, jj=j: EI(x, L_value)*d2phi_list[ii](x)*d2phi_list[jj](x))

    # === 4.  Piezoelectric coupling vector Λ ===
    z_eff = h_s/2
    def lambda_integrand(i):
        return lambda x: d31 * E_p * b_rect(x, L_value) * z_eff * d2phi_list[i](x)

    Lambda = np.array([integrate(lambda_integrand(i)) for i in range(n_modes)]).reshape((1,n_modes))

    # === 5.  Frequency sweep ===
    f_Hz = np.logspace(0, 5, 1000)
    w = 2*np.pi*f_Hz
    Q_over_P = np.zeros_like(w, dtype=complex)
    Q_over_P_piezostack = np.zeros_like(w, dtype=complex)
    U_over_P = np.zeros_like(w, dtype=complex)
    U_over_P_unloaded = np.zeros_like(w, dtype=complex)
    U_piezostack = np.zeros_like(w, dtype=complex)
    p_vec = np.array([phi(L_value) for phi in phi_list]).reshape((n_modes,1))

    H_amp, mag_amp, phase_amp = calculate_amplifier_tf(f_Hz)

    for k, omega in enumerate(w):
        D = K - omega**2 * M
        invD = np.linalg.inv(D)
        k_b  = 1.0 / (p_vec.T @ invD @ p_vec)
        Z_msd = -omega**2 * m_lump - 1j*omega*c_lump + k_lump
        den   = k_b + Z_msd

        # tip displacement with piezostack
        U_piezostack[k] = (K_p * delta_f) / (k_b + K_p)
        # charge with piezostack
        q_piezostack = (invD @ p_vec) * (k_b * U_piezostack[k])
        Q_over_P_piezostack[k] = Lambda @ q_piezostack
        
        # tip displacement with umbo
        U_over_P[k] = A_tm / den

        # umbo displacement no loading
        U_over_P_unloaded[k] = A_tm/Z_msd
        
        # charge calculation
        q = (invD @ p_vec) * (k_b * U_over_P[k])
        Q_over_P[k] = Lambda @ q

    V_over_P = Q_over_P * H_amp
    return f_Hz, np.abs(V_over_P), np.abs(Q_over_P), np.abs(Q_over_P_piezostack)

def find_bandwidth(freq, response, ref_idx):
    """Find the -3dB bandwidth around the reference gain"""
    # Find the peak frequency
    peak_idx = np.argmax(response)
    ref_gain = response[ref_idx]
    peak_freq = freq[peak_idx]
    
    # Find the -3dB points
    target_gain = ref_gain / np.sqrt(2)
    
    # Find lower cutoff
    lower_idx = np.argmin(np.abs(response[:peak_idx] - target_gain))
    lower_freq = freq[lower_idx]
    
    # Find upper cutoff
    upper_idx = ref_idx + np.argmin(np.abs(response[ref_idx:] - target_gain))
    upper_freq = freq[upper_idx]
    
    return lower_freq, upper_freq, upper_freq - lower_freq

def plot_length_sweep_results(f_Hz, lengths, responses, midband_gains, bandwidths, piezostack_responses, charge_responses):
    """Plot waterfall and analysis plots"""
    # Waterfall plot for pressure to voltage
    plt.figure(figsize=(12, 8))
    for i, (L, response) in enumerate(zip(lengths, responses)):
        # Plot the frequency response
        plt.loglog(f_Hz, response, label=f'L={L*1e3:.1f}mm')
        
        # Find and plot midband point (1kHz)
        mid_idx = np.argmin(np.abs(f_Hz - 1000))
        plt.plot(f_Hz[mid_idx], response[mid_idx], 'ko', markersize=5)
        
        # Find and plot bandwidth points
        lower_freq, upper_freq, _ = find_bandwidth(f_Hz, response, mid_idx)
        lower_idx = np.argmin(np.abs(f_Hz - lower_freq))
        upper_idx = np.argmin(np.abs(f_Hz - upper_freq))
        plt.plot(f_Hz[lower_idx], response[lower_idx], 'rx', markersize=5)
        plt.plot(f_Hz[upper_idx], response[upper_idx], 'rx', markersize=5)
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [V/Pa]')
    plt.title('Pressure to Voltage Response for Different Beam Lengths\n(Black dots: midband gain, Red crosses: -3dB points)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Waterfall plot for charge output
    plt.figure(figsize=(12, 8))
    for i, (L, response) in enumerate(zip(lengths, charge_responses)):
        plt.loglog(f_Hz, response*1e12, label=f'L={L*1e3:.1f}mm')
        
        # Find and plot midband point (1kHz)
        mid_idx = np.argmin(np.abs(f_Hz - 1000))
        plt.plot(f_Hz[mid_idx], response[mid_idx]*1e12, 'ko', markersize=5)
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [pC/Pa]')
    plt.title('Pressure to Charge Response for Different Beam Lengths\n(Black dots: midband gain)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Waterfall plot for piezostack response
    plt.figure(figsize=(12, 8))
    for i, (L, response) in enumerate(zip(lengths, piezostack_responses)):
        plt.loglog(f_Hz, response*1e15/(delta_f*1e9), label=f'L={L*1e3:.1f}mm')
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [fC/nm]')
    plt.title('Piezostack Displacement Response')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Create figure for all gain and bandwidth plots
    plt.figure(figsize=(12, 8))
    
    # V/P midband gain vs length
    plt.subplot(1, 2, 1)
    plt.plot(lengths*1e3, midband_gains, 'o-')
    plt.xlabel('Beam Length [mm]')
    plt.ylabel('Midband Gain [V/Pa]')
    plt.title('V/P Midband Gain vs Beam Length')
    plt.grid(True)

    # V/P bandwidth vs length
    plt.subplot(1, 2, 2)
    plt.plot(lengths*1e3, bandwidths, 'o-')
    plt.xlabel('Beam Length [mm]')
    plt.ylabel('Bandwidth [Hz]')
    plt.title('V/P Bandwidth vs Beam Length')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_beam_profile(L_value):
    """Plot the beam width profile along its length"""
    x = np.linspace(0, L_value, 100)
    
    # Calculate both taper profiles using the defined functions
    width_exp = b_exp(x, L_value)  # exponential taper
    width_lin = b_linear(x, L_value)  # linear taper
    width_rect = b_rect(x, L_value)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x*1e3, width_exp*1e3, 'b-', label='Exponential taper')
    plt.plot(x*1e3, width_lin*1e3, 'r--', label='Linear taper')
    plt.xlabel('Position along beam [mm]')
    plt.ylabel('Beam width [mm]')
    plt.title(f'Beam Width Profile (L = {L_value*1e3:.1f}mm)')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    # Run original simulation with default parameters
    print("\nRunning simulation with default parameters:")
    results = run_simulation()
    plot_results(*results)
    
    # Plot beam profile for default length
    print("\nPlotting beam profile:")
    plot_beam_profile(L)
    
    # Run length sweep analysis
    print("\nRunning length sweep analysis:")
    # Length sweep parameters
    lengths = np.linspace(2e-3, 5e-3, 25)  # 25 points between 2 mm and 5 mm
    responses = []
    piezostack_responses = []
    charge_responses = []
    midband_gains = []
    bandwidths = []

    # Run simulation for each length
    for L_value in lengths:
        f_Hz, response, charge_response, response_piezostack = run_simulation_for_length(L_value)
        responses.append(response)
        charge_responses.append(charge_response)
        piezostack_responses.append(response_piezostack)
        
        # Find midband gain (around 1kHz)
        mid_idx = np.argmin(np.abs(f_Hz - 1000))
        midband_gain = response[mid_idx]
        midband_gains.append(midband_gain)
        
        # Find bandwidth
        lower_freq, upper_freq, bandwidth = find_bandwidth(f_Hz, response, mid_idx)
        bandwidths.append(bandwidth)
        
        print(f"Length: {L_value*1e3:.1f}mm, Midband Gain: {midband_gain:.2e} V/Pa, Bandwidth: {bandwidth:.0f} Hz")

    # Plot length sweep results
    plot_length_sweep_results(f_Hz, lengths, responses, midband_gains, bandwidths, piezostack_responses, charge_responses)

if __name__ == "__main__":
    main()
