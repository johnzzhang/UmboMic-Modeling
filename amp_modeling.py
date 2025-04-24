# plot_charge_amp_tf.py
"""Plot magnitude and phase of the piezo charge amplifier transfer function
    H(s) = V_OUT_HPF(s) / Q_piezo(s)

Schematic values (copy exact):
    R1  = 10 GΩ  , C1  = 1 pF    # feedback of charge amp
    R9  = 10 kΩ  , C8  = 1 nF    # post low pass
    R10 = 100 kΩ , C4  = 100 nF  # ac coupling high pass

The algebra (see chat):
    H(s) = -(2*R1) * s                      # first zero (at DC) and gain
            / (1 + s R1 C1)               # feedback pole
          * 1 / (1 + s R9 C8)             # post low pass pole
          * (s R10 C4) / (1 + s R10 C4)   # ac coupling zero & pole (HP)

Above a few Hz and below ~10 kHz the magnitude should level at 2 V/pC
(≈ +6 dB relative to 1 V/pC or ≈ +246 dB relative to 1 V/C).
"""

import numpy as np
import matplotlib.pyplot as plt

def get_amplifier_parameters():
    """Return the amplifier component values."""
    return {
        'R1': 10e9,    # 10  GΩ
        'C1': 1e-12,   # 1   pF
        'R9': 10e3,    # 10  kΩ
        'C8': 1e-9,    # 1   nF
        'R10': 100e3,  # 100 kΩ
        'C4': 100e-9   # 100 nF
    }

def calculate_amplifier_tf(freq):
    """Calculate the amplifier transfer function at given frequencies.
    
    Args:
        freq: Array of frequencies in Hz
        
    Returns:
        H: Complex transfer function values
        mag_db: Magnitude in dB (relative to 1 V/pC)
        phase_deg: Phase in degrees
    """
    params = get_amplifier_parameters()
    R1, C1, R9, C8, R10, C4 = [params[k] for k in ['R1', 'C1', 'R9', 'C8', 'R10', 'C4']]
    
    w = 2 * np.pi * freq
    s = 1j * w
    
    H = ( -2 * R1 * s ) / (1 + s * R1 * C1)       # charge amp core
    H *= 1 / (1 + s * R9 * C8)                    # post LP pole
    H *= (s * R10 * C4) / (1 + s * R10 * C4)      # HP zero+pole
    
    mag_db = 20 * np.log10(np.abs(H))
    phase_deg = np.unwrap(np.angle(H)) * 180/np.pi
    
    return H, mag_db, phase_deg

def plot_amplifier_tf(freq, H, mag_db, phase_deg):
    """Plot the amplifier transfer function.
    
    Args:
        freq: Array of frequencies in Hz
        H: Complex transfer function values
        mag_db: Magnitude in dB
        phase_deg: Phase in degrees
    """
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax[0].semilogx(freq, mag_db)
    ax[0].set_ylabel('Magnitude [dB re 1 V/C]')
    ax[0].grid(True, which='both', ls=':')
    ax[0].set_title('Charge Amplifier Transfer Function |H(jω)| & ∠H(jω)')

    ax[1].semilogx(freq, phase_deg)
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('Phase [deg]')
    ax[1].grid(True, which='both', ls=':')

    plt.tight_layout()
    plt.show()
    
    # Print mid-band gain for sanity check
    mid_idx = np.argmin(np.abs(freq - 1000))   # ~1 kHz
    print(f"Mid band |H| ≈ {np.abs(H[mid_idx]):.2e} V/C  ( {mag_db[mid_idx]:.1f} dB re 1 V/C )")

if __name__ == "__main__":
    # Example usage when run directly
    f_min, f_max = 0.1, 1e6     # Hz
    n_points = 3000
    f = np.logspace(np.log10(f_min), np.log10(f_max), n_points)
    
    H, mag_db, phase_deg = calculate_amplifier_tf(f)
    plot_amplifier_tf(f, H, mag_db, phase_deg)
