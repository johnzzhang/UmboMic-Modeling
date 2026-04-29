import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def calculate_charge_frf():
    # 1. Physical Parameters
    L = 0.006            # m
    b_base = 0.003       # m (width at base)
    h = 200e-6           # m
    E = 2.7e9            # Pa
    rho = 1800           # kg/m^3
    k_u = 833            # N/m
    m_u = 10.8e-6          # kg
    c_u = 0.038          # Ns/m
    c_nd = 20.0          # distributed damping of the beam
    a = 1.609            # Taper parameter
    A_TM = 50e-6         # m^2
    F = 10e-6            # N 80 dB SPL (0.2 Pa) for 50 mm^2 TM area
    
    # 2. Piezoelectric Parameters
    z_m = 50e-6         # m (Distance to neutral axis)
    e31 = 0.025           # C/m^2 (Piezoelectric stress constant)
    
    # 3. Structural Properties
    I_0 = (b_base * h**3) / 12
    EI_0 = E * I_0
    rho_0 = rho * b_base * h
    
    # Nondimensional Parameters
    kappa = (k_u * L**3) / EI_0
    nu = m_u / (rho_0 * L)
    zeta = (c_u * L) / np.sqrt(rho_0 * EI_0)
    
    # 4. Frequency Sweep Setup
    f_hz = np.linspace(10, 10e3, 1000)
    # Charge computed from the weighted curvature integral:
    # Q = -z_m e31 b_base * integral( w''(x) * exp(-a x/L) dx, 0..L )
    charge_frf_magnitude = np.zeros_like(f_hz)

    # Spatial grid (nondimensional and physical) for storing beam shapes
    n_x = 100
    x_nd = np.linspace(0, 1, n_x)
    x_phys = x_nd * L
    w_shapes = np.zeros((len(f_hz), n_x), dtype=complex)   # displacement w(x) per N input
    wpp_shapes = np.zeros((len(f_hz), n_x), dtype=complex) # curvature w''(x) per N input

    B = np.array([0, 0, 0, 1], dtype=complex)
    dx = 1e-4 
    
    # 5. Evaluate Frequency Response
    for i, f in enumerate(f_hz):
        # Nondimensional frequency
        w = 2 * np.pi * f * (L**2) * np.sqrt(rho_0 / EI_0)
        
        # for damping, use complex frequency
        w_complex = np.sqrt(w**2 - 1j * w * c_nd)

        # Spatial frequencies
        alpha = np.sqrt(4*w_complex + a**2) / 2
        beta = np.sqrt(np.abs(4*w_complex - a**2)) / 2 
        gamma = np.sqrt(np.abs(- 4*w_complex + a**2)) / 2
        
        # Basis functions
        def X_func(x_val):
            e_val = np.exp(a * x_val / 2)
            return np.array([
                e_val * np.cosh(alpha * x_val),
                e_val * np.sinh(alpha * x_val),
                e_val * np.cos(beta * x_val),
                e_val * np.sin(beta * x_val)
            ], dtype=complex)
            
        # Extract numerical derivatives needed for the boundary conditions
        X_1 = X_func(1.0)
        X_prime_0 = (X_func(dx) - X_func(-dx)) / (2 * dx)
        X_prime2_1 = (X_func(1 + dx) - 2*X_func(1) + X_func(1 - dx)) / (dx**2)
        X_prime3_1 = (X_func(1 + 2*dx) - 2*X_func(1 + dx) + 2*X_func(1 - dx) - X_func(1 - 2*dx)) / (2 * dx**3)
        
        # Build Complex Boundary Condition Matrix A
        A = np.zeros((4, 4), dtype=complex)
        A[0, :] = X_func(0.0)
        A[1, :] = X_prime_0
        A[2, :] = X_prime2_1
        
        tip_dynamics = kappa + 1j * w * zeta - nu * w**2
        A[3, :] = np.exp(-a) * X_prime3_1 - (tip_dynamics * X_1)
        
        try:
            # Solve for mode shape coefficients
            C = np.linalg.solve(A, B)

            # --- BEAM SHAPE: w(x) and w''(x) along the length ---
            # Physical scaling factor per Newton of input force
            disp_scaling = (L**3) / EI_0
            # Physical curvature scaling: d^2w/dx^2 = (1/L^2) d^2w_hat/dxi^2
            curv_scaling = disp_scaling / L**2  # = L / EI_0

            # Stack basis evaluations: shape (n_x, 4)
            X_at_x = np.array([X_func(xv) for xv in x_nd])
            # Numerical second derivative w.r.t. nondimensional xi = x/L
            X_pp_at_x = np.array([
                (X_func(xv + dx) - 2*X_func(xv) + X_func(xv - dx)) / dx**2
                for xv in x_nd
            ])

            w_x_array = (X_at_x @ C) * disp_scaling          # complex w(x) [m/N]
            wpp_x_array = (X_pp_at_x @ C) * curv_scaling     # complex w''(x) [1/(m*N)]

            w_shapes[i, :] = w_x_array
            wpp_shapes[i, :] = wpp_x_array

            # --- CHARGE FROM WEIGHTED CURVATURE INTEGRAL ---
            # Q = -z_m e31 b_base * integral( w''(x) * exp(-a x/L) dx, 0..L )
            curvature_integrand = wpp_x_array * np.exp(-a * x_nd)
            curvature_integral = np.trapz(curvature_integrand, x=x_nd * L)
            Q = -z_m * e31 * b_base * curvature_integral
            charge_frf_magnitude[i] = np.abs(Q)

        except np.linalg.LinAlgError:
            charge_frf_magnitude[i] = np.nan
            w_shapes[i, :] = np.nan
            wpp_shapes[i, :] = np.nan
            
    # 6. Combined interactive plot: FRF (left) + beam shape (right)
    # Convert to per-Pa amplitude.
    w_per_pa = w_shapes * A_TM       # [m/Pa]
    wpp_per_pa = wpp_shapes * A_TM   # [1/(m*Pa)]
    charge_per_pa = charge_frf_magnitude * A_TM * 1e15  # [fC/Pa]

    i_init = 0
    fig_int = plt.figure(figsize=(14, 8))
    gs = fig_int.add_gridspec(
        2, 2, width_ratios=[1.1, 1.0], height_ratios=[1, 1],
        left=0.08, right=0.97, top=0.93, bottom=0.16, wspace=0.28, hspace=0.35,
    )
    ax_frf = fig_int.add_subplot(gs[:, 0])
    ax_w = fig_int.add_subplot(gs[0, 1])
    ax_wpp = fig_int.add_subplot(gs[1, 1], sharex=ax_w)

    # --- Left: FRF with marker indicating slider frequency ---
    ax_frf.loglog(f_hz, charge_per_pa, color='#ff7f0e', linewidth=2,
                  label=r'$|Q/P|$ from $\int_0^L w\,\!^{\prime\prime}(x)\,e^{-ax/L}dx$')
    marker_frf, = ax_frf.plot(f_hz[i_init], charge_per_pa[i_init],
                              marker='o', markersize=10,
                              markerfacecolor='#d62728', markeredgecolor='k',
                              markeredgewidth=1.2, linestyle='None', zorder=5,
                              label=f'f = {f_hz[i_init]:.1f} Hz')
    ax_frf.set_title('Piezoelectric Charge Output FRF $|Q/P|$', fontsize=14)
    ax_frf.set_xlabel('Frequency [Hz]', fontsize=12)
    ax_frf.set_ylabel('Charge re Ear Canal Pressure [fC/Pa]', fontsize=12)
    ax_frf.grid(True, which='both', ls='--', alpha=0.5)
    ax_frf.legend(loc='best', fontsize=10)
    ax_frf.set_ylim(1e0, 1e2)
    ax_frf.set_xlim(100,10e3)

    # --- Right top: real part of w(x) ---
    line_w_re, = ax_w.plot(x_phys * 1e3, np.real(w_per_pa[i_init]) * 1e9,
                           color='#1f77b4', linewidth=2, label='Re{w(x)}')
    ax_w.axhline(0, color='k', linewidth=0.5, alpha=0.4)
    ax_w.set_ylabel('Displacement [nm/Pa]', fontsize=12)
    ax_w.set_title(f'Beam Displacement')
    ax_w.grid(True, ls='--', alpha=0.5)
    ax_w.legend(loc='upper left', fontsize=9)

    # --- Right bottom: real part of w''(x) ---
    line_wpp_re, = ax_wpp.plot(x_phys * 1e3, np.real(wpp_per_pa[i_init]),
                               color='#d62728', linewidth=2, label="Re{w''(x)}")
    ax_wpp.axhline(0, color='k', linewidth=0.5, alpha=0.4)
    ax_wpp.set_xlabel('x [mm]', fontsize=12)
    ax_wpp.set_ylabel("Curvature [1/(m·Pa)]", fontsize=12)
    ax_wpp.set_title("Beam Curvature")
    ax_wpp.grid(True, ls='--', alpha=0.5)
    ax_wpp.legend(loc='upper left', fontsize=9)

    # --- Frequency slider ---
    ax_slider = fig_int.add_axes([0.15, 0.04, 0.7, 0.03])
    freq_slider = Slider(
        ax_slider, 'Frequency [Hz]',
        f_hz[0], f_hz[-1],
        valinit=f_hz[i_init],
        valstep=(f_hz[1] - f_hz[0]),
    )

    def update(_val):
        i_sel = int(np.argmin(np.abs(f_hz - freq_slider.val)))

        w_re = np.real(w_per_pa[i_sel]) * 1e9
        wpp_re = np.real(wpp_per_pa[i_sel])

        line_w_re.set_ydata(w_re)
        line_wpp_re.set_ydata(wpp_re)
        marker_frf.set_data([f_hz[i_sel]], [charge_per_pa[i_sel]])
        marker_frf.set_label(f'f = {f_hz[i_sel]:.1f} Hz')
        ax_frf.legend(loc='best', fontsize=10)

        # Symmetric y-limits with a small pad so sign changes are visible
        w_pad = 1.1 * max(np.nanmax(np.abs(w_re)), 1e-30)
        wpp_pad = 1.1 * max(np.nanmax(np.abs(wpp_re)), 1e-30)
        ax_w.set_ylim(-w_pad, w_pad)
        ax_wpp.set_ylim(-wpp_pad, wpp_pad)

        fig_int.canvas.draw_idle()

    freq_slider.on_changed(update)
    update(None)  # initialize axis limits

    # Keep slider reference alive after function returns
    fig_int._freq_slider = freq_slider

    plt.show()

if __name__ == "__main__":
    calculate_charge_frf()