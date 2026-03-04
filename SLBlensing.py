import numpy as np
import scipy as cp
from scipy import integrate
import matplotlib.pyplot as plt
from astropy.constants import G, c, M_sun
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib import gridspec

class SLBlensing:
    def __init__(self, mass_lens, r_lens, l_lens, mass_star, r_star, l_star, ecc, a, d, omega=np.pi, inc=90*u.deg, period=0.0*u.day, cycles=1.0, limb_darkening_wd=[0.0, 0.0], limb_darkening_star=[0.0, 0.0], N=1000, spin=0):
        self.M_l = mass_lens * M_sun
        self.R_l = r_lens.to(u.m)
        self.M_s = mass_star * M_sun
        self.R_star = r_star.to(u.m)
        self.L_star = l_star * u.L_sun
        self.L_wd = l_lens * u.L_sun
        self.N = N

        self.A_star = np.pi * self.R_star**2
        self.A_l = np.pi * self.R_l**2
        self.spin = spin

        self.e = ecc
        self.a = a.to(u.m)
        self.inc = inc.to(u.rad)
        self.d = d.to(u.m)
        self.omega = omega.to(u.rad)      #argument of perhelion
        self.limb_darkening_coeffs_wd = limb_darkening_wd
        self.limb_darkening_coeffs_star = limb_darkening_star
        self.cycles = cycles

        self.circ = (self.e == 0)
        self.F_lens = self.L_wd / (4 * np.pi * self.d**2)
        self.F_star = self.L_star / (4 * np.pi * self.d**2)

        if (period == 0.0):
            self.orbital_period()
        else:
            self.P = (period).to(u.yr)

        # offset calc
        f_conj = np.pi/2 - self.omega.to(u.rad).value

        E_conj = 2 * np.arctan2(np.sqrt(1 - self.e) * np.sin(f_conj / 2),
                                np.sqrt(1 + self.e) * np.cos(f_conj / 2))

        M_conj = E_conj - self.e * np.sin(E_conj)
        M_conj = M_conj % (2 * np.pi)

        t_conj = (M_conj / (2 * np.pi)) * self.P.to(u.day)
        t_offset = -t_conj + 0.25 * self.P.to(u.day)
        self.t = ((np.linspace(0, self.cycles, N)) * self.P.to(u.day) + t_offset).flatten()

        if self.circ:
            self.r = np.ones(self.t.size) * a
            self.f = (((2 * np.pi) / self.P) * self.t).to(u.dimensionless_unscaled) * u.rad
        else:
            self.ecc_anomaly(self.t)
            Es = self.Es
            self.f = 2 * np.arctan2(np.sqrt(1 + self.e) * np.sin(Es / 2), np.sqrt(1 - self.e) * np.cos(Es / 2))  
            self.r = (a * (1 - self.e)) / (1 + self.e * np.cos(self.f))

        self.k = self.R_l / self.R_star
            
        self.cartesian_coords()
        self.projected_separation()
        self.einstein_radius()
        self.overlapping_area()
        self.alpha()
        self.amplification()
        self.geometric_flux()

    def orbital_period(self):
        Mlens = self.M_l 
        Msource = self.M_s
        a = self.a

        self.P = ((2*np.pi) * np.sqrt(a**3 / (G*(Mlens+Msource)))).to(u.yr)

    def radial_velocity(self, r):
        return ((2 * np.pi) / self.P) * r
    
    def cartesian_coords(self):
        r = self.r
        i = self.inc
        omega = self.omega
        f = self.f

        self.X = -r * np.cos(omega + f)
        self.Y = -r * np.sin(omega + f) * np.cos(i)
        self.Z = r * np.sin(omega + f) * np.sin(i)

    def projected_separation(self):
        a = self.a
        e = self.e
        i = self.inc
        X = self.X
        Y = self.Y
        rstar = self.R_star
        omega = self.omega

        self.r_sky = np.sqrt(X**2 + Y**2)
        # self.r_sky = r * np.sqrt(1-np.sin(omega + f)**2 * np.sin(i)**2)

        self.b = a * np.cos(i)
        self.b_tra = (self.b / rstar) * ((1-e**2) / (1 + e * np.sin(omega)))
        self.b_occ = (self.b / rstar) * ((1-e**2) / (1 - e * np.sin(omega)))

    def eclipse_duration(self):
        # equations from (Winn 2010)
        P = self.P
        R_star = self.R_star
        a = self.a
        k = self.k
        b_tra = self.b_tra
        b_oc = self.b_occ
        i = self.inc
        e = self.e
        circ = self.circ
        omega = self.omega
        ing_eg_ratio = e * np.cos(omega) * (R_star / a)**3

        factor_tra = factor_oc = 1.0
        if not circ: 
            factor_tra = np.sqrt(1-e**2) / (1+e*np.sin(omega))
            factor_oc = np.sqrt(1-e**2) / (1-e*np.sin(omega))

        # PRIMARY ECLIPSE
        self.tra_tot = (P / np.pi) * np.arcsin(((R_star) / a) * (np.sqrt((1+k)**2 - b_tra**2)) / np.sin(i)) * factor_tra
        self.tra_full = (P / np.pi) * np.arcsin(((R_star) / a) * (np.sqrt((1-k)**2 - b_tra**2)) / np.sin(i)) * factor_tra
        self.tra_t0 = ((R_star * P) / (np.pi * a)) * factor_tra 

        tau = self.tra_tot - self.tra_full
        self.tra_ingress = tau * (1 + ing_eg_ratio * (1-b_tra)**(3/2)) / 2
        self.tra_egress = tau * (1 - ing_eg_ratio * (1-b_tra)**(3/2)) / 2

        # SECONDARY ECLIPSE
        self.oc_tot = (P / np.pi) * np.arcsin(((R_star) / a) * (np.sqrt((1+k)**2 - b_oc**2)) / np.sin(i)) * factor_oc
        self.oc_full = (P / np.pi) * np.arcsin(((R_star) / a) * (np.sqrt((1-k)**2 - b_oc**2)) / np.sin(i)) * factor_oc
        self.oc_t0 = ((R_star * P) / (np.pi * a)) * factor_oc 

        tau = self.oc_tot - self.oc_full
        self.oc_ingress = tau * (1 + ing_eg_ratio * (1-b_oc)**(3/2)) / 2
        self.oc_egress = tau * (1 - ing_eg_ratio * (1-b_oc)**(3/2)) / 2

    def overlapping_area(self):
        d = self.r_sky.to(u.m)
        rstar = (np.ones(d.size) * self.R_star).to(u.m)
        rlens = (np.ones(d.size) * self.R_l).to(u.m)

        arccos1 = np.arccos(np.clip(((rstar**2 - rlens**2 + d**2) / (2 * d * rstar)).decompose().value, -1, 1))
        arccos2 = np.arccos(np.clip(((rlens**2 - rstar**2 + d**2) / (2 * d * rlens)).decompose().value, -1, 1))

        overlap = (
            rstar**2 * arccos1
            + rlens**2 * arccos2
            - 0.5 * np.sqrt((-d + rlens + rstar) * (d + rlens - rstar) * (d - rlens + rstar) * (d + rlens + rstar))).to(u.m**2) 
        
        max_overlap = np.minimum(self.A_star.to(u.m**2).value, self.A_l.to(u.m**2).value)
        self.overlap = np.minimum(overlap.value, max_overlap) * u.m**2

    def alpha(self):
        Z = self.Z
        d = self.r_sky.to(u.m)
        rstar = (np.ones(d.size) * self.R_star).to(u.m)
        rlens = (np.ones(d.size) * self.R_l).to(u.m)
        star_area = self.A_star.to(u.m**2)
        lens_area = self.A_l.to(u.m**2)
        overlap = self.overlap

        transits = np.zeros(Z.size)
        occultations = np.zeros(Z.size)

        tra_mask = Z.value > 0
        occ_mask = Z.value < 0

        transits[tra_mask] = (overlap / lens_area).decompose().value[tra_mask]
        occultations[occ_mask] = (overlap / lens_area).decompose().value[occ_mask]

        transits[d >= rstar + rlens] = 0
        occultations[d >= rstar + rlens] = 0

        transits[tra_mask & ((d + rlens) <= rstar)] = 1.0
        occultations[occ_mask & ((d + rlens) <= rstar)] = 1.0

        transits[tra_mask & ((d + rstar) <= rlens)] = (star_area / lens_area).decompose().value
        occultations[occ_mask & ((d + rstar) <= rlens)] = (star_area / lens_area).decompose().value

        self.alpha_tra = transits
        self.alpha_occ = occultations

    def geometric_flux(self):
        t = self.t
        k = self.k
        F_lens = self.F_lens
        F_star = self.F_star 
        u1, u2 = self.limb_darkening_coeffs_star
        u3, u4 = self.limb_darkening_coeffs_wd
        ld_tra, ld_occ = self.geometric_limb_darkening(u1, u2, u3, u4)  
        A = self.amp
        Z = self.Z

        alpha_tra = self.alpha_tra
        alpha_occ = self.alpha_occ

        # outside eclipse
        flux = np.ones(t.size) * (F_star + F_lens)

        # amplification
        transit_mask = Z.value > 0
        flux[transit_mask] += A[transit_mask] * F_star * ld_tra * (1 - k**2 * alpha_tra[transit_mask])

        # occult
        occ_mask = Z.value < 0
        flux[occ_mask] -= alpha_occ[occ_mask] * F_lens * ld_occ

        self.geo_flux = flux
        self.base_flux = np.ones(t.size) * (F_star + F_lens)

    def geometric_limb_darkening(self, u1, u2, u3, u4):
        Z = self.Z
        Xt = (self.X / self.R_star).decompose().value
        Yt = (self.Y / self.R_star).decompose().value

        mu = np.sqrt(1 - np.clip(Xt**2 + Yt**2, 0, 1))

        transit_mask = Z.value > 0
        occult_mask = Z.value < 0

        # but actually is just proportional to, need elliptic integrals to be accurate
        I_int_tra = (1 - u1/3 - u2/6)  # integral of I
        I_int_occ = (1 - u3/3 - u4/6) 

        ld_cont_tra = (1 - u1 * (1-mu) - u2 * (1-mu)**2) / I_int_tra # I / integral of I
        ld_occ = (1 - u3 * (1-mu) - u4 * (1-mu)**2) / I_int_occ

        return ld_cont_tra[transit_mask], ld_occ[occult_mask]
    
    def amplification(self):
        rstar = (np.ones(self.t.size) * self.R_star).to(u.m)
        rlens = (np.ones(self.t.size) * self.R_l).to(u.m)
        r = self.r_sky.to(u.m)
        rE = self.r_E.to(u.m)

        r_in  = rstar - rE
        r_out = rstar + rE

        A = np.zeros(self.t.size)
        case1 = r.value < r_in.value
        case2 = (r.value >= r_in.value) & (r.value <= r_out.value)

        A[case1] = (2*rE[case1]**2 - rlens[case1]**2) / rstar[case1]**2
        A[case2] = ((2*rE[case2]**2 - rlens[case2]**2) / rstar[case2]**2) * ((r_out[case2] - r[case2]) / (2 * rE[case2]))
        self.amp = np.clip(A, 0, None)

    def einstein_radius(self):
        l = self.d
        ls = self.r
        s = self.d + self.r

        self.r_E = (np.sqrt(4 * G * self.M_l * ls / ((c**2) * l * s)) * l).to(u.m)
    
    def ecc_anomaly(self, ti):
        e = self.e
        P = self.P

        M = ((2 * np.pi / P) * ti).to(u.dimensionless_unscaled).value

        E = M.copy()
        for _ in range(50):
            dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
            E -= dE
            if np.max(np.abs(dE)) < 1e-10:
                break

        self.Es = E * u.rad

    def light_curve_plotter(self, name):
        fig = plt.figure(figsize=(20, 8)) 
        gs = gridspec.GridSpec(1, 2)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])

        time = (self.t).to(u.day).value
        frac = self.geo_flux / self.base_flux

        transit_mask = frac > 1
        occ_mask = frac < 1

        tra_mid = np.median(time[transit_mask])
        occ_mid = np.median(time[occ_mask])

        dt = np.abs(time[1] - time[0])
        change = 0.007 * self.N * dt

        plt.suptitle(f"{name} Light Curve Flares")

        ax0.plot(time, frac)
        ax0.set_xlabel("Time (days)")
        ax0.set_ylabel("Relative Flux")
        ax0.set_title("Primary")
        ax0.set_xlim(tra_mid - change, tra_mid + change)
        ax0.set_ylim(0.9985, 1.0015)

        ax1.plot(time, frac)
        ax1.set_xlabel("Time (days)")
        ax1.set_ylabel("Relative Flux")
        ax1.set_title("Secondary")
        ax1.set_xlim(occ_mid - change, occ_mid + change)
        ax1.set_ylim(0.9985, 1.0015)

        plt.tight_layout()
        plt.show()

    #### GR Ray-Tracing taken from TDE project #####
    def mom_kerr_analytic(self, rp, a):
        """
        Computes Lz for Kerr orbits
        """
        num = -2 * a * rp 
        num += np.sqrt(2) * np.sqrt(rp**3 * (a**2 + (-2 + rp) * rp))
        
        return num / ((-2 + rp) * rp)

    def geodesic_kerr(self, τ, y):
        """
        Computes outgoing geodesics for equatorial parabolic prograde Kerr orbits.
        """
        t, r, phi, psi = y

        theta = np.pi/2
        q = 0 # not dealing with this rn
        E = self.OrbitEnergy # can be calculated from eccentric anomaly
        rp = np.min(self.r)
        a = self.spin

        Lz = self.mom_kerr_analytic(rp, a)

        p = E * (r**2 + a**2) - a * Lz
        rho = r**2 + a**2 * np.cos(theta)**2
        delta = r**2 - 2 * r + a**2

        dt_dτ = (-a * (a * E * np.sin(theta)**2 - Lz) + ((r**2 + a**2) / delta) * p) / rho
        dr_dτ = (np.sqrt(p**2 - delta * (r**2 + (Lz - a * E)**2 + q))) / rho
        dφ_dτ = (-(a * E  - (Lz/np.sin(theta)**2)) + (a/delta) * p) / rho 
        # dθ_dτ = (np.sign(τ) * np.sqrt(q - np.cos(theta)**2 * (a**2 * (1 - E**2) + (Lz**2/np.sin(theta)**2)))) / rho 

        dpsi_dτ = np.abs(a - Lz) * (((r**2 + a**2) - a * Lz) / ((a - Lz)**2 + r**2) + a * (Lz - a) / (a - Lz)**2) / r**2

        return [dt_dτ, dr_dτ, dφ_dτ, dpsi_dτ]

    def _compute_rel_orbit(self):
        """
        For Kerr prograde orbits, compute the orbital phase (Phi), radius (R), and their time derivatives
        for a parabolic trajectory.
        """
        rp = np.min(self.r)
        ε = 1e-6
        tau_max = np.max(self.t)

        y0_out = [0.0, rp + ε, 0.0, 0.0]
        sol_in = sol_out = cp.integrate.solve_ivp(
            self.geodesic_kerr,
            (0, tau_max),
            y0_out,
            t_eval=self.t[self.t >= 0]
        )   

        # connect inbound and outbound leg
        tau = np.hstack([-sol_in.t[::-1], sol_out.t[1:]])
        time = np.hstack([-sol_in.y[0][::-1], sol_out.y[0][1:]])
        radius = np.hstack([sol_in.y[1][::-1], sol_out.y[1][1:]])
        phi = np.hstack([-sol_in.y[2][::-1], sol_out.y[2][1:]])
        psi = np.hstack([-sol_in.y[3][::-1], sol_out.y[3][1:]])
        psi += phi[0] - psi[0]

        # R, phi, t, and psi
        self.R = cp.interpolate.interp1d(tau, radius, kind='cubic', fill_value='extrapolate')
        self.Phi = cp.interpolate.interp1d(tau, phi, kind='cubic', fill_value='extrapolate')
        self.obs_t = cp.interpolate.interp1d(tau, time, kind='cubic', fill_value='extrapolate')
        self.Psi = cp.interpolate.interp1d(tau, psi, kind='cubic', fill_value='extrapolate')

        # calculate Rdot, phidot, and psidot
        a = self.spin
        Lz = self.mom_kerr_analytic(rp, a)
        theta = np.pi/2
        E = self.OrbitEnergy
        q = 0

        p = (radius**2 + a**2) - a * Lz
        rho = radius**2 + a**2 * np.cos(theta)**2
        delta = radius**2 - 2 * radius + a**2

        tdot = (-a * (a * E * np.sin(theta)**2 - Lz) + ((radius**2 + a**2) / delta) * p) / rho
        Rdot = (np.sqrt(p**2 - delta * (radius**2 + (Lz - a * E)**2 + q))) / rho
        phidot = (-(a * E  - (Lz/np.sin(theta)**2)) + (a/delta) * p) / rho  
        psidot = np.abs(a - Lz) * (((radius**2 + a**2) - a * Lz) / ((a - Lz)**2 + radius**2) + a * (Lz - a) / (a - Lz)**2) / radius**2

        self.tdot = cp.interpolate.interp1d(tau, tdot, kind='cubic', fill_value='extrapolate')
        self.Rdot = cp.interpolate.interp1d(tau, Rdot, kind='cubic', fill_value='extrapolate')
        self.phidot = cp.interpolate.interp1d(tau, phidot, kind='cubic', fill_value='extrapolate')
        self.psidot = cp.interpolate.interp1d(tau, psidot, kind='cubic', fill_value='extrapolate')

    def kerr_metric(self, r, theta):
        # Initialize arrays: G[a,b,i]
            G = np.zeros((4, 4))
            a = self.spin
            s, c = np.sin(theta), np.cos(theta)
            
            delta = r**2 + a**2 - 2 * r
            sigma = r**2 + a**2 * c**2
            A = (r**2 + a**2)**2 - sigma * a**2 * s**2

            G[0,0] = -(1 - ((2*r)/sigma))
            G[1,1] = sigma / delta
            G[2,2] = sigma
            G[3,3] = (A / sigma) * s**2

            G[0,3] = G[3,0] = -(2 * a * r * s**2) / sigma

            self.G = G

    # Will probably need to change
    def christoffel_symb(self, r, theta):
        C = np.zeros((4, 4, 4))
        theta = np.pi/2
        s, c = np.sin(theta), np.cos(theta)
        a = self.a

        delta = r**2 + a**2 - 2 * r
        sigma = r**2 + a**2 * c**2
        A = (r**2 + a**2)**2 - sigma * a**2 * s**2

        C[1,0,0] = (delta / sigma**3) * (2 * r**2 - sigma)
        C[2,0,0] = -(2 * a**2 * r * s * c) / sigma**3

        C[1,1,1] = (r / sigma) - ((r - 1) / delta)
        C[2,1,1] = (a**2 * s * c) / (sigma * delta)

        C[1,2,2] = -(r * delta) / sigma
        C[2,2,2] = -(a**2 * s * c) / sigma

        C[1,3,3] = -((delta * s**2)/sigma) * (r - (((a**2 * s**2) / sigma**2) * (2*r**2 - sigma)))
        C[2,3,3] = -((s*c)/sigma**3) * ((r**2 + a**2) * A - sigma * delta * a**2 * s**2)

        C[0,0,1] = C[0,1,0] = ((r**2 + a**2) / (sigma**2 * delta)) * (2*r**2 - sigma)
        C[3,0,1] = C[3,1,0] = (a / (sigma**2 * delta)) * (2*r**2 - sigma)

        C[0,0,2] = C[0,2,0] = -(2 * a**2 * r * s * c) / sigma**2
        C[3,0,2] = C[3,2,0] = -(2 * a * r * c) / (sigma**2 * s)

        C[1,0,3] = C[1,3,0] = -((a * delta * s**2) / sigma**3) * (2*r**2 - sigma)
        C[2,0,3] = C[2,3,0] = (2 * a * r * (r**2 + a**2) * s * c) / sigma**3

        C[1,1,2] = C[1,2,1] = -(a**2 * s * c) / sigma
        C[2,1,2] = C[2,2,1] = r / sigma

        C[0,1,3] = C[0,3,1] = -((a * s**2) / (sigma * delta)) * (((2*r**2)/sigma) * (r**2 + a**2) + r**2 - a**2)
        C[3,1,3] = C[3,3,1] = (r / sigma) - (((a**2 * s**2) / (sigma * delta)) * (r - 1 + ((2*r**2)/sigma)))

        C[0,2,3] = C[0,3,2] = (2 * a**3 * r * s**3 * c) / sigma**2
        C[3,2,3] = C[3,3,2] = (c/s) * (1 + ((2*a**2 * r * s**2)/sigma**2))

        self.C = C

    def rel_lambda(self, tdot, r, rdot, theta, thetadot, phidot):
        # Initialize arrays: LAMBDA[a,b,i]
        LAMBDA = np.zeros((4, 4))

        # Precompute cos(Φ) and sin(Φ)
        a = self.a 
        q = self.Carter
        c, s = np.cos(theta), np.sin(theta)
        Lz = self.mom_kerr_analytic(self.Rp, self.a)
        K = q + (Lz - a)**2
        sigma = r**2 + a**2 * np.cos(theta)**2
        delta = r**2 + a**2 - 2 * r

        alpha = np.sqrt((K - a**2 * np.cos(theta)**2) / (r**2 + K))
        beta = 1 / alpha

        LAMBDA[0,0] = tdot
        LAMBDA[1,0] = rdot
        LAMBDA[2,0] = thetadot
        LAMBDA[3,0] = phidot
        
        LAMBDA[0,1] = (1 / np.sqrt(K)) * ((alpha * (r**2 + a**2) * r * rdot) / delta + (beta * a**2 * s * c * thetadot))
        LAMBDA[1,1] = ((alpha * r) / (sigma * np.sqrt(K))) * ((r**2 + a**2) - a * Lz)
        LAMBDA[2,1] = ((beta * a * c) / (sigma * np.sqrt(K))) * (a * s - (Lz / s))
        LAMBDA[3,1] = (a / np.sqrt(K)) * ((alpha * r * rdot) / delta + ((beta * c * thetadot) / s))

        LAMBDA[0,2] = (a / np.sqrt(K)) * (((r**2 + a**2) * c * rdot) / delta - r * s * thetadot)
        LAMBDA[1,2] = ((a * c) / (sigma * np.sqrt(K))) * ((r**2 + a**2) - a*Lz)
        LAMBDA[2,2] = -(r / (sigma * np.sqrt(K))) * (a * s - (Lz / s))
        LAMBDA[3,2] = (1 / np.sqrt(K)) * ((a**2 * c * rdot) / delta - ((r * thetadot) / s))

        LAMBDA[0,3] = alpha * ((r**2 + a**2) / (sigma * delta)) * ((r**2 + a**2) - a * Lz) - beta * (a / sigma) * (a * s**2 - Lz)
        LAMBDA[1,3] = alpha * rdot
        LAMBDA[2,3] = beta * thetadot
        LAMBDA[3,3] = ((alpha * a) / (sigma * delta)) * ((r**2 + a**2) - a * Lz) - (beta / sigma) * (a - (Lz / s**2))

        self.LAMBDA = LAMBDA

    def l_null(self, r, theta):
        a = self.a 
        delta = r**2 + a**2 - 2 * r

        dt = (r**2 + a**2) / delta
        dr = 1.0
        dtheta = 0.0
        dphi = a / delta

        lalpha = np.array([dt, dr, dtheta, dphi]) #l

        self.l_null = lalpha

    def n_null(self, r, theta):
        a = self.a 
        sigma = r**2 + a**2 * np.cos(theta)**2
        delta = r**2 + a**2 - 2 * r

        dt = (r**2 + a**2) / (2*sigma)
        dr = -delta / (2 * sigma)
        dtheta = 0.0
        dphi = a / (2*sigma)

        n = np.array([dt, dr, dtheta, dphi]) #n

        self.n_null = n

    def partials_table(self, r):
        a = self.a
        delta = r**2 + a**2 - 2 * r

        table = np.zeros((4,4,4))

        table[1,0,0] = -(((-2 + 2 * r) * (a**2 + r**2)**2) / (2 * delta**2)) + ((2 * r * (a**2 + r**2)) / delta)
        table[1,0,1] = -r
        table[1,0,3] = table[1,3,0] = -(a * ((-2 + 2 * r) * (a**2 + r**2)**2) / (2 * delta**2)) + ((a * r) / delta)

        table[1,1,0] = r
        table[1,1,1] = 0.5 * (2 - 2 * r)

        table[1,3,3] = -((a**2 * (2 * r - 2))/ (2 * delta**2))

        self.partials_table = table