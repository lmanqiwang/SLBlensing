import numpy as np
import scipy as cp
from scipy import integrate
import matplotlib.pyplot as plt
from astropy.constants import G, c, M_sun, sigma_sb, h, k_B
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib import gridspec

class SLBlensing:
    def __init__(self, mass_lens, r_lens, l_lens, T_eff_lens, mass_star, r_star, l_star, T_eff_star, ecc, a, d, omega=0 * u.deg, inc=90*u.deg, period=0.0*u.day, cycles=1.0, limb_darkening_l=[0.0, 0.0], limb_darkening_star=[0.0, 0.0], gravity_darkening = 0.0, offset=False, freq=None, N=1000):
        self.M_l = mass_lens * M_sun
        self.R_l = (r_lens * u.Rsun).to(u.m)
        self.M_s = mass_star * M_sun
        self.R_star = (r_star * u.Rsun).to(u.m)
        self.L_star = l_star * u.Lsun
        self.L_l = l_lens * u.Lsun
        self.N = N
        self.nu = freq
        self.T_eff_star = T_eff_star * u.K
        self.T_eff_lens = T_eff_lens * u.K

        self.A_star = np.pi * self.R_star**2
        self.A_l = np.pi * self.R_l**2

        self.e = ecc
        self.a = a.to(u.m)
        self.inc = inc.to(u.rad)
        self.d = d.to(u.m)
        self.omega = omega.to(u.rad)      #argument of perhelion
        self.limb_darkening_coeffs_l = limb_darkening_l
        self.limb_darkening_coeffs_star = limb_darkening_star
        self.cycles = cycles
        self.circ = (self.e == 0)

        if (period == 0.0):
            self.orbital_period()
        else:
            self.P = (period).to(u.yr)

        if offset:
            # offset calc
            f_conj = np.pi/2 - self.omega.to(u.rad).value

            E_conj = 2 * np.arctan2(np.sqrt(1 - self.e) * np.sin(f_conj / 2),
                                    np.sqrt(1 + self.e) * np.cos(f_conj / 2))

            M_conj = E_conj - self.e * np.sin(E_conj)
            M_conj = M_conj % (2 * np.pi)

            t_conj = (M_conj / (2 * np.pi)) * self.P.to(u.day)
            t_offset = -t_conj + 0.25 * self.P.to(u.day)
            self.t = ((np.linspace(0, self.cycles, N)) * self.P.to(u.day) + t_offset).flatten()
        else:
            self.t = ((np.linspace(0, self.cycles, N)) * self.P.to(u.day)).flatten()

        if self.circ:
            self.r = np.ones(self.t.size) * a
            self.f = (((2 * np.pi) / self.P) * self.t).to(u.dimensionless_unscaled) * u.rad
        else:
            self.ecc_anomaly(self.t)
            Es = self.Es
            self.f = 2 * np.arctan2(np.sqrt(1 + self.e) * np.sin(Es / 2), np.sqrt(1 - self.e) * np.cos(Es / 2))  
            self.r = (a * (1 - self.e)) / (1 + self.e * np.cos(self.f))

        self.k = self.R_l / self.R_star
        self.F_bol_star = (self.L_star / (4 * np.pi * self.d**2)).to(u.W / u.m**2)
        self.F_bol_lens = (self.L_l / (4 * np.pi * self.d**2)).to(u.W / u.m**2)

        if (self.nu is not None):
            u1 = self.limb_darkening_coeffs_star[0]
            phi = self.f + np.ones(self.f.size) * self.omega

            # bandpass photometry flux, from blackbody approx
            if self.T_eff_lens.value < 1.0: 
                self.F_nu_lens = 0.0 * (u.W / u.m**2 / u.Hz) # BH case
            else:
                self.F_nu_lens = (((np.pi * (self.R_l / self.d)**2 * 2 * h * self.nu**3) / c**2) * (1 / (np.exp(((h * self.nu)/(k_B * self.T_eff_lens)).to(u.dimensionless_unscaled).value) - 1))).to(u.W / u.m**2 / u.Hz)
            self.F_nu_star = (((np.pi * (self.R_star / self.d)**2 * 2 * h * self.nu**3) / c**2) * (1 / (np.exp(((h * self.nu)/ (k_B * self.T_eff_star)).to(u.dimensionless_unscaled).value) - 1))).to(u.W / u.m**2 / u.Hz)
            
            # ellipoisdial variation, we assume compact object is not deformed by MS star
            self.ellip_var = (((3/20) * (((15 + u1) * (1 + gravity_darkening)) / (3 - u1)) * (self.M_l / self.M_s) * (self.R_star / self.r)**3 * np.sin(self.inc)**2) * np.cos(2 * phi)) * self.F_nu_star
            
        self.cartesian_coords()
        self.projected_separation()
        self.einstein_radius()
        self.rho = self.r_E / self.R_star 

        # geometric coverage
        self.overlapping_area()
        self.alpha() 
        self.velocity_los()
    
        # lensing
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

    def velocity_los(self):
        i = self.inc
        a = self.a
        e = self.e
        comb_mass = self.M_l + self.M_s
        f = self.f
        omega = self.omega

        p = a * (1-e**2)
        K = np.sqrt((G * comb_mass) / p) * np.sin(i)

        v_los = K * (np.cos(omega + f) + e * np.cos(omega))

        self.v_star_los = v_los * self.M_l / comb_mass
        self.v_lens_los = v_los * self.M_s / comb_mass

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
        self.tra_tot = ((P / np.pi) * np.arcsin(((R_star) / a) * (np.sqrt((1+k)**2 - b_tra**2)) / np.sin(i)) * factor_tra) / u.rad
        self.tra_full = ((P / np.pi) * np.arcsin(((R_star) / a) * (np.sqrt((1-k)**2 - b_tra**2)) / np.sin(i)) * factor_tra) / u.rad
        self.tra_t0 = (((R_star * P) / (np.pi * a)) * factor_tra) / u.rad

        tau = self.tra_tot - self.tra_full
        self.tra_ingress = tau * (1 + ing_eg_ratio * (1-b_tra**2)**(3/2)) / 2
        self.tra_egress = tau * (1 - ing_eg_ratio * (1-b_tra**2)**(3/2)) / 2

        # SECONDARY ECLIPSE
        self.oc_tot = ((P / np.pi) * np.arcsin(((R_star) / a) * (np.sqrt((1+k)**2 - b_oc**2)) / np.sin(i)) * factor_oc) / u.rad
        self.oc_full = ((P / np.pi) * np.arcsin(((R_star) / a) * (np.sqrt((1-k)**2 - b_oc**2)) / np.sin(i)) * factor_oc) / u.rad
        self.oc_t0 = (((R_star * P) / (np.pi * a)) * factor_oc) / u.rad

        tau = self.oc_tot - self.oc_full
        self.oc_ingress = tau * (1 + ing_eg_ratio * (1-b_oc**2)**(3/2)) / 2
        self.oc_egress = tau * (1 - ing_eg_ratio * (1-b_oc**2)**(3/2)) / 2

    def overlapping_area(self):
        d = self.r_sky.to(u.m)
        rstar = (np.ones(d.size) * self.R_star).to(u.m)
        rlens = (np.ones(d.size) * self.R_l).to(u.m)

        # BH no physical disk, overlap is always zero
        if self.R_l.value == 0.0:
            self.overlap = np.zeros(self.t.size) * u.m**2
            return

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

        # BH no physical disk, overlap is always zero
        if self.R_l.value == 0.0:
            self.alpha_tra = np.zeros(self.Z.size)
            self.alpha_occ = np.zeros(self.Z.size)
            return

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
        F_lens = self.F_nu_lens if self.nu is not None else self.F_bol_lens
        F_star = self.F_nu_star if self.nu is not None else self.F_bol_star
        u1, u2 = self.limb_darkening_coeffs_star
        u3, u4 = self.limb_darkening_coeffs_l
        ld_tra, ld_occ = self.geometric_limb_darkening(u1, u2, u3, u4)  
        A = self.amp
        Z = self.Z

        alpha_tra = self.alpha_tra
        alpha_occ = self.alpha_occ

        # outside eclipse
        flux = np.ones(t.size) * (F_star)

        # amplification
        transit_mask = Z.value > 0
        flux[transit_mask] += A[transit_mask] * F_star * ld_tra * (1 - k**2 * alpha_tra[transit_mask])

        # occult
        occ_mask = Z.value < 0
        flux[occ_mask] -= alpha_occ[occ_mask] * F_lens * ld_occ

        self.geo_flux = flux
        self.base_flux = np.ones(t.size) * (F_star)

        if (self.nu is not None):
            Tstar = self.T_eff_star
            Tlens = self.T_eff_lens
            Rlens = self.R_l
            Rstar = self.R_star
            a = self.a

            x_star = ((h * self.nu) / (Tstar * k_B)).to(u.dimensionless_unscaled).value
            anu_star = 3 - (np.exp(x_star) / (np.exp(x_star) - 1)) * x_star

            if Tlens.value == 0:
                anu_lens = 0.0
                x_lens = 0.0
            else:
                x_lens = ((h * self.nu) / (Tlens * k_B)).to(u.dimensionless_unscaled).value
                anu_lens = 3 - (np.exp(x_lens) / (np.exp(x_lens) - 1)) * x_lens
                cos_theta = np.sin(self.inc) * np.sin(self.f + self.omega)

                # magnitude diff
                del_m = -2.5 * np.log10((F_lens / F_star).decompose().value)
                exp = np.power(10, -0.4 * del_m)

                # irradiation to leading order, only valid when lens is hotter than star
                if Tlens > Tstar: 
                    lumin_eff = (Tlens / Tstar)**4 * ((np.exp(x_lens) - 1) / (np.exp(x_star) - 1))
                    self.geo_flux += (5/6) * (np.log10(np.e) / np.pi) * (Rstar / a)**2 * (1/lumin_eff) * (1 / (1 + exp)) * F_star * cos_theta
                    self.geo_flux += (5/6) * (np.log10(np.e) / np.pi) * (Rlens / a)**2 * (lumin_eff) * (exp / (1 + exp)) * F_lens * cos_theta

            # doppler beaming
            self.geo_flux += (3 - anu_star) * (self.v_star_los / c).to(u.dimensionless_unscaled) * F_star
            self.geo_flux -= (3 - anu_lens) * (self.v_lens_los / c).to(u.dimensionless_unscaled) * F_lens

            # elliposidal variation
            self.geo_flux -= self.ellip_var

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

        A[case1] = (2*rE[case1]**2 - rlens[case1]**2) / rstar[case1]**2    # amp full einstein ring
        A[case2] = ((2*rE[case2]**2 - rlens[case2]**2) / rstar[case2]**2) * ((r_out[case2] - r[case2]) / (2 * rE[case2])) # amp partial eintstein ring
        self.amp = A

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

    def light_curve_plotter(self, name, n):
        fig = plt.figure(figsize=(24, 10)) 
        gs = gridspec.GridSpec(1, 2)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])

        time = (self.t).to(u.day).value
        frac = self.geo_flux / self.base_flux
        period = self.P.to(u.day).value

        first_cycle = time <= period
        wtime  = time[first_cycle]
        wfrac  = frac[first_cycle]      

        transit_mask = self.Z[first_cycle].value > 0
        occult_mask  = self.Z[first_cycle].value < 0

        tra_mid = np.median(wtime[np.where(wfrac == np.max(wfrac[transit_mask]))])
        occ_mid = np.median(wtime[np.where(wfrac == np.min(wfrac[occult_mask]))])

        dt = np.abs(time[1] - time[0])
        change = n * self.N * dt

        plt.suptitle(name)

        ax0.plot(time, frac)
        ax0.set_xlabel("Time (days)")
        ax0.set_ylabel("Normalized Flux")
        ax0.set_title("Lens Transit")
        ax0.set_xlim(tra_mid - change, tra_mid + change)
        ax0.set_ylim(1 - 0.2 * np.max(wfrac - 1))

        ax1.plot(time, frac)
        ax1.set_xlabel("Time (days)")
        ax1.set_title("Lens Occultation")
        ax1.set_xlim(occ_mid - change, occ_mid + change)
        ax1.set_ylim(np.min(wfrac[occult_mask]) - (1 - np.min(wfrac[occult_mask])) * (0.2), 1 + 0.2 * (1 - np.min(wfrac[occult_mask])))

        ax0.ticklabel_format(style='plain', axis='y', useOffset=False)
        ax1.ticklabel_format(style='plain', axis='y', useOffset=False)
        plt.tight_layout()
        plt.show()

    def transit_plotter(self, name, n):
        plt.figure(figsize=(15, 8)) 

        time = (self.t).to(u.day).value
        frac = self.geo_flux / self.base_flux
        period = self.P.to(u.day).value

        first_cycle = time <= period
        wtime  = time[first_cycle]
        wfrac  = frac[first_cycle]      

        transit_mask = self.Z[first_cycle].value > 0

        tra_mid = np.median(wtime[np.where(wfrac == np.max(wfrac[transit_mask]))])

        dt = np.abs(time[1] - time[0])
        change = n * self.N * dt

        plt.suptitle(name)

        plt.plot(time, frac)
        plt.xlabel("Time (days)")
        plt.ylabel("Normalized Flux")
        plt.title("Lens Transit")
        plt.xlim(tra_mid - change, tra_mid + change)
        plt.ylim(1 - 0.2 * np.max(wfrac - 1))
        plt.ticklabel_format(style='plain', axis='y', useOffset=False)

        plt.tight_layout()
        plt.show()
