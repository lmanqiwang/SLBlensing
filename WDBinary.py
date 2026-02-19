import numpy as np
import scipy as sp
from scipy import integrate, optimize
import matplotlib.pyplot as plt
from astropy.constants import G, c, M_sun
from astropy import units as u

class WDBinary:
    """
    Simulates the microlensing effects of a white dwarf lens
    orbiting a luminous star companion.
    """
    def __init__(self, mass_wd, r_wd, l_wd, mass_star, r_star, l_star, 
                 ecc, a, d, t, inc=90*u.deg, period=0.0, 
                 limb_darkening=[0.2, 0.3]):
        self.M_l = mass_wd * M_sun
        self.R_l = r_wd.to(u.m)
        self.M_s = mass_star * M_sun
        self.R_star = r_star.to(u.m)
        self.L_star = l_star * u.L_sun
        self.L_wd = l_wd * u.L_sun

        self.e = ecc
        self.a = a.to(u.m)
        self.inc = inc.to(u.rad)
        self.d = d.to(u.m)
        self.t = t
        
        self.u1 = limb_darkening[0]
        self.u2 = limb_darkening[1]

        self.circ = (self.e == 0)

        if period == 0.0:
            self._orbital_period()
        else:
            self.P = (period * u.day).to(u.yr)

        if not self.circ:
            self.ecc_anomaly(t)
            
        self.projected_separation()
        self.einstein_radius()
        self.compute_light_curve()

    def _orbital_period(self):
        """Calculate orbital period using Kepler's third law"""
        Mlens = self.M_l 
        Msource = self.M_s
        a = self.a
        self.P = ((2*np.pi) * np.sqrt(a**3 / (G*(Mlens+Msource)))).to(u.yr)

    def projected_separation(self):
        """Calculate projected separation between WD and star"""
        a = self.a
        circ = self.circ
        i = self.inc
        f = (((2 * np.pi) / self.P) * self.t).to(u.dimensionless_unscaled) * u.rad

        if circ:
            self.sep = a * np.abs(np.sin(f))
        else:
            e = self.e
            Es = self.Es
            r = a * (1 - e * np.cos(Es))  # orbital radius
            
            # True anomaly from eccentric anomaly
            true_f = 2 * np.arctan2(
                np.sqrt(1 + e) * np.sin(Es / 2),
                np.sqrt(1 - e) * np.cos(Es / 2)
            )            
            self.sep = r * np.abs(np.sin(true_f))

    def mean_anomaly_noM(self, P, t):
        return ((2 * np.pi * np.mod(t, P)) / P)

    def root(self, M, e):
        """Initial guess for eccentric anomaly"""
        M = np.mod(M, 2*np.pi)
        if M >= np.pi:
            return M - (e / 2)
        elif M >= 2:
            return M + ((e * (np.pi - M)) / (1 + e))
        elif M >= 0.25:
            return M + e
        else:
            return M + ((e*np.sin(M)) / (1 - np.sin((M+e)) + np.sin(M)))
    
    def ecc_anomaly(self, ti):
        """Solve Kepler's equation for eccentric anomaly"""
        e = self.e
        P = self.P
        Es_i = []

        for i in range(len(ti)):
            m_ti = self.mean_anomaly_noM(P, ti[i]).value
            f = lambda x: x - e * np.sin(x) - m_ti
            res_i = optimize.fsolve(func=f, x0=self.root(m_ti, e))
            Es_i.append(res_i[0])

        self.Es = Es_i * u.rad
    
    def einstein_radius(self):
        """Calculate Einstein radius for microlensing"""
        # For self-lensing, source is at distance d+a, lens at d
        d_l = self.d  # lens distance
        d_s = self.d + self.a  # source distance (approximately)
        d_ls = self.a  # lens-source distance
        
        # Einstein radius at lens plane
        self.r_E = np.sqrt(4 * G * self.M_l * d_ls * d_l / (c**2 * d_s))

    def compute_magnification_single_point(self, impact_param, source_radius):
        """
        Compute magnification for a point source at given impact parameter.
        
        For point source microlensing:
        u = impact_param / einstein_radius
        A = (u^2 + 2) / (u * sqrt(u^2 + 4))
        """
        u = impact_param / self.r_E
        u_val = u.value if hasattr(u, 'value') else u
        
        # Avoid division by zero
        u_val = np.maximum(u_val, 1e-10)
        
        mag = (u_val**2 + 2) / (u_val * np.sqrt(u_val**2 + 4))
        return mag

    def limb_darkening_profile(self, r):
        """
        Quadratic limb darkening law.
        Returns surface brightness at radius r from center of star.
        """
        R_star = self.R_star.value
        r_val = r if isinstance(r, (int, float, np.ndarray)) else r.value
        
        # mu = cos(theta) = sqrt(1 - (r/R)^2) for points on visible hemisphere
        mu_sq = 1 - (r_val / R_star)**2
        mu_sq = np.maximum(mu_sq, 0)  # avoid negative values
        mu = np.sqrt(mu_sq)
        
        # Quadratic limb darkening
        I = 1 - self.u1 * (1 - mu) - self.u2 * (1 - mu)**2
        return I

    def compute_magnification_extended(self, impact_param_center):
        """
        Compute magnification for extended source with limb darkening.
        Integrates over the stellar disk.
        """
        R_star = self.R_star
        b = impact_param_center  # impact parameter to star center
        
        # Number of annuli for integration
        n_annuli = 100
        r_points = np.linspace(0, R_star.value, n_annuli)
        
        magnifications = []
        
        for b_single in b:
            b_val = b_single.value if hasattr(b_single, 'value') else b_single
            
            # Integrate over stellar disk
            flux_magnified = 0
            flux_unmagnified = 0
            
            for i in range(len(r_points) - 1):
                r_mid = (r_points[i] + r_points[i+1]) / 2
                dr = r_points[i+1] - r_points[i]
                
                # Area element
                dA = 2 * np.pi * r_mid * dr
                
                # Surface brightness at this radius
                I_r = self.limb_darkening_profile(r_mid)
                
                # For each annulus, we need to consider points at different
                # impact parameters from the lens
                n_phi = 50
                phi_points = np.linspace(0, 2*np.pi, n_phi)
                
                mag_sum = 0
                for phi in phi_points:
                    # Impact parameter from lens to this point on star
                    x = b_val + r_mid * np.cos(phi)
                    y = r_mid * np.sin(phi)
                    impact = np.sqrt(x**2 + y**2)
                    
                    mag_point = self.compute_magnification_single_point(
                        impact * u.m, R_star
                    )
                    mag_sum += mag_point
                
                avg_mag = mag_sum / n_phi
                
                flux_magnified += I_r * dA * avg_mag
                flux_unmagnified += I_r * dA
            
            # Total magnification
            if flux_unmagnified > 0:
                total_mag = flux_magnified / flux_unmagnified
            else:
                total_mag = 1.0
                
            magnifications.append(total_mag)
        
        return np.array(magnifications)

    def compute_light_curve(self):
        """
        Compute the observed light curve including microlensing magnification
        and limb darkening effects.
        """
        # Compute magnification with extended source and limb darkening
        self.A = self.compute_magnification_extended(self.sep)
        
        # Baseline fluxes at Earth
        F_star_base = self.L_star / (4 * np.pi * self.d**2)
        F_wd_base = self.L_wd / (4 * np.pi * self.d**2)
        
        # Magnified star flux (WD doesn't significantly lens star's light back)
        F_star_obs = F_star_base * self.A
        
        # WD flux (not significantly affected by star lensing)
        F_wd_obs = F_wd_base
        
        # Total observed flux
        F_obs = F_star_obs + F_wd_obs
        F_baseline = F_star_base + F_wd_base
        
        # Fractional flux change
        self.frac_flux = ((F_obs / F_baseline) - 1).value

    def plot_light_curve(self, t):
        """Plot the microlensing light curve"""
        plt.figure(figsize=(10, 6))
        plt.plot((t / self.P).value, self.frac_flux, 'b-', linewidth=2)
        plt.xlabel("Orbital Phase (orbits)", fontsize=12)
        plt.ylabel("Fractional Flux Change", fontsize=12)
        plt.title("White Dwarf Self-Lensing Light Curve", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def plot_magnification(self, t):
        """Plot just the magnification factor"""
        plt.figure(figsize=(10, 6))
        plt.plot((t / self.P).value, self.A, 'r-', linewidth=2)
        plt.xlabel("Orbital Phase (orbits)", fontsize=12)
        plt.ylabel("Magnification", fontsize=12)
        plt.title("Magnification vs Time", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()