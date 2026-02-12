import numpy as np
import scipy as cp
import batman
from scipy import integrate
import matplotlib.pyplot as plt
from astropy.constants import G, c, M_sun
from astropy import units as u

class WDBinary2:
    """
    Simulates the microlensing effects of a white dwarf lens
    orbiting a luminous star companion.
    """
    def __init__(self, mass_wd, r_wd, l_wd, mass_star, r_star, l_star, ecc, a, d, omega=np.pi, inc=90*u.deg, period=0.0*u.day, limb_darkening=[0.2, 0.3], N=1000):
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
        self.omega = omega * u.rad
        self.limb_darkening_coeffs = limb_darkening

        self.circ = (self.e == 0)

        if (period == 0.0):
            self.orbital_period()
        else:
            self.P = (period).to(u.yr)

        self.t = np.linspace(0, self.P.value, N) * u.yr

        if self.circ:
            self.r = a
            self.f = (((2 * np.pi) / self.P) * self.t).to(u.dimensionless_unscaled) * u.rad
        else:
            self.ecc_anomaly(self.t)
            Es = self.Es
            self.r = a * (1 - self.e*np.cos(Es))
            self.f = 2 * np.arctan2(np.sqrt(1 + self.e) * np.sin(Es / 2), np.sqrt(1 - self.e) * np.cos(Es / 2))  
            
        self.projected_separation()
        self.einstein_radius()
        self.limb_darkening()
        self.magnification()

    def orbital_period(self):
        Mlens = self.M_l 
        Msource = self.M_s
        a = self.a

        self.P = ((2*np.pi) * np.sqrt(a**3 / (G*(Mlens+Msource)))).to(u.yr)

    def projected_separation(self):
        a = self.a
        i = self.inc
        f = self.f
        r = self.r

        self.sep = r * np.sqrt(np.cos(f)**2 + (np.sin(f) * np.cos(i))**2)
        self.b = r * np.abs(np.sin(f)) * np.cos(i)

    def einstein_radius(self):
        l = self.d
        ls = self.r
        s = self.d + self.r

        self.r_E = np.sqrt(4 * G * self.M_l * ls / ((c**2) * l * s)) * l

    def check_eclipses(self):
        R_star = self.R_star
        R_wd = self.R_l
        
        f = self.f
        b = self.b
        r = self.r
        i = self.inc
        z = -r * np.sin(f) * np.sin(i)
        
        eclipse_possible = b < (R_star + R_wd)
        self.primary_eclipse = eclipse_possible & (z > 0)
        self.secondary_eclipse = eclipse_possible & (z < 0)
        
        return self.primary_eclipse, self.secondary_eclipse

    def magnification(self):
        r_E = self.r_E
        b = self.b 

        u = b / r_E
        self.A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    
    def max_magnification(self):
        return (1 + 2 * (self.r_E / self.R_star)**2).decompose().value

    def limb_darkening(self):
        P = self.P
        R_star = self.R_star
        R_l = self.R_l
        a = self.a
        i = self.inc
        e = self.e
        omega = self.omega

        L_total = self.L_star + self.L_wd
        self.f_star = (self.L_star / L_total).decompose().value
        self.f_wd = (self.L_wd / L_total).decompose().value
        
        # PRIMARY ECLIPSE
        params_primary = batman.TransitParams()
        params_primary.t0 = 0.0 
        params_primary.per = P.to(u.day).value
        params_primary.rp = (R_l / R_star).decompose().value  
        params_primary.a = (a / R_star).decompose().value 
        params_primary.inc = np.degrees(i).value
        params_primary.ecc = e
        params_primary.w = np.degrees(omega).value
        params_primary.u = self.limb_darkening_coeffs 
        params_primary.limb_dark = "quadratic" #I = 1 - u1 * (1 - mu) - u2 * (1 - mu)**2
        
        t = self.t
        m_primary = batman.TransitModel(params_primary, t)
        flux_primary = m_primary.light_curve(params_primary)
        
        # SECONDARY ECLIPSE
        params_secondary = batman.TransitParams()
        params_secondary.t0 = P.to(u.day).value / 2.0
        params_secondary.per = P.to(u.day).value
        params_secondary.rp = (R_star / R_l).decompose().value  
        params_secondary.a = (self.a / R_l).decompose().value
        params_secondary.inc = np.degrees(i).value 
        params_secondary.ecc = e
        params_secondary.w = np.degrees(omega).value
        params_secondary.u = [0.0, 0.0]  # WD has no limb darkening
        params_secondary.limb_dark = "quadratic"
        
        m_secondary = batman.TransitModel(params_secondary, t)
        flux_secondary = m_secondary.light_curve(params_secondary)

        self.flux_eclipse_star = flux_primary  #Fractional flux from star
        self.flux_eclipse_wd = flux_secondary  #Fractional flux from WD
        
        self.flux_eclipse = self.f_star * flux_primary + self.f_wd * flux_secondary
        
        return self.flux_eclipse
    
    def total_flux(self):
        flux_star_total = self.f_star * self.A * self.flux_eclipse_star
        flux_wd_total = self.f_wd * self.flux_eclipse_wd
        
        self.total_flux_normalized = flux_star_total + flux_wd_total
        
        return self.total_flux_normalized

    def plot_light_curve(self):
        self.total_flux()
        
        t = self.t
        P = self.P
        
        # Combined
        plt.plot(t/P, self.total_flux_normalized)
        plt.xlabel("Orbits")
        plt.ylabel("Total Normalized Flux")
        plt.title("Light Curve")
        plt.grid(True)
        
        plt.show()

    def animate_orbit(self):
        pass

    # anomaly calculations from AST303
    def mean_anomaly_noM(self, P, t):
        return ((2 * np.pi * np.mod(t, P)) / P)

    def mean_anomaly2(self, P, t, m0):
        return np.mod((((2 * np.pi * t) / P) - m0), 2 * np.pi)

    def root(self, M, e):
        M = np.mod(M, 2*np.pi)

        if (M < 2 * np.pi and M >= np.pi):
            return M - (e / 2)
        elif (M < np.pi and M >= 2):
            return M + ((e * (np.pi - M)) / (1 + e))
        elif (M < 2 and M >= 0.25):
            return M + e
        elif (M < 0.25 and M >= 0):
            return M + ((e*np.sin(M)) / (1 - np.sin((M+e)) + np.sin(M)))
        else:
            print("Invalid M")
            return NaN
        
    def true_anomaly(E, e):
        # theta-prime function
        theta_prime_mcmc = lambda E: np.arccos((np.cos(E) - e) / (1 - e * np.cos(E)))

        theta = []
        for i in range(len(E)):
            if (E[i] <= np.pi):
                theta.append(theta_prime_mcmc(E[i]))
            else:
                theta.append((2 * np.pi) - theta_prime_mcmc(E[i]))

        return theta
    
    def ecc_anomaly(self, ti):
        e = self.e
        P = self.P
        # get E(t) values
        Es_i = [] # E(ti)

        for i in range(len(ti)):
            m_ti = self.mean_anomaly_noM(P, ti[i]).value
            f = lambda x: x - e * np.sin(x) - m_ti

            res_i = cp.optimize.fsolve(func=f, x0=self.root(m_ti,e))
    
            Es_i.append(res_i[0])

        self.Es = Es_i * u.rad