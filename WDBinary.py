import numpy as np
import scipy as cp
from scipy import integrate
import matplotlib.pyplot as plt
from astropy.constants import G, c, M_sun
from astropy import units as u

class WDBinary:
    """
    Simulates the microlensing effects of a white dwarf lens
    orbiting a luminous star companion.
    """
    def __init__(self, mass_wd, r_wd, l_wd, mass_star, r_star, l_star, ecc, a, d, t, inc=90*u.deg, period=0.0, limb_darkening=[0.2, 0.3]):
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

        if self.e == 0: 
            self.circ = True

        if (period == 0.0):
            self._orbital_period()
        else:
            self.P = (period * u.day).to(u.yr)

        if self.circ != True:
            self.ecc_anomaly(t)
            
        self.projected_separation()
        self.einstein_radius()
        self.magnification2()
        self.limb_darkening_quadratic(limb_darkening[0], limb_darkening[1])

    def _orbital_period(self):
        Mlens = self.M_l 
        Msource = self.M_s
        a = self.a

        self.P = ((2*np.pi) * np.sqrt(a**3 / (G*(Mlens+Msource)))).to(u.yr)

    def projected_separation(self):
        a = self.a
        circ = self.circ
        i = self.inc
        f = (((2 * np.pi) / self.P) * self.t).to(u.dimensionless_unscaled) * u.rad
        sqrt = np.sqrt(np.cos(f)**2 + (np.sin(f) * np.cos(i))**2)

        if circ == True:
           self.sep = a * sqrt
            
        else:
            e = self.e
            Es = self.Es
            
            sep = a * (1 - e*np.cos(Es)) * sqrt
            self.sep = sep

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
    
    def einstein_radius(self):
        l = self.d
        ls = self.a
        s = self.d + self.a

        self.r_E = np.sqrt(2 * G * self.M_l * ls / ((c**2) * l * s)) * l

    def magnification_perfect_align(self):
        inc = self.inc
        r_L = self.sep
        rE   = self.r_E
        r_s  = self.R_star

        r_in  = r_s - rE
        r_out = r_s + rE

        A = np.zeros_like(r_L.value, dtype=float)

        ingress = (r_L < r_in)
        if np.any(ingress):
            A[ingress] = 1 + 2*(rE/r_s)**2

        inside_mask = (r_L >= r_in) & (r_L < r_out)
        if np.any(inside_mask):
            A[inside_mask] = 1 + 2*(rE/r_s)**2 - (r_L[inside_mask]/r_s)**2

        egress = (r_L >= r_out)
        if np.any(egress):
            A[egress] = 0.0

        self.A = A

    def magnification2(self):
        r_L = self.sep
        rE   = self.r_E
        r_s  = self.R_star

        ro = 0.5 * ((r_s**2 + 4 * rE**2)**(1/2) + r_s)

        r_out = np.ones(r_L.size) * (ro)

        A = (r_out**2 - r_L**2) / r_s**2

        self.A = A

    def limb_darkening_quadratic(self, u1, u2):
        r = self.sep
        R_star = self.R_star
        I0 = self.L_star / (4 * np.pi * self.d**2)

        mu = np.sqrt(np.abs(1 - (r/R_star)**2))
        I = I0 * (1 - u1 * (1 - mu) - u2 * (1 - mu)**2)
        # I[r > R_star] = 0
        
        self.I = I

    def plot_light_curve(self, t):
        A = self.A
        I = self.I

        plt.figure(figsize=(7,4))
        plt.plot((t/self.P).value, I*A)
        plt.xlabel("Orbits")
        plt.ylabel("Fractional flux change")
        plt.title("White Dwarf Self-lensing Light Curve")
        plt.grid(True)
        plt.show()
    
    # def plot_light_curve(self, t):
    #     A = self.A
    #     I = self.I

    #     F_obs = I * A

    #     frac = np.zeros_like(F_obs.value)  # initialize
    #     mask = I > 0                       # only divide where I > 0
    #     frac[mask] = (F_obs[mask] / I[mask]) - 1.0  

    #     plt.figure(figsize=(7,4))
    #     plt.plot((t/self.P).value, frac)
    #     plt.xlabel("Orbits")
    #     plt.ylabel("Fractional flux change")
    #     plt.title("White Dwarf Microlensing Light Curve")
    #     plt.grid(True)
    #     plt.show()

    # def plot_light_curve(self, t):
    #     A = self.A

    #     F_star = self.L_star / (4 * np.pi * self.d**2)
    #     F_wd   = self.L_wd   / (4 * np.pi * self.d**2)

    #     F_star_obs = F_star * A
    #     F_wd_obs = F_wd    # no significant lensing by star

    #     # total observed and baseline flux
    #     F_obs  = F_star_obs + F_wd_obs
    #     F_base = F_star + F_wd

    #     frac = (F_obs / F_base - 1).value

    #     plt.figure(figsize=(7,4))
    #     plt.plot((t/self.P).value, frac)
    #     plt.xlabel("Orbits")
    #     plt.ylabel("Fractional flux change")
    #     plt.title("White Dwarf Self-Lensing Binary")
    #     plt.grid(True)
    #     plt.show()
