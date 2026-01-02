import pygyre as pg
import numpy as np
import scipy as cp
from scipy import integrate
import matplotlib.pyplot as plt

class BHBinary:
    def __init__(self, star_name, orbit="nwtn", MBH=1e6, Rp=17, a=0.0, E=1.0, Q=0.0, N=1000):
        pass

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
        q = self.Carter
        E = self.OrbitEnergy
        rp = self.Rp
        a = self.a

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
    
    def geodesic_kerr_s2(self, τ, dq, dE, dLz, y):
        """
        Testing new geodesic function based off of Kesden 2012.
        """
        t, r, phi, theta, psi = y

        q = dq
        E = dE
        a = self.a
        Lz = dLz

        sigma = r**2 + a**2 * np.cos(theta)**2
        delta = r**2 + a**2 - 2 * r
        alpha = (r**2 + a**2)**2 - delta * a**2 * np.sin(theta)**2

        dt_dτ = ((alpha * E - 2 * a * r * Lz) / delta) / sigma
        dr_dτ = np.sqrt((E * (r**2 + a**2) - a * Lz)**2 - delta * (r**2 + (Lz - a * E)**2 + q)) / sigma
        dφ_dτ = (Lz * np.csc(theta)**2 + (2 * a * r * E - a**2 * Lz) / delta) / sigma
        dθ_dτ = np.sqrt(q - Lz**2 * np.cot(theta)**2 - a**2 * (1 - E**2) * np.cos(theta)**2) / sigma

        dpsi_dτ = np.abs(a - Lz) * (((r**2 + a**2) - a * Lz) / ((a - Lz)**2 + r**2) + a * (Lz - a) / (a - Lz)**2) / r**2

        return [dt_dτ, dr_dτ, dφ_dτ, dθ_dτ, dpsi_dτ]