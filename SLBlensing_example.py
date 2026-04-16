from SLBlensing import SLBlensing
from astropy import units as u
import numpy as np

# white dwarf
mass_wd = 0.634         
r_wd = 0.01166       
l_wd = 0.00120       
t_wd = 10000        
lds_wd  = [0.35, 0.18]  
 
# sun-like star
mass_s = 1.0          
r_s = 1.0         
l_s = 1.0         
T_eff_s  = 5778        
lds_s = [0.44, 0.23] 
tg_s = 0.32        
 
# orbital params
d         = 500.0 * u.pc 
inc_deg   = 90 * u.deg  
omega_deg = 0 * u.deg    
N         = 1000000
 
binary = SLBlensing(
    mass_wd, r_wd, l_wd, t_wd,
    mass_s,  r_s,  l_s,  T_eff_s,
    ecc=0.05,
    a=2.0 * u.AU,
    d=d,
    omega=omega_deg,
    inc=inc_deg,
    limb_darkening_l=lds_wd,
    limb_darkening_star=lds_s,
    gravity_darkening=tg_s,
    N=N,
)
 
# print statements of binary variables
print("=" * 50)
print("Calculated Binary Properties")
print("=" * 50)
print(f"Period:              {binary.P:.4}")
print(f"Einstein Radius:     {binary.r_E}")
print(f"LOS Velocity Lens:   {binary.v_lens_los}")
print(f"LOS Velocity Star:   {binary.v_star_los}")
print()
 
binary.eclipse_duration()
 
print("=" * 50)
print(f"Total Transit Time:              {binary.tra_tot.to(u.hr):.4}")
print(f"Transit Time w/o Ingress/Egress: {binary.tra_full.to(u.hr):.4}")
print(f"Transit Ingress Time:            {binary.tra_ingress.to(u.hr):.4}")
print(f"Transit Egress Time:             {binary.tra_egress.to(u.hr):.4}")
print()
 
print("=" * 50)
print(f"Total Occultation Time:              {binary.oc_tot.to(u.hr):.4}")
print(f"Occultation Time w/o Ingress/Egress: {binary.oc_full.to(u.hr):.4}")
print(f"Occultation Ingress Time:            {binary.oc_ingress.to(u.hr):.4}")
print(f"Occultation Egress Time:             {binary.oc_egress.to(u.hr):.4}")
print()
 
TITLE = "White Dwarf-Sun-Like Star System"
TS = 0.007
 
# Full light curve (transit + occultation panels)
binary.light_curve_plotter(TITLE, TS)
 
# Transit-only zoom plot
binary.transit_plotter(TITLE, TS)
 
 
# binary with frequency
binary_w_freq = SLBlensing(
    mass_wd, r_wd, l_wd, t_wd,
    mass_s,  r_s,  l_s,  T_eff_s,
    ecc=0.05,
    a=2.0 * u.AU,
    d=d,
    omega=omega_deg,
    inc=inc_deg,
    limb_darkening_l=lds_wd,
    limb_darkening_star=lds_s,
    gravity_darkening=tg_s,
    freq=5e14 * u.Hz,
    N=N,
)
 
binary_w_freq.transit_plotter(TITLE, TS)