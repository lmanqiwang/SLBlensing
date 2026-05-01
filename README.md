# SLBlensing

`SLBlensing` is a publicly available Python class that simulates light curve profiles of SLBs to leading order, implementing geometric lensing and occultation, limb and gravity darkening, Doppler boosting, ellipsoidal variation, and irradiation. 

## Parameter Setup

The `SLBlensing` object class takes in a series of orbital, stellar, and compact object-lens specific parameters to calculate lensing light curves for SLBs, and will also calculate reasonable light curves of non-lensing binaries of stars with compact objects.

### Input Parameters

| Parameter | Description |
|-----------|-------------|
| `mass_lens` | Mass of the compact object lens (M☉) |
| `r_lens` | Radius of the compact object lens (R☉) |
| `l_lens` | Luminosity of the compact object lens (L☉) |
| `T_eff_lens` | Effective temperature of the compact object lens (K) |
| `mass_star` | Mass of the companion star (M☉) |
| `r_star` | Radius of the companion star (R☉) |
| `l_star` | Luminosity of the companion star (L☉) |
| `T_eff_star` | Effective temperature of the companion star (K) |
| `ecc` | Eccentricity of the binary orbit |
| `a` | Semi-major axis of the binary orbit (any distance unit) |
| `d` | Distance to binary (any distance unit) |
| `omega` | Argument of periastron (degrees) |
| `inc` | Inclination to the observer's line of sight (degrees) |
| `period` | Period of the binary (days) |
| `cycles` | Total number of complete orbits to plot |
| `limb_darkening_l` | Quadratic limb darkening coefficients for the compact object |
| `limb_darkening_star` | Quadratic limb darkening coefficients for the companion star |
| `gravity_darkening` | Gravity darkening coefficient of the star due to the compact object |
| `offset` | Boolean — shifts `t=0` by a quarter period for convenient viewing of lensing flares |
| `freq` | Observation frequency (Hz); also enables Doppler boosting, ellipsoidal radiation, and irradiation effects |
| `N` | Number of timesteps for the light curves |

---

## Instance Variables

The following instance variables are computed during initialization (unless otherwise noted) and can be used for direct comparison with other SLB systems.

| Variable | Description |
|----------|-------------|
| `t` | Time array (days) |
| `f` | True anomaly array |
| `r` | Orbital separation |
| `r_sky` | Sky-projected separation between compact object and star |
| `r_E` | Einstein radius at each timestep |
| `overlap` | Area overlap between compact object and star |
| `P` | Period (years) |
| `geo_flux` | Modeled flux profile with all effects considered |
| `base_flux` | Unperturbed stellar flux (for normalization) |
| `amp` | Lensing magnification factor at each timestep |
| `ellip_var` | Flux variation due to ellipsoidal variation |
| `alpha_tra` | Fractional compact object disk area covered by star during transit |
| `alpha_occ` | Fractional compact object disk area covered by star during occultation |
| `v_star_los` | Line-of-sight velocity of the star |
| `v_lens_los` | Line-of-sight velocity of the compact object |
| `F_bol_star` | Flux of the star at the frequency given by `freq` |
| `F_bol_lens` | Flux of the compact object at the frequency given by `freq` |
| `tra_ingress` | Ingress duration for the compact object transiting the star — run `eclipse_duration()` first |
| `tra_engress` | Egress duration for the compact object transiting the star — run `eclipse_duration()` first |
| `tra_full` | Duration of compact object transit between ingress and egress — run `eclipse_duration()` first |
| `tra_tot` | Total duration of the compact object transit — run `eclipse_duration()` first |
| `tra_t0` | Characteristic timescale of the transit — run `eclipse_duration()` first |
| `occ_ingress` | Ingress duration for the compact object occultation — run `eclipse_duration()` first |
| `occ_engress` | Egress duration for the compact object occultation — run `eclipse_duration()` first |
| `occ_full` | Duration of compact object occultation between ingress and egress — run `eclipse_duration()` first |
| `occ_tot` | Total duration of the compact object occultation — run `eclipse_duration()` first |
| `occ_t0` | Characteristic timescale of the occultation |

---

## Methods

| Method | Description |
|--------|-------------|
| `orbital_period()` | Calculates the orbital period if not user-defined; stores as `self.P` in years |
| `velocity_los()` | Computes line-of-sight velocities for the compact object and companion star at every timestep |
| `ecc_anomaly(ti)` | Solves for the eccentric anomaly (*E*) at each timestep via Newton-Raphson |
| `cartesian_coords()` | Converts orbital elements (*r*, *f*, *ω*, *i*) into Cartesian sky coordinates |
| `projected_separation()` | Calculates the sky-plane separation *r*_sky and impact parameters for lens transit and occultation |
| `einstein_radius()` | Computes the Einstein radius of the binary system |
| `eclipse_duration()` | Calculates total, full, ingress, and egress durations for lens transits and occultations |
| `overlapping_area()` | Calculates the geometric intersection of the lens and source star on the sky |
| `alpha()` | Helper — computes `alpha_tra` and `alpha_occ` (fraction of lens disk covered by star during transit/occultation) |
| `geometric_limb_darkening(u1, u2, u3, u4)` | Calculates limb-darkening effects on transit and occultation curves |
| `magnification()` | Computes the lensing magnification factor |
| `geometric_flux()` | Computes the total light curve with geometric transit/occultation dips, lensing magnification, Doppler boosting, ellipsoidal variation, and irradiation |
| `light_curve_plotter(name, n)` | Plots lensing bumps and occultation dips. `name` sets the plot title; `n` sets the fractional period padding on the time axis |
| `transit_plotter(name, n)` | Plots only the lensing bump of the binary |
