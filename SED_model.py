import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os

def pro_sed_4():
    """
    This procedure estimates a rough SED for star+disk and compares it with observations
    Star is simulated as a black body or Kurucz model
    The disk is assumed to be optically thick
    """
    
    # Star parameters
    teff = 4400.     # effective temperature in K
    rstar = 1.75     # stellar radius in solar radii
    mstar = 1.04     # stellar mass in solar masses
    par = 6.61       # parallax in mas
    Av = 1.146       # interstellar absorption in mag
    #aa = "B"        # black body spectrum
    aa = "K"         # Kurucz model spectrum
    mdot = -8.40     # logarithm of mass accretion rate (in solar Mass/yr)

    # Grain properties
    alb1 = 0.5       # ice albedo
    alb2 = 0.15      # silicate albedo
    tice = 170.      # evaporation temperature for ices (in K)
    tsil = 1500.     # evaporation temperature for silicates (in K)
    alphasil = -2.64 # exponent of power law distribution in radius for silicate grains
    alphaice = -2.98 # exponent of power law distribution in radius for ice grains
    rmaxgrain = 3.   # maximum grain size (cm)

    # Calculate evaporation radii
    rsil = ((1.0 - alb2)**0.25 / tsil**2 * rstar * teff**2 / 2.0 / 214.8)
    rice = ((1.0 - alb2)**0.25 / tice**2 * rstar * teff**2 / 2.0 / 214.8)
    rice2 = ((1.0 - alb1)**0.25 / tice**2 * rstar * teff**2 / 2.0 / 214.8)
    
    print(f"Silicate line (au)       = {rsil}")
    print(f"Ice line (au)            = {rice}")
    print(f"Pure ice line (au)       = {rice2}")

    # Disk parameters (the disk is supposed to be made of two segments)
    mdisk = 0.001    # inner disk mass in Msun
    mouter = 0.078 * 44.4 / 231.   # outer disk mass (within rout2 au) in Msun
    rin = rsil       # inner radius of silicate disk in au
    rout1 = 13.1     # outer radius of ice disk in au
    rin2 = 28.4      # inner radius of outer disk segment in au
    rout2 = 44.4     # outer radius of outer disk segment in au
    inc = 39.22      # disk inclination in degree

    print(f"Inner disk mass (Msun)   = {mdisk}")
    print(f"Inner disk lifetime (Myr)= {mdisk / 10**mdot / 1.0e6}")

    # Read observational data
    dire = "./"
    star = "sed_data.txt"
    
    # Note: You'll need to implement the readcol function or use numpy.loadtxt
    # For now, I'll create dummy data
    try:
        data = np.loadtxt(dire + star)
        wlo, f1, err1, f2, err2 = data.T
    except:
        print("Warning: Could not load observational data, using dummy data")
        wlo = np.linspace(0.1, 100, 100)
        f1 = 1e-14 * wlo**-2
        err1 = 0.1 * f1
        f2 = f1
        err2 = err1
    
    f1 = f1 + (Av / 1.33) * (f2 - f1)
    errf1 = np.log10(1.0 + err2 / f2)

    # Useful data
    nwl = 2155
    wl = np.arange(nwl, dtype=float)
    wl = 0.3 + 0.01 * wl**1.5
    wlcm = 1e-4 * wl
    h = 6.63e-27
    c = 2.9978e10
    kb = 1.38e-16
    gc = 6.67e-8
    msun = 1.989e33
    dcm = 3.09e21 / par
    cosinc = np.cos(inc * np.pi / 180.0)
    dil = 8.0 + 2.0 * np.log10(dcm)  # distance effect
    
    mouter = mouter / mdisk
    mdisk = msun * mdisk

    # Accretion luminosity
    mstarg = msun * mstar
    mdotgs = 10**mdot * msun / (86400.0 * 365.243)
    rstarcm = 6.96e10 * rstar
    lacc = 10**(np.log10(gc) + np.log10(mstarg) + np.log10(mdotgs) - np.log10(rstarcm))

    # Star photospheric spectrum
    phs = np.zeros(nwl)
    c1 = 2.0 * h * c**2
    c2 = h * c / kb

    if aa == "B":
        # Black body spectrum
        phs = c1 / wlcm**5 / (np.exp(c2 / (wlcm * teff)) - 1.0)
        phs = np.log10(np.pi * rstarcm**2 * phs)
        phs = phs - dil
    elif aa == "K":
        # Kurucz model spectrum
        dire2 = dire + "Kurucz/"
        tmod = 3500 + 250 * np.arange(27)
        
        # Find appropriate temperature models
        i = 0
        while teff > tmod[i] and i < len(tmod) - 1:
            i += 1
        
        # Note: You'll need to implement the FITS reading for Kurucz models
        # For now, using black body as fallback
        print("Warning: Kurucz model reading not implemented, using black body")
        phs = c1 / wlcm**5 / (np.exp(c2 / (wlcm * teff)) - 1.0)
        phs = np.log10(np.pi * rstarcm**2 * phs)
        phs = phs - dil

    # Accretion spectrum
    pha = np.zeros(nwl)
    tacc = 10000.0  # assumed temperature for the accretion region
    pha = c1 / wlcm**5 / (np.exp(c2 / (wlcm * tacc)) - 1.0)
    lum1 = 5.6704e-5 * tacc**4
    surf = lacc / lum1

    pha = np.log10(pha * surf)
    pha = pha - dil

    # Disk thermal spectrum
    phd = np.zeros(nwl)

    # The disk is subdivided into nseg segments
    nseg = 470
    r1 = np.arange(nseg, dtype=float)
    rstep = 0.0005
    expo = 2.0
    r = 0.02 + rstep * r1**expo
    dr = expo * rstep * r1**(expo - 1.0)

    # Calculate gas surface density
    mtot = 0.0
    for i in range(nseg):
        if r[i] > rsil and r[i] < rout1:
            mtot += dr[i]
    
    norma = mdisk / mtot / 1.495e13
    sigmagas = norma / (2.0 * np.pi * r * 1.495e13)

    # Grain size distribution
    dg = 0.1
    fact = 10**dg - 1.0
    ndim = int(10.0 * (np.log10(rmaxgrain) + 5.0)) + 1
    rgrain = dg * np.arange(ndim) - 5.0  # cm
    rgrain = 10**rgrain
    dgrain = rgrain * fact
    
    # Grain masses
    silgrainmass = 3.5 * (4.0 * np.pi / 3.0) * rgrain**3
    icegrainmass = 0.92 * (4.0 * np.pi / 3.0) * rgrain**3
    
    mtotsil = silgrainmass * rgrain**alphasil * dgrain
    mtotice = icegrainmass * rgrain**alphaice * dgrain
    ntotsil = rgrain**alphasil * dgrain
    ntotice = rgrain**alphaice * dgrain
    
    ngrainsil = np.sum(ntotsil)
    ngrainice = np.sum(ntotice)
    ntotsil = ntotsil / ngrainsil
    ntotice = ntotice / ngrainice

    msil = np.sum(mtotsil)
    mice = np.sum(mtotice)
    mmedsil = msil / ngrainsil
    mmedice = msil / ngrainice

    # Calculate opacity
    q = np.zeros((nwl, ndim))
    for j in range(ndim):
        for i in range(nwl):
            xx = 0.0001 * wl[i] / rgrain[j]
            if xx < 0.375:
                q[i, j] = 0.3 * xx
            elif xx < 2.188:
                q[i, j] = 0.8 * xx**2
            elif xx < 1000.0:
                q[i, j] = 2.0 + 4.0 / xx
            else:
                q[i, j] = 2.0
            q[i, j] = q[i, j] * np.pi * rgrain[j]**2

    # Calculate disk segments
    tseg = np.zeros(nseg)
    for i in range(nseg):
        area = 2.0 * np.pi * r[i] * dr[i] * 214.8**2 * cosinc
        
        # Surface densities
        if r[i] > rsil:
            sigmasil = 0.0043 * sigmagas[i]
        else:
            sigmasil = 0.0
            
        if r[i] > rice:
            sigmaice = 0.0094 * sigmagas[i]
        else:
            sigmaice = 0.0
            
        a = np.zeros(nwl)
        optthick = np.zeros(nwl)
        
        if r[i] > rin and r[i] < rout1:
            for j in range(nwl):
                for k in range(ndim):
                    optthick[j] += q[j, k] * ((sigmasil / mmedsil) * ntotsil[k] + 
                                            (sigmaice / mmedice) * ntotice[k])
                a[j] = area * optthick[j] / (1.0 + optthick[j])
                
        if r[i] > rin2 and r[i] < rout2:
            for j in range(nwl):
                for k in range(ndim):
                    optthick[j] += q[j, k] * ((sigmasil / mmedsil) * ntotsil[k] + 
                                            (sigmaice / mmedice) * ntotice[k])
                optthick[j] = mouter * optthick[j]
                a[j] = area * optthick[j] / (1.0 + optthick[j])
        
        # Temperature in segment
        if r[i] < rice:
            tseg[i] = teff * np.sqrt(rstar / (2.0 * r[i] * 214.8)) * (1.0 - alb2)**0.25
        else:
            tseg[i] = teff * np.sqrt(rstar / (2.0 * r[i] * 214.8)) * (1.0 - alb1)**0.25
            
        for j in range(nwl):
            phd[j] += a[j] * c1 / wlcm[j]**5 / (np.exp(c2 / (wlcm[j] * tseg[i])) - 1.0)

    # Find N2 ice line
    i = 0
    while i < len(tseg) and tseg[i] > 30.0:
        i += 1
    if i < len(tseg):
        print(f"N2 ice line (au) {r[i]}")

    phd = np.log10(phd) + 2.0 * np.log10(6.96e10) - dil

    # Total SED
    pht = np.log10(10**phd + 10**phs + 10**pha)

    # Plotting
    plt.figure(figsize=(10, 8))

    plt.rcParams.update({'font.size': 14})
    
    # Observational data with errors
    plt.errorbar(np.log10(wlo), np.log10(f1), yerr=errf1, fmt='o', 
                capsize=3, label='Observations')
    plt.errorbar(np.log10(870), np.log10(5.608e-20), yerr=8e-21, fmt='ro', 
                capsize=3, label='ALMA flux')
    
    # Model components
    plt.plot(np.log10(wl), pht, 'k-', linewidth=2, label='Total')
    plt.plot(np.log10(wl), phs, 'r--', label='Stellar')
    plt.plot(np.log10(wl), phd, 'b--', label='Disk')
    plt.plot(np.log10(wl), pha, 'g--', label='Accretion')
    
    plt.xlabel('log Wavelength (micron)')
    plt.ylabel('log Flux (erg cm$^{-2}$ s$^{-1}$ A$^{-1}$)')
    plt.xlim(-1, 3.5)
    plt.ylim(-20, -12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('SED Model')
    
    # Save plot
    plt.savefig(dire + 'sed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'wavelength': wl,
        'total_flux': pht,
        'stellar_flux': phs,
        'disk_flux': phd,
        'accretion_flux': pha,
        'observations': {
            'wavelength': wlo,
            'flux': f1,
            'error': errf1
        }
    }

# Run the function
if __name__ == "__main__":
    results = pro_sed_4()