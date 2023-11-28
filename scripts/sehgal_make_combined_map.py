import healpy as hp
import numpy as np

cmb = hp.read_map("../sehgal/Sehgalsimparams_healpix_4096_KappaeffLSStoCMBfullsky_phi_SimLens_Tsynfastnopell_fast_lmax8000_nside4096_interp2.5_method1_1_lensed_map.fits")

tsz = hp.read_map("../sehgal/tSZ_skymap_healpix_nopell_Nside4096_y_tSZrescale0p75.fits")

hplanck = 6.626e-34
kb = 1.38e-23
nu_ghz = np.array([93,145,217])
nu = nu_ghz*1e9
T = 2.726
x = hplanck*nu/(kb*T)
f = T*1e6*(x*(np.exp(x)+1)/(np.exp(x)-1) - 4)

for i in range(len(nu_ghz)):
    tot = cmb + f[i]* tsz
    fn = "../sehgal/tot_"+str(nu_ghz[i])+".fits"
    hp.write_map(fn,tot,dtype='float64',overwrite=True) 
print("done")
