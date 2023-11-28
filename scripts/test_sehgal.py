import pixell
from pixell import enplot
from pixell import reproject,enmap,utils
import numpy as np
import healpy as hp
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io import fits

tsz = "/data5/sims/sehgal/data/tSZ_skymap_healpix_nopell_Nside4096_y_tSZrescale0p75.fits"
#tot_93_map = "/data6/kaper/ml_sz_clusters/sehgal/tot_93.fits"
res = np.deg2rad(0.5 / 60.)
width = 16*utils.arcmin
shape,wcs = enmap.fullsky_geometry(res=res)
h_map = hp.read_map(tsz).astype(np.float64)
p_map = reproject.healpix2map(h_map,shape=shape,wcs=wcs)
p_map.write("tsz_seh.fits")
"""
#enplot.write("tot_seh",enplot.plot(p_map))
"""
halo_coords = "/data6/kaper/ml_sz_clusters/sehgal/halos.fits"
hdul = fits.open(halo_coords)
data = hdul[1].data
idx,ra,dec,m200,z = data["name"],data["RADeg"],data["decDeg"],data["M200m"],data["z"]
print(ra[10000])
print(dec[10000])

for i in range(20000,20010):
    ra_i,dec_i = ra[i]*u.deg,dec[i]*u.deg
    cutout = reproject.thumbnails(p_map,(dec_i.to(u.radian).value,ra_i.to(u.radian).value),r=width,res=res)
    print(cutout.shape)
    cutout = cutout[:64,:64]
    fig = plt.figure()
    plt.imshow(cutout)
    plt.savefig("plot_seh/sehgal_"+str(i)+".png")
    plt.close(fig)

