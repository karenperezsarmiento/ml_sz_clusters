import healpy as hp
from cosmology import *
import numpy as np
import pandas as pd
import astropy.table as table

rho = 2.775e11*omegam*h**2
f=open('/data5/sims/websky/data/halos.pksc')
N=np.fromfile(f,count=3,dtype=np.int32)[0]

catalog = np.fromfile(f,count=N*10,dtype=np.float32)
catalog = np.reshape(catalog,(N,10))

x  = catalog[:,0];  y = catalog[:,1];  z = catalog[:,2] # Mpc (comoving)
vx = catalog[:,3]; vy = catalog[:,4]; vz = catalog[:,5] # km/sec
R  = catalog[:,6] # Mpc

M200m    = 4*np.pi/3.*rho*R**3
chi      = np.sqrt(x**2+y**2+z**2) #Mpc
vrad     = (x*vx + y*vy + z*vz) / chi #km/s
redshift = zofchi(chi)

theta,phi = hp.vec2ang(np.column_stack((x,y,z))) #radians
nside_1 = 2048
nside_2 = 4092
nside_3 = 8192

#map_1 =np.zeros((hp.nside2npix(nside_1)))
#map_2 =np.zeros((hp.nside2npix(nside_2)))

pix_1 = hp.vec2pix(nside_1, x, y, z)
pix_2 = hp.vec2pix(nside_2, x, y, z)
pix_3 = hp.vec2pix(nside_3, x, y, z)

decDeg=-1*(np.degrees(theta)-90)
raDeg = np.degrees(phi)

names=[]
for i in range(len(raDeg)):
    n = 'MOCK-CL_'+ str(i)
    names.append(n)

outFileName="/data6/kaper/ml_sz_clusters/websky/halos.fits"
tab = table.Table()
tab.add_column(table.Column(names, 'name'))
tab.add_column(table.Column(raDeg, 'RADeg'))
tab.add_column(table.Column(decDeg, 'decDeg'))
tab.add_column(table.Column(M200m, 'M200m'))
tab.add_column(table.Column(redshift, "z"))
tab=tab[np.where(tab['M200m'] >= 0.5e14)]
tab=tab[np.where(tab['M200m'] <= 1e15)]
##add redshift filter
tab.write(outFileName,overwrite=True)
print("Websky created")
