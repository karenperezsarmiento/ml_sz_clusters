import numpy as np
import pandas as pd
import astropy.table as table

f = open("/data5/sims/sehgal/data/halo_nbody.ascii",'r')
data = np.genfromtxt(f)
idx,ra,dec,m200,z = np.arange(len(data)),data[:,1],data[:,2],data[:,9],data[:,0]

names =[]
for i in range(len(idx)):
    n = 'MOCK-CL_'+str(i)
    names.append(n)

outFileName = "/data6/kaper/ml_sz_clusters/sehgal/halos.fits"
tab = table.Table()
tab.add_column(table.Column(names, 'name'))
tab.add_column(table.Column(ra, 'RADeg'))
tab.add_column(table.Column(dec, 'decDeg'))
tab.add_column(table.Column(m200, 'M200m'))
tab.add_column(table.Column(z, "z"))
tab=tab[np.where(tab['M200m'] >= 0.5e14)]
tab=tab[np.where(tab['M200m'] <= 1e15)]
tab.write(outFileName,overwrite=True)
print("Sehgal created")


