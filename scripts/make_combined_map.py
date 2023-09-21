import healpy as hp
import numpy as np

lensed_alm = "../websky/lensed_alm.fits"
alm = hp.read_alm(lensed_alm)
cmb_lensed_map = hp.alm2map(alm,8192)

tsz = hp.read_map("../websky/tsz_8192.fits")
f_145 = -2.7685e6
tsz_145 = f_145*tsz
f_93 = -4.2840e6
tsz_93 = f_93*tsz
f_217 = -2.1188e4
tsz_217 = f_217*tsz

tot_217 = tsz_217 + cmb_lensed_map
tot_145 = tsz_145 + cmb_lensed_map
tot_93 = tsz_93 + cmb_lensed_map

hp.write_map("../websky/tot_217.fits",tot_217)
hp.write_map("../websky/tot_145.fits",tot_145)
hp.write_map("../websky/tot_93.fits",tot_93)

