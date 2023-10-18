import pixell
from pixell import reproject,enmap,utils
import numpy as np
import healpy as hp
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import json
import jsonlines
import argparse as ap

print("Initialized")
parser = ap.ArgumentParser(description="Create cutouts around websky halos to use as input in CNN")
parser.add_argument("-of","--output_file",type=str,help="Name of output file")
parser.add_argument("-s","--size",type=str,help="Size of output dataset (mini,small or full")
parser.add_argument("-nc","--num_channels",type=int,help="Number of channels in output cutout (1 or 3 or 5)")
args = parser.parse_args()
output_fn = args.output_file
output_sz = args.size
nchannels = args.num_channels
print("output filename "+output_fn)
print("size of dataset "+output_sz)
print("num channels "+str(nchannels))

res = np.deg2rad(0.5 / 60.)

tot_145_map = "../websky/tot_145.fits"
tot_93_map = "../websky/tot_93.fits"
tot_217_map = "../websky/tot_217.fits"

tsz_map = "../websky/tsz_8192.fits"
halo_coords = "../websky/halos.fits"
width = 16*utils.arcmin

z_max = 4.6
mass_min = 0.5e14
mass_max = 1e15

def healpy_2_pixell(filename,res):
    shape,wcs = enmap.fullsky_geometry(res=res)
    h_map = hp.read_map(filename).astype(np.float64)
    p_map = reproject.healpix2map(h_map,shape=shape,wcs=wcs)
    min_val = np.min(p_map)
    max_val = np.max(p_map)
    return p_map, min_val, max_val

def get_cutouts(ra,dec,p_map,width=10.0*utils.arcmin,res=res):
    ra,dec = ra*u.deg,dec*u.deg
    cutouts = reproject.thumbnails(p_map,(dec.to(u.radian).value,ra.to(u.radian).value),r=width,res=res)
    return cutouts

def load_coords(coord_file):
    hdul = fits.open(coord_file)
    data = hdul[1].data 
    return data["name"],data["RADeg"],data["decDeg"],data["M200m"],data["z"]

idx,ra,dec,m200,z = load_coords(halo_coords)

maps_dict = {}
files = ["tot_93","tot_145","tot_217","tsz_8192"]
for i in files:
    f = "../websky/"+ i + ".fits"
    p_map, min_val, max_val = healpy_2_pixell(f,res)
    maps_dict[i] = {
        "p_map" : p_map,
        "min" : min_val,
        "max" : max_val,    
    }
    
ofn = "../data/"+output_fn
if output_sz=="mini":
    osize = 1000
elif output_sz=="small":
    osize = 20000
else:
    osize = len(ra)
with jsonlines.open(ofn,mode="w") as write_file:
    for i in range(osize):
        if (nchannels == 3) or (nchannels == 5):
            z_norm = z[i]/z_max
            m200_norm = (m200[i]-mass_min)/(mass_max-mass_min)
            z_cutout = np.ones((64,64))*z_norm
            m_cutout = np.ones((64,64))*m200_norm 
            cutout_out = np.dstack((m_cutout,z_cutout))
        cutout_dict = {}
        cutout_dict["name"]=str(idx[i])
        if (nchannels == 1) or (nchannels == 3):
            list_freqs = ["tot_93"]
        elif nchannels == 5:
            list_freqs = ["tot_93","tot_145","tot_217"]
        for k in list_freqs:
            cutout = get_cutouts(ra[i],dec[i],maps_dict[k]["p_map"],width=width,res=res)
            cutout = cutout[:64,:64]
            cutout_norm = (cutout - maps_dict[k]["min"])/(maps_dict[k]["max"]-maps_dict[k]["min"])
            if (nchannels == 5) or (nchannels == 3):
                cutout_out = np.dstack((cutout_out,cutout_norm))
            elif nchannels == 1:
                cutout_out = cutout_norm
        
        cutout_dict["tot"] = cutout_out.tolist()
        cutout_label = get_cutouts(ra[i],dec[i],maps_dict["tsz_8192"]["p_map"],width=width,res=res)
        cutout_label = cutout_label[:64,:64]
        cutout_label_norm = (cutout_label - maps_dict["tsz_8192"]["min"])/(maps_dict["tsz_8192"]["max"]-maps_dict["tsz_8192"]["min"])
        cutout_dict["tsz_8192"] = cutout_label_norm.tolist()
        write_file.write(cutout_dict)

write_file.close()
print("Created input dataset (normalized)")
