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
from orphics import io

print("Initialized")
parser = ap.ArgumentParser(description="Create cutouts around websky halos to use as input in CNN")
parser.add_argument("-s","--size",type=str,help="Size of output dataset (mini,small or full")
parser.add_argument("-nc","--num_channels",type=int,help="Number of channels in output cutout (1 or 2 or 3 or 5)")
parser.add_argument("-ns","--noise",type=str,help="Include noise")
parser.add_argument("-bm","--beam",type=str,help="Include beam")
parser.add_argument("-nl","--noise_level",type=float,help="Noise level (ignored if ns is false)")
parser.add_argument("-ap","--apodization",type=str,help="Apodization type (None or (Pixell pix))")
args = parser.parse_args()
output_sz = args.size
nchannels = args.num_channels
noise = args.noise
beam = args.beam
noise_level = args.noise_level
apod_str = args.apodization

# Parse noise and beam arguments
if noise == "True":
    noise = True
else:
    noise = False
if beam == "True":
    beam = True
else:
    beam = False


output_fn = output_sz+"_"+str(nchannels)+"ch"
if beam:
    output_fn = output_fn+"_b"
if noise:
    output_fn = output_fn+"_n"
    output_fn = output_fn+str(int(noise_level))

# Parse apodization arguments
l = apod_str.split(" ")
if l[0] == "None":
    apod_info = {"mode":"None"}
    output_fn = output_fn+"_noap"
elif l[0] == "Pixell":
    try:
        pix = int(l[1])
        apod_info = {"mode":"Pixell", "pix":pix} # Width in  pixels for apodization 
    except:
        raise ValueError("When passing in apodization type Pixell, it must be followed with an int. For example: Pixell 5")
    output_fn = output_fn+"_pxlap_"+str(pix)
else:
    raise ValueError("Invalid apodization type. Must start with None or Pixell")

identity = output_fn
output_fn = output_fn+".jsonl"

# Print parsed arguments
print("output filename "+output_fn)
print("size of dataset "+output_sz)
print("num channels "+str(nchannels))
print("include noise "+str(noise))
print("include beam "+str(beam))
print("If noise, then noise level set to "+str(noise_level))
print("apod_info "+ str(apod_info))

res = np.deg2rad(0.5 / 60.)

tot_145_map = "/data6/kaper/ml_sz_clusters/websky/tot_145.fits"
tot_93_map = "/data6/kaper/ml_sz_clusters/websky/tot_93.fits"
tot_217_map = "/data6/kaper/ml_sz_clusters/websky/tot_217.fits"

tsz_map = "/data6/kaper/ml_sz_clusters/websky/tsz_8192.fits"
halo_coords = "/data6/kaper/ml_sz_clusters/websky/halos.fits"
width = 16*utils.arcmin

z_max = 4.6
mass_min = 0.5e14
mass_max = 1e15

beam_dict = {"tot_93":2.2,"tot_145":1.4,"tot_217":1.0,"tsz_8192":1.6}

def gauss_beam(ell,fwhm):
    tht_fwhm= np.deg2rad(fwhm/60.)
    return np.exp(-(tht_fwhm**2.)*(ell**2.) / (16.*np.log(2.)))

def conv_beam_noise(p_map,freq,add_noise,noise_level):
    shape,wcs = enmap.fullsky_geometry(res=res)
    imap = enmap.zeros(shape, wcs=wcs)
    alm = pixell.curvedsky.map2alm(p_map,lmax=10000)
    alm_conv = pixell.curvedsky.almxfl(alm,lambda ell:gauss_beam(ell,beam_dict[freq]))
    p_map_beam = pixell.curvedsky.alm2map(alm_conv,imap)
    if add_noise:
        dra, ddec = wcs.wcs.cdelt*utils.degree
        dec = enmap.posmap([shape[-2],1],wcs)[0,:,0]
        area = np.abs(dra*(np.sin(np.minimum(np.pi/2.,dec+ddec/2))-np.sin(np.maximum(-np.pi/2.,dec-ddec/2))))
        Nx = shape[-1]
        a_rad_map = enmap.ndmap(area[...,None].repeat(Nx,axis=-1),wcs)
        arm_rad_map = a_rad_map*((180.*60./np.pi)**2.)
        div = arm_rad_map/noise_level**2.
        seed = 8192
        np.random.seed(seed)
        w_n_map = np.random.standard_normal(shape) / np.sqrt(div)
        p_map_out = p_map_beam + w_n_map
    else:
        p_map_out = p_map_beam
    return p_map_out

def healpy_2_pixell(filename,res):
    shape,wcs = enmap.fullsky_geometry(res=res)
    h_map = hp.read_map(filename).astype(np.float64)
    p_map = reproject.healpix2map(h_map,shape=shape,wcs=wcs)
    return p_map

def get_cutouts(ra,dec,p_map,width=10.0*utils.arcmin,res=res, apod_info={"mode":"None"}):
    ra,dec = ra*u.deg,dec*u.deg
    cutouts = reproject.thumbnails(p_map,(dec.to(u.radian).value,ra.to(u.radian).value),r=width,res=res)
    cutout = np.array(cutouts[:64,:64])
    if apod_info["mode"]=="Pixell":   #Pixell
        cutout = pixell.enmap.apod(cutout, apod_info["pix"])
    return cutout

def load_coords(coord_file):
    hdul = fits.open(coord_file)
    data = hdul[1].data 
    return data["name"],data["RADeg"],data["decDeg"],data["M200m"],data["z"]

idx,ra,dec,m200,z = load_coords(halo_coords)

maps_dict = {}
files = ["tot_93","tot_145","tot_217","tsz_8192"]
for i in files:
    f = "/data6/kaper/ml_sz_clusters/websky/"+ i + ".fits"
    p_map = healpy_2_pixell(f,res)
    if beam & (i!="tsz_8192"):
        p_map_out = conv_beam_noise(p_map,freq = i,add_noise = noise, noise_level=noise_level)
    elif beam & (i=="tsz_8192"):
        p_map_out = conv_beam_noise(p_map,freq = i,add_noise = False, noise_level=noise_level)   
    else:
        p_map_out = p_map
    #io.plot_img(p_map_out,"/home3/avharris/ml_sz_clusters/data/debugging/"+identity+"_"+i+".png")
    min_val = np.min(p_map_out)
    max_val = np.max(p_map_out)
    maps_dict[i] = {
        "p_map" : p_map_out,
        "min" : min_val,
        "max" : max_val,    
    }
    """
    plt.hist(p_map_out[0].flatten())
    plt.title("Histogram of p_map_out for "+i)
    plt.savefig("/home3/avharris/ml_sz_clusters/data/debugging/"+identity+"p_map_out_hist_"+i+".png")
    plt.close("all")
    """

ofn = "/data6/avharris/ml_sz_clusters/datasets/"+output_fn
if output_sz=="tiny":
    osize = 10
elif output_sz=="mini":
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
	elif nchannels == 2:
	    list_freqs = ["tot_93","tot_145"]
        elif nchannels == 5:
            list_freqs = ["tot_93","tot_145","tot_217"]
        for k in list_freqs:
            cutout = get_cutouts(ra[i],dec[i],maps_dict[k]["p_map"],width=width,res=res, apod_info=apod_info)
            cutout_norm = (cutout - maps_dict[k]["min"])/(maps_dict[k]["max"]-maps_dict[k]["min"])
            if (nchannels == 5) or (nchannels == 3) or (nchannels == 2):
                cutout_out = np.dstack((cutout_out,cutout_norm))
            elif nchannels == 1:
                cutout_out = cutout_norm
        
        cutout_dict["tot"] = cutout_out.tolist()
        cutout_label = get_cutouts(ra[i],dec[i],maps_dict["tsz_8192"]["p_map"],width=width,res=res, apod_info=apod_info)
        cutout_label = cutout_label[:64,:64]
        cutout_label_norm = (cutout_label - maps_dict["tsz_8192"]["min"])/(maps_dict["tsz_8192"]["max"]-maps_dict["tsz_8192"]["min"])
        cutout_dict["tsz_8192"] = cutout_label_norm.tolist()
        write_file.write(cutout_dict)

write_file.close()
print("Created input dataset (normalized)")
