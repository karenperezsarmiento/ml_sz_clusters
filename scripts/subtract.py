import numpy as np
import tensorflow as tf
import keras
import datasets
from datasets import load_dataset
import matplotlib.pyplot as plt
import re
from pixell import reproject,enmap,utils
import healpy as hp
import argparse as ap

print("Initialized")
parser = ap.ArgumentParser(description="Create residuals and subtractions")
parser.add_argument("-mf","--model_file",type=str,help="Model filename")
parser.add_argument("-td","--test_dataset",type=str,help="Test dataset")
parser.add_argument("-nc","--num_channels",type=int,help="Number of channels")

args = parser.parse_args()
model_fn = args.model_file
input_fn = args.test_dataset
nchannels = args.num_channels
print("Parsed arguments")
m = re.sub(".keras","",model_fn)
f = re.sub(".jsonl","",input_fn)
img_dir = "../test_imgs_"+m+"_"+f+"/"

res = np.deg2rad(0.5 / 60.)

def map_min_max(filename,res):
    shape,wcs = enmap.fullsky_geometry(res=res)
    h_map = hp.read_map(filename).astype(np.float64)
    p_map = reproject.healpix2map(h_map,shape=shape,wcs=wcs)
    min_val = np.min(p_map)
    max_val = np.max(p_map)
    return min_val, max_val

maps_dict = {}
if nchannels == 5:
    files = ["tot_93","tot_145","tot_217","tsz_8192"]
    flabels = {2:"93",3:"145",4:"217"}
    f_arr = [2,3,4,5]
    f_arr2 = [2,3,4]
    n = {2:-4.2840e6,3:-2.7685e6,4:-2.1188e4}
elif nchannels == 2:
    files = ["tot_93","tot_145","tsz_8192"]
    flabels = {0:"93", 1:"145"}
    f_arr = [0, 1, 5]
    f_arr2 = [0, 1]
    n = {0:-4.2840e6, 1:-2.7685e6}
else:
    files = ["tot_93","tsz_8192"]
    flabels = {2:"93"}
    f_arr = [2,5]
    f_arr2 = [2]
    n = {2:-4.2840e6}

print("Loading maps")
for i in range(len(files)):
    f = "/data6/kaper/ml_sz_clusters/websky/"+ files[i] + ".fits"
    min_val, max_val = map_min_max(f,res)
    maps_dict[f_arr[i]] = {
        "min" : min_val,
        "max" : max_val,
    }

print("Loading and splitting dataset")
if "linear" in model_fn:
    act_func = "linear"
elif "selu" in model_fn:
    act_func = "selu"
f = "/data6/avharris/ml_sz_clusters/models/"+model_fn
r_model = keras.models.load_model(f)
data = load_dataset("json",data_files="/data6/avharris/ml_sz_clusters/datasets/"+input_fn,split="train")
train_testvalid = data.train_test_split(test_size=0.1, shuffle=False)
df_test = train_testvalid["test"].to_tf_dataset(
    columns=["tot"],
    label_cols=["tsz_8192"],
    batch_size=64,
    prefetch=False,
    shuffle=False)

print("Making predictions on validation set")

predictions = r_model.predict(df_test,batch_size=64)

print("Plotting")
i = 0
ifile = re.sub(".jsonl","",input_fn)
Cy_min = maps_dict[5]["min"]
Cy_max = maps_dict[5]["max"]
if nchannels == 5:
    sub_tot = {2:np.zeros((64,64)),3:np.zeros((64,64)),4:np.zeros((64,64))}
    sub_flat = {2:[],3:[],4:[]}
elif nchannels == 2:
    sub_tot = {0:np.zeros((64,64)),1:np.zeros((64,64))}
    sub_flat = {0:[],1:[]}
res_flat = []
res_tot = np.zeros((64,64))
for inputs,labels in df_test.map(lambda x,y: (x,y)):
    larr = labels.numpy()
    iarr = inputs.numpy()
    for l in range(larr.shape[0]):
        resfile = img_dir+ifile + str(i)+"_res_sep_conv2d_"+m+".png"
        lab = larr[l].reshape(64,64)
        pred = predictions[i].reshape(64,64)
        res = lab - pred
        res_tot = res_tot + res
        res_flat = np.append(res_flat,res.flatten())
        fig = plt.figure()
        plt.imshow(res)
        plt.colorbar()
        #plt.clim(-0.03,0.03)
        plt.savefig(resfile)
        plt.close(fig)
        labfile = img_dir+ifile + str(i)+"_label_sep_conv2d_"+m+".png"
        fig = plt.figure()
        plt.imshow(lab)
        plt.colorbar()
        #plt.clim(0,0.25)
        plt.savefig(labfile)
        plt.close(fig)
        predfile = img_dir+ifile+str(i)+"_predict_sep_conv2d_"+m+".png"
        fig = plt.figure()
        plt.imshow(pred)
        plt.colorbar()
        #plt.clim(0,0.25)
        plt.savefig(predfile)
        plt.close(fig)
        for f in f_arr2:
            subfile = img_dir+ifile + str(i)+"_sub_freq_"+flabels[f]+"_input_sep_conv2d_"+m+".png"
            infile = img_dir+ifile + str(i)+"_freq_"+flabels[f]+"_input_sep_conv2d_"+m+".png"
            T_norm = iarr[l,:,:,f].reshape(64,64)
            fig = plt.figure()
            plt.imshow(T_norm)
            plt.colorbar()
            #plt.clim(0.3,0.8)
            plt.savefig(infile)
            plt.close(fig)
            T_min = maps_dict[f]["min"]
            T_max = maps_dict[f]["max"] 
            T_tot = T_norm*(T_max - T_min) + T_min
            Cy = pred*(Cy_max - Cy_min) + Cy_min
            T_cmb = T_tot - n[f]*Cy
            sub_tot[f] = sub_tot[f] + T_cmb
            sub_flat[f] = np.append(sub_flat[f],T_cmb.flatten())
            fig = plt.figure()
            plt.imshow(T_cmb)
            plt.colorbar()
            #plt.clim(-250,250)
            plt.savefig(subfile)
            plt.close(fig)
        i+=1


fig = plt.figure()
res_avg_flat = res_flat/i
plt.hist(res_avg_flat,range=[-0.006,0.006])
plt.savefig("../plots/"+ifile+"_"+m+"_stack_res_hist.png")
plt.close(fig)

fig = plt.figure()
plt.imshow(res_tot/i)
plt.colorbar()
plt.savefig("../plots/"+ifile+"_"+m+"_stack_res.png")
plt.close(fig)

for f in f_arr2:
    fig = plt.figure()
    plt.imshow(sub_tot[f]/i)
    plt.colorbar()
    plt.savefig("../plots/"+ifile+"_"+m+"_stack_sub_f"+flabels[f]+".png")
    plt.close(fig)
    fig = plt.figure()
    plt.hist(sub_flat[f]/i)
    plt.savefig("../plots/"+ifile+"_"+m+"_stack_sub_hist_f"+flabels[f]+".png")
    plt.close(fig)
print("done")
