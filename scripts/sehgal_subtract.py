import numpy as np
import tensorflow as tf
import keras
import datasets
from datasets import load_dataset
import matplotlib.pyplot as plt
import re
from pixell import reproject,enmap,utils
import healpy as hp

res = np.deg2rad(0.5 / 60.)

def map_min_max(filename,res):
    shape,wcs = enmap.fullsky_geometry(res=res)
    h_map = hp.read_map(filename).astype(np.float64)
    p_map = reproject.healpix2map(h_map,shape=shape,wcs=wcs)
    min_val = np.min(p_map)
    max_val = np.max(p_map)
    return min_val, max_val

maps_dict = {}
files = ["tot_93","tot_145","tot_217","tsz_8192"]
flabels = {2:"93",3:"145",4:"217"}
f_arr = [2,3,4,5]
n = {2:-4.2840e6,3:-2.7685e6,4:-2.1188e4}
for i in range(len(files)):
    f = "../websky/"+ files[i] + ".fits"
    min_val, max_val = map_min_max(f,res)
    maps_dict[f_arr[i]] = {
        "min" : min_val,
        "max" : max_val,
    }


model_fn = "sep_conv2d_linear_small_5ch.keras"
act_func = "linear"
img_dir = "../sehgal_test_imgs_temp_"+act_func+"/"
r_model = keras.models.load_model("../models/"+model_fn)
input_fn = "sehgal_small_5ch.jsonl"
data = load_dataset("json",data_files="../data/"+input_fn,split="train")

train_testvalid = data.train_test_split(test_size=0.05)

df_test = train_testvalid["test"].to_tf_dataset(
        columns=["tot"],
        label_cols=["tsz_8192"],
        batch_size=64,
        prefetch=False,
        shuffle=False)

predictions = r_model.predict(df_test,batch_size=64)
i = 0
ifile = re.sub(".jsonl","",input_fn)
Cy_min = maps_dict[5]["min"]
Cy_max = maps_dict[5]["max"]
res_flat = []
res_tot = np.zeros((64,64))
sub_tot = {2:np.zeros((64,64)),3:np.zeros((64,64)),4:np.zeros((64,64))}
sub_flat = {2:[],3:[],4:[]}
for inputs,labels in df_test.map(lambda x,y: (x,y)):
    larr = labels.numpy()
    iarr = inputs.numpy()
    for l in range(larr.shape[0]):
        resfile = img_dir+ifile + str(i)+"_res_sep_conv2d_"+act_func+".png"
        lab = larr[l].reshape(64,64)
        pred = predictions[i].reshape(64,64)
        res = lab - pred
        res_tot = res_tot + res
        res_flat = np.append(res_flat,res.flatten())
        fig = plt.figure()
        plt.imshow(res)
        plt.colorbar()
        plt.savefig(resfile)
        plt.close(fig)
        labfile = img_dir+ifile + str(i)+"_label_sep_conv2d_"+act_func+".png"
        fig = plt.figure()
        plt.imshow(lab)
        plt.colorbar()
        plt.savefig(labfile)
        plt.close(fig)
        predfile = img_dir+ifile+str(i)+"_predict_sep_conv2d_"+act_func+".png"
        fig = plt.figure()
        plt.imshow(pred)
        plt.colorbar()
        plt.savefig(predfile)
        plt.close(fig)
        for f in [2,3,4]:
            subfile = img_dir+ifile + str(i)+"_sub_freq_"+flabels[f]+"_input_sep_conv2d_"+act_func+".png"
            infile = img_dir+ifile + str(i)+"_freq_"+flabels[f]+"_input_sep_conv2d_"+act_func+".png"
            T_norm = iarr[l,:,:,f].reshape(64,64)
            fig = plt.figure()
            plt.imshow(T_norm)
            plt.colorbar()
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
            plt.savefig(subfile)
            plt.close(fig)
        i+=1


fig = plt.figure()
res_avg_flat = res_flat/i
logbins = np.logspace(np.log10(np.min(res_avg_flat)),np.log10(1.+np.max(res_avg_flat)),10)
plt.hist(res_avg_flat,logbins)
plt.xscale("log")
plt.yscale("log")
plt.savefig("../plots/"+ifile+"_"+act_func+"_stack_res_hist.png")
plt.close(fig)

fig = plt.figure()
plt.imshow(res_tot/i)
plt.colorbar()
plt.savefig("../plots/"+ifile+"_"+act_func+"_stack_res.png")
plt.close(fig)

for f in [2,3,4]:
    fig = plt.figure()
    plt.imshow(sub_tot[f]/i)
    plt.colorbar()
    plt.savefig("../plots/"+ifile+"_"+act_func+"_stack_sub_f"+flabels[f]+".png")
    plt.close(fig)
    fig = plt.figure()
    plt.hist(sub_flat[f]/i)
    plt.savefig("../plots/"+ifile+"_"+act_func+"_stack_sub_hist_f"+flabels[f]+".png")
    plt.close(fig)
print("done")
