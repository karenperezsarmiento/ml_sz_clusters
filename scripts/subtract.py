import numpy as np
import tensorflow as tf
import keras
import datasets
from datasets import load_dataset
import matplotlib.pyplot as plt
import re
from pixell import reproject,enmap,utils
import healpy as hp
from scipy.stats import pearsonr
import csv

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
    f = "/data6/kaper/ml_sz_clusters/websky/"+ files[i] + ".fits"
    min_val, max_val = map_min_max(f,res)
    maps_dict[f_arr[i]] = {
        "min" : min_val,
        "max" : max_val,
    }

model_fn = "sep_conv2d_linear_small_5ch_b_n10_noap.keras"
act_func = "linear"
img_dir = "../test_imgs_temp_" + act_func + "_200_epochs/"
r_model = keras.models.load_model("/data6/avharris/ml_sz_clusters/models/"+model_fn)
input_fn = "small_5ch_b_n5_noap.jsonl"
data = load_dataset("json",data_files="/data6/avharris/ml_sz_clusters/datasets/"+input_fn,split="train")
ifile = re.sub(".jsonl","",input_fn)

"""
Before running this script, ensure that in the ml_sz_clusters directory there are the following directories:
/test_imgs_temp_ + act_func + /
/data/csv_files_ + act_func + /

label, prediction, and residual images will be stored in /test_imgs_temp_ + act_func + /
with filenames  ifile + i + desc + act_func +.png
where i is an integer and desc is _res_sep_conv2d_ or _label_sep_conv2d_ or _predict_sep_conv2d_

Plots of residuals stacked and histograms are stored in /plots/ with filenames starting with ifile + act_func

Csv files storing important arrays are stored in /data/csv_files_ + act_func + /
The filenames are of the form ifile + arr_name + .csv
"""


train_testvalid = data.train_test_split(test_size=0.05)

df_test = train_testvalid["test"].to_tf_dataset(
        columns=["tot"],
        label_cols=["tsz_8192"],
        batch_size=64,
        prefetch=False,
        shuffle=False)

predictions = r_model.predict(df_test,batch_size=64)
i = 0
Cy_min = maps_dict[5]["min"]
Cy_max = maps_dict[5]["max"]
res_flat = []
res_tot = np.zeros((64,64))
sub_tot = {2:np.zeros((64,64)),3:np.zeros((64,64)),4:np.zeros((64,64))}
sub_flat = {2:[],3:[],4:[]}
sub_norms = {2:[], 3:[], 4:[]}
res_norms = []    # the norm of all pixels in the residual
"""
off_norms = []    # the norm of the offset in radians. off_norm[i] corresponds to res_norm[i]
off_vals = []     # the values of the offset in radians for the declination angle and the ascension angle
off_vals.append([])   # off_norms[0][i] is the declination angle that corresponds to res_norms[i]
off_vals.append([])   # off_norms[1][i] is the ascension angle that corresponds to res_norms[i]
"""
for inputs,labels in df_test.map(lambda x,y: (x,y)):
    larr = np.array(labels)
    """
    offset = np.array(labels["offset"])
    off_vals[0].extend(offset[:,0])
    off_vals[1].extend(offset[:,1])
    off_norms.extend(np.linalg.norm(offset, axis = 1))
    """
    iarr = inputs.numpy()
    for l in range(larr.shape[0]):
        resfile = img_dir+ifile + str(i)+"_res_sep_conv2d_"+act_func+".png"
        lab = larr[l].reshape(64,64)
        pred = predictions[i].reshape(64,64)
        res = lab - pred
        res_tot = res_tot + res
        res_flat = np.append(res_flat,res.flatten())
        res_norms.append(np.linalg.norm(res))
        fig = plt.figure()
        plt.imshow(res)
        plt.savefig(resfile)
        plt.close(fig)
        labfile = img_dir+ifile + str(i)+"_label_sep_conv2d_"+act_func+".png"
        fig = plt.figure()
        plt.imshow(lab)
        plt.savefig(labfile)
        plt.close(fig)
        predfile = img_dir+ifile+str(i)+"_predict_sep_conv2d_"+act_func+".png"
        fig = plt.figure()
        plt.imshow(pred)
        plt.savefig(predfile)
        plt.close(fig)
        for f in [2,3,4]:
            subfile = img_dir+ifile + str(i)+"_sub_freq_"+flabels[f]+"_input_sep_conv2d_"+act_func+".png"
            infile = img_dir+ifile + str(i)+"_freq_"+flabels[f]+"_input_sep_conv2d_"+act_func+".png"
            T_norm = iarr[l,:,:,f].reshape(64,64)
            fig = plt.figure()
            plt.imshow(T_norm)
            plt.savefig(infile)
            plt.close(fig)
            T_min = maps_dict[f]["min"]
            T_max = maps_dict[f]["max"] 
            T_tot = T_norm*(T_max - T_min) + T_min
            Cy = pred*(Cy_max - Cy_min) + Cy_min
            T_cmb = T_tot - n[f]*Cy
            sub_tot[f] = sub_tot[f] + T_cmb
            sub_flat[f] = np.append(sub_flat[f],T_cmb.flatten())
            sub_norms[f].append(np.linalg.norm(T_cmb))
            fig = plt.figure()
            plt.imshow(T_cmb)
            plt.savefig(subfile)
            plt.close(fig)
        i+=1

fig = plt.figure()
plt.hist(res_flat/i)
plt.savefig("../plots/"+ifile+"_"+act_func+"_stack_res_hist.png")
plt.close(fig)

fig = plt.figure()
plt.imshow(res_tot/i)
plt.savefig("../plots/"+ifile+"_"+act_func+"_stack_res.png")
plt.close(fig)

for f in [2,3,4]:
    fig = plt.figure()
    plt.imshow(sub_tot[f]/i)
    plt.savefig("../plots/"+ifile+"_"+act_func+"_stack_sub_f"+flabels[f]+".png")
    plt.close(fig)
    fig = plt.figure()
    plt.hist(sub_flat[f]/i)
    plt.savefig("../plots/"+ifile+"_"+act_func+"_stack_sub_hist_f"+flabels[f]+".png")
    plt.close(fig)
print("plots made")

"""
# Save important matrices to csv files
# res_flat
with open("../data/csv_files_"+act_func+"/"+ifile+"_res_flat.csv", "w", newline="") as f:
    writer=csv.writer(f)
    writer.writerow(res_flat.tolist())
# res_tot
with open("../data/csv_files_"+act_func+"/"+ifile+"_res_tot.csv", "w", newline="") as f:
    writer=csv.writer(f)
    writer.writerows(res_tot.tolist())
# sub_tot
with open("../data/csv_files_"+act_func+"/"+ifile+"_sub_tot.csv", "w", newline="") as f:
    writer=csv.DictWriter(f, fieldnames=[2, 3, 4])
    writer.writeheader()
    writer.writerow(sub_tot)
# sub_flat
with open("../data/csv_files_"+act_func+"/"+ifile+"_sub_flat.csv", "w", newline="") as f:
    writer=csv.DictWriter(f, fieldnames=[2, 3, 4])
    writer.writeheader()
    writer.writerow(sub_flat)
# sub_norms
with open("../data/csv_files_"+act_func+"/"+ifile+"_sub_norms.csv", "w", newline="") as f:
    writer=csv.DictWriter(f, fieldnames=[2, 3, 4])
    writer.writeheader()
    writer.writerow(sub_norms)
# res_norms
with open("../data/csv_files_"+act_func+"/"+ifile+"_res_norms.csv", "w", newline="") as f:
    writer=csv.writer(f)
    writer.writerow(res_norms)
# off_norms
with open("../data/csv_files_"+act_func+"/"+ifile+"_off_norms.csv", "w", newline="") as f:
    writer=csv.writer(f)
    writer.writerow(off_norms)
# off_vals
with open("../data/csv_files_"+act_func+"/"+ifile+"_off_vals.csv", "w", newline="") as f:
    writer=csv.writer(f)
    writer.writerows(off_vals)
"""
print("done")
