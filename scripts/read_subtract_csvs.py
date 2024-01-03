import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

act_func = "linear"
ifile = "high_offset_small_5ch"
file_loc = "../data/csv_files_"+act_func+"/"

def sub_dict_string_to_np_array(sub_dict):
    output = {}
    for i in sub_idc:
        output[i] = np.array(sub_dict[i].replace("[", "").replace("]", "").split(", ")).astype("float")
    return output

#res_flat = np.array(pd.read_csv(file_loc+act_func+"_res_flat.csv").columns).astype("float")
#res_tot = np.array(pd.read_csv(file_loc+act_func+"_res_tot.csv").columns).astype("float")
res_norms = np.array(pd.read_csv(file_loc+ifile+"_res_norms.csv").columns).astype("float")
off_norms = np.array(pd.read_csv(file_loc+ifile+"_off_norms.csv").columns).astype("float")
off_vals_df = pd.read_csv(file_loc+ifile+"_off_vals.csv")
dec_vals = np.array(off_vals_df.columns).astype("float")
ra_vals = off_vals_df.iloc[3].to_numpy().astype("float")

sub_idc = ['2', '3', '4']
flabels = {"2":"93","3":"145","4":"217"}
#with open(file_loc+act_func+"_sub_flat.csv") as f:
#    reader = csv.DictReader(f)
#    sub_flat = next(reader)
#    sub_flat = sub_dict_string_to_np_array(sub_flat)
#with open(file_loc+act_func+"_sub_tot.csv") as f:
#    reader = csv.DictReader(f)
#    sub_tot = next(reader)
#    sub_tot = sub_dict_string_to_np_array(sub_tot)
with open(file_loc+ifile+"_sub_norms.csv") as f:
    reader = csv.DictReader(f)
    sub_norms = next(reader)
    sub_norms=sub_dict_string_to_np_array(sub_norms)

print(type(sub_norms['2']))

plot_res_vs_offset = False
plot_freq_T_cmb_vs_offset = True
if plot_res_vs_offset:
    # Plot the goodness of the fit against the offset
    fig = plt.figure()
    plt.scatter(off_norms, res_norms)
    a, b = np.polyfit(off_norms, res_norms, 1)
    plt.plot(off_norms, a*np.array(off_norms)+b, "-r")
    corr_coeff = str(getattr(pearsonr(off_norms, res_norms), "statistic"))
    plt.title("Pearson correlation coefficient: " + corr_coeff)
    plt.xlabel("Offset norm [radians]")
    plt.ylabel("Norm of all pixels in residual")
    plt.savefig("../plots/"+ifile+"_"+act_func+"_res_vs_offset.png")
    plt.close(fig)

"""
if plot_freq_T_cmb_vs_offset:
    for i in sub_idc:
        fig = plt.figure()
        plt.scatter(off_norms, sub_norms[i], s=(1.25/4.0)**2)
        a, b = np.polyfit(off_norms, sub_norms[i], 1)
        plt.plot(off_norms, a*np.array(off_norms)+b, "-r")
        corr_coeff = str(getattr(pearsonr(off_norms, sub_norms[i]), "statistic"))
        plt.title("Pearson correlation coefficient: " + corr_coeff)
        plt.xlabel("Offset norm [radians]")
        plt.ylabel("Norm of all pixels in T_cmb for freq "+flabels[i])
        plt.savefig("../plots/"+ifile+"_"+act_func+"_sub_"+flabels[i]+"_vs_offset.png")
        plt.close(fig)
"""
