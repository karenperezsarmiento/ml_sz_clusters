from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import argparse as ap

parser = ap.ArgumentParser(description="Create cutouts around websky halos to use as input in CNN")
parser.add_argument("-f","--filename",type=str,help="Filename of jsonl file to read")
parser.add_argument("-nc","--nchannels",type=str,help="Number of channels")
parser.add_argument("-fol","--folder",type=str,help="Folder in which to save images")
parser.add_argument("-min","--min_row",type=str,help="Minimum row to plot")
parser.add_argument("-max","--max_row",type=str,help="Maximum row to plot")
parser.add_argument("-mr","--plot_mr",type=str,help="If True, plots the Mass and Redshift arrays")
args = parser.parse_args()
fn = args.filename
nchannels = int(args.nchannels)
folder = args.folder
min_row = int(args.min_row)
max_row = int(args.max_row)
if args.plot_mr == "True":
    plot_mr = True
else:
    plot_mr = False

m = fn.replace(".jsonl", "")

data = load_dataset("json",data_files="/data6/avharris/ml_sz_clusters/datasets/"+fn,split="train")
train_testvalid = data.train_test_split(test_size=0.1)

print("Loaded dataset")
print("Forming tot  array...")

tot_arr = np.array(train_testvalid["train"]["tot"])
print(np.shape(tot_arr))

print("Forming tsz array...")
tsz_arr = np.array(train_testvalid["train"]["tsz_8192"])
print(np.shape(tsz_arr))

layer_dict = {}   # Records which "slice" of tot_arr contains which 64x64 input arrays
if (nchannels == 1):
    layer_dict = {"93":0}
elif (nchannels == 2):
    layer_dict = {"93":0, "145":1}
elif (nchannels == 3):
    layer_dict = {"Mass":0, "Redshift":1, "93":2}
elif (nchannels == 5):
    layer_dict = {"Mass":0, "Redshift":1, "93":3, "145":4, "217":5}
elif (nchannels == 6):
    layer_dict = {"93":0, "145":1, "217":2}
    

for row in range(min_row, max_row+1):
    if plot_mr:
        if "Mass" in layer_dict:
            plt.imshow(tot_arr[row][:,:,layer_dict["Mass"]])
            plt.title("Mass array of training example "+str(row))
            plt.colorbar()
            plt.savefig(folder + "row_"+str(row)+"_mass"+"_of_"+m+".png")
            plt.close("all")
   
        if "Redshift" in layer_dict:
            plt.imshow(tot_arr[row][:,:,layer_dict["Redshift"]])
            plt.title("Redshift array of training example "+str(row))
            plt.colorbar()
            plt.savefig(folder + "row_"+str(row)+"_redshift"+"_of_"+m+".png")
            plt.close("all")

    if "93" in layer_dict:
        plt.imshow(tot_arr[row][:,:,layer_dict["93"]])
        plt.title("Frequency 93 array of training example "+str(row))
        plt.colorbar()
        plt.savefig(folder + "row_"+str(row)+"_93"+"_of_"+m+".png")
        plt.close("all")

    if "145" in layer_dict:
        plt.imshow(tot_arr[row][:,:,layer_dict["145"]])
        plt.title("Frequency 145 array of training example "+str(row))
        plt.colorbar()
        plt.savefig(folder + "row_"+str(row)+"_145"+"_of_"+m+".png")
        plt.close("all")

    if "217" in layer_dict:
        plt.imshow(tot_arr[row][:,:,layer_dict["217"]])
        plt.title("Frequency 217 array of training example "+str(row))
        plt.colorbar()
        plt.savefig(folder + "row_"+str(row)+"_217"+"_of_"+m+".png")
        plt.close("all")

    plt.imshow(tsz_arr[row])
    plt.title("True cluster of training example "+str(row))
    plt.colorbar()
    plt.savefig(folder + "row_"+str(row)+"_true_cluster"+"_of_"+m+".png")
    plt.close("all")

print("Done")
