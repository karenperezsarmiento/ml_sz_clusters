from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
fn = "small_5ch_b_n5"
data = load_dataset("json",data_files="/data6/kaper/ml_sz_clusters/datasets/"+fn+".jsonl",split="train")
train_testvalid = data.train_test_split(test_size=0.1)

print(np.array(train_testvalid["train"]["tot"])[0].shape)

for i in range(20):
    fig = plt.figure()
    plt.imshow(np.array(train_testvalid["train"]["tot"])[i][:,:,2])
    plt.colorbar()
    plt.savefig("../test_plots/bn_"+str(i)+".png")
    plt.close(fig)
