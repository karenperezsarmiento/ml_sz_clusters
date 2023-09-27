import numpy as np
import tensorflow as tf
import keras
import datasets
from datasets import load_dataset
import matplotlib.pyplot as plt

r_model = keras.models.load_model("../models/sep_conv2d_small_3ch.keras")
datafile = "../data/small_3ch.jsonl"
data = load_dataset("json",data_files=datafile,split="train")

train_testvalid = data.train_test_split(test_size=0.01)

df_test = train_testvalid["test"].to_tf_dataset(
        columns=["tot_93"],
        label_cols=["tsz_8192"],
        batch_size=1,
        prefetch=False,
        shuffle=False)

print(type(df_test))
print(len(df_test))

i = 0
for inputs,labels in df_test.map(lambda x,y: (x,y)):
    ifile = "../test_imgs/"+str(i)+"_input.png"
    ofile = "../test_imgs/"+str(i)+"_label.png"
    fig = plt.figure()
    larr = labels.numpy()
    plt.imshow(larr.reshape(64,64))
    plt.savefig(ofile)
    plt.close(fig)
    iarr = inputs.numpy()
    iarr = iarr[0,:,:,0]
    fig = plt.figure()
    plt.imshow(iarr.reshape(64,64))
    plt.savefig(ifile)
    plt.close(fig)
    i+=1

predictions = r_model.predict(df_test)

print(predictions.shape)

for i in range(predictions.shape[0]):
    ofile = "../test_imgs/"+str(i)+"_predict.png"
    fig = plt.figure()
    plt.imshow(predictions[i].reshape(64,64))
    plt.savefig(ofile)
    plt.close(fig)

print("done")

