import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras import datasets as tfdatasets
import datasets
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from datasets import load_dataset
import numpy as np
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Dense, Activation, Cropping2D, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Concatenate,SeparableConv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, UpSampling2D, UpSampling3D, Add
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import model_to_dot
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
import tensorflow.keras.backend as K
import argparse as ap
from pixell import reproject,enmap,utils
import healpy as hp
import re
import time

start = time.time()

parser = ap.ArgumentParser(description="Train CNN, test and make plot")
parser.add_argument("-if","--input_file",type=str,help="Name of input dataset file")
parser.add_argument("-nc","--num_channels",type=int,help="Number of channels in input cutout (1 or 3 or 5)")
parser.add_argument("-af","--activation_func",type=str,help="Activation function in CNN (relu,selu,linear,etc)")
parser.add_argument("-ep","--epochs", type=int,help="Number of epochs to train")
args = parser.parse_args()
input_fn = args.input_file
nchannels = args.num_channels
act_func = args.activation_func
num_epochs = int(args.epochs)
#infile = "/data6/kaper/ml_sz_clusters/datasets/"+input_fn
infile = "/data6/avharris/ml_sz_clusters/datasets/"+input_fn

data = load_dataset("json",data_files=infile,split="train")

batch_sz = 64

freq_key = "tot"
train_testvalid = data.train_test_split(test_size=0.1)
df_train = train_testvalid["train"].to_tf_dataset(
    columns=[freq_key],
    label_cols=["tsz_8192"],
    batch_size=batch_sz,
    prefetch=False,
    shuffle=True)
df_test = train_testvalid["test"].to_tf_dataset(
    columns=[freq_key],
    label_cols=["tsz_8192"],
    batch_size=batch_sz,
    prefetch=False,
    shuffle=False)

N_CLASSES = 1
window_function = 3
droupout_rate = 0.00
def ResUNet(img_shape):
    inputs = Input(shape = img_shape)
    #tabs_input = Input(shape = img_shape)
    ###encoding block 1
    iden1 = SeparableConv2D(64, 1, activation = None, padding='same', kernel_initializer = 'he_normal')(inputs)
    #conv1 = Dropout(droupout_rate)(inputs)
    conv1 = SeparableConv2D(64, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs) #(conv1)
    conv1 = Activation(act_func)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(droupout_rate)(conv1)
    conv1 = SeparableConv2D(64, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = Activation(act_func)(conv1)
    conv1 = BatchNormalization()(conv1)
    add1  = Add()([iden1,conv1])
    pool1 = add1 #MaxPooling2D()(add1) #[40 - > 40]

    ###encoding block2
    iden2 = SeparableConv2D(64, 1, activation = None, padding='same', kernel_initializer = 'he_normal')(pool1) #[40 - > 40]
    conv2 = Dropout(droupout_rate)(pool1)
    conv2 = SeparableConv2D(64, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv2) #[40 - > 40]
    conv2 = Activation(act_func)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(droupout_rate)(conv2)
    conv2 = SeparableConv2D(64, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = Activation(act_func)(conv2)
    conv2 = BatchNormalization()(conv2)
    add2  = Add()([iden2,conv2])
    pool2 = add2 #MaxPooling2D()(add2) #[40 -> 40]

    ###encoding block3
    iden3 = SeparableConv2D(128, 1, strides=(2,2), activation = None, padding='same', kernel_initializer = 'he_normal')(pool2) #[40 -> 20]
    conv3 = Dropout(droupout_rate)(pool2)
    conv3 = SeparableConv2D(128, window_function, strides=(2,2), activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv3) #200 #[40 -> 20]
    conv3 = Activation(act_func)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(droupout_rate)(conv3)
    conv3 = SeparableConv2D(128, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = Activation(act_func)(conv3)
    conv3 = BatchNormalization()(conv3)
    add3 = Add()([iden3,conv3])
    pool3= add3 #MaxPooling2D()(add3) #[20 -> 20]

    ###encoding block4
    iden4 = SeparableConv2D(128, 1, activation=None, padding='same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Dropout(droupout_rate)(pool3)
    conv4 = SeparableConv2D(128, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Activation(act_func)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(droupout_rate)(conv4)
    conv4 = SeparableConv2D(128, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Activation(act_func)(conv4)
    conv4 = BatchNormalization()(conv4)
    add4 = Add()([iden4,conv4])
    pool4 = add4 #MaxPooling2D()(add4) #[20 -> 20]

    ###encoding block4.1
    iden4 = SeparableConv2D(256, 1, strides=(2,2), activation=None, padding='same', kernel_initializer = 'he_normal')(pool4) # [20 -> 10]
    conv4 = Dropout(droupout_rate)(pool4)
    conv4 = SeparableConv2D(256, window_function, strides=(2,2), activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv4) # [20->10]
    conv4 = Activation(act_func)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(droupout_rate)(conv4)
    conv4 = SeparableConv2D(256, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Activation(act_func)(conv4)
    conv4 = BatchNormalization()(conv4)
    add5 = Add()([iden4,conv4])
    pool4 = add5 #MaxPooling2D()(add4) #[10 -> 10]

    ###encoding block5.2
    iden4 = SeparableConv2D(256, 1, activation=None, padding='same', kernel_initializer = 'he_normal')(pool4) # [10 -> 10]
    conv4 = Dropout(droupout_rate)(pool4)
    conv4 = SeparableConv2D(256, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv4) # [10->10]
    conv4 = Activation(act_func)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(droupout_rate)(conv4)
    conv4 = SeparableConv2D(256, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Activation(act_func)(conv4)
    conv4 = BatchNormalization()(conv4)
    add5 = Add()([iden4,conv4])
    pool4 = add5 #MaxPooling2D()(add4) #[10 -> 10]

    ###bridge1
    iden5 = SeparableConv2D(512, 1, strides=(2,2), activation=None, padding='same', kernel_initializer = 'he_normal')(pool4)  #[10 -> 5]
    conv5 = Dropout(droupout_rate)(pool4)
    conv5 = SeparableConv2D(512, window_function, strides=(2,2), activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv5) #[10 -> 5]
    conv5 = Activation(act_func)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Dropout(droupout_rate)(conv5)
    conv5 = SeparableConv2D(512, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv5) #[5 -> 5]
    conv5 = Activation(act_func)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Add()([iden5,conv5])

    ###bridge2
    iden5 = SeparableConv2D(512, 1, activation=None, padding='same', kernel_initializer = 'he_normal')(conv5)  #[5 -> 5]
    conv5 = Dropout(droupout_rate)(conv5)
    conv5 = SeparableConv2D(512, window_function,  activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv5) #[5 -> 5]
    conv5 = Activation(act_func)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Dropout(droupout_rate)(conv5)
    conv5 = SeparableConv2D(512, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv5) #[5 -> 5]
    conv5 = Activation(act_func)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Add()([iden5,conv5])

    ###decoding block0.8
    up6 = UpSampling3D()(conv5) #[5 -> 10]
    concat6 = Concatenate(axis=3)([up6,add5])
    iden6 = SeparableConv2D(256, 1, activation=None, padding='same', kernel_initializer = 'he_normal')(concat6)
    conv6 = Dropout(droupout_rate)(concat6)
    conv6 = SeparableConv2D(256, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = Activation(act_func)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(droupout_rate)(conv6)
    conv6 = SeparableConv2D(256, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = Activation(act_func)(conv6)
    conv6 = BatchNormalization()(conv6)
    add6 = Add()([iden6,conv6])

    ###decoding block0.9
    up6 = add6 #UpSampling2D()(conv5) #[10 -> 10]
    #concat6 = Concatenate(axis=3)([up6,add5])
    iden6 = SeparableConv2D(256, 1, activation=None, padding='same', kernel_initializer = 'he_normal')(up6) #(concat6)
    conv6 = Dropout(droupout_rate)(up6) #(concat6)
    conv6 = SeparableConv2D(256, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = Activation(act_func)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(droupout_rate)(conv6)
    conv6 = SeparableConv2D(256, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = Activation(act_func)(conv6)
    conv6 = BatchNormalization()(conv6)
    add6 = Add()([iden6,conv6])

    ###decoding block1
    up6 = UpSampling2D()(add6) #[10 -> 20]
    concat6 = Concatenate(axis=3)([up6,add4])
    iden6 = SeparableConv2D(128, 1, activation=None, padding='same', kernel_initializer = 'he_normal')(concat6)
    conv6 = Dropout(droupout_rate)(concat6)
    conv6 = SeparableConv2D(128, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = Activation(act_func)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(droupout_rate)(conv6)
    conv6 = SeparableConv2D(128, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = Activation(act_func)(conv6)
    conv6 = BatchNormalization()(conv6)
    add6 = Add()([iden6,conv6])

    ###decoding block2
    up7 = add6 #UpSampling2D()(add6) #[20 -> 20]
    #concat7 = Concatenate(axis=3)([up7,add3])
    iden7 = SeparableConv2D(128, 1, activation=None, padding='same', kernel_initializer = 'he_normal')(up7) #(concat7)
    conv7 = Dropout(droupout_rate)(up7) #(concat7)
    conv7 = SeparableConv2D(128, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = Activation(act_func)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(droupout_rate)(conv7)
    conv7 = SeparableConv2D(128, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = Activation(act_func)(conv7)
    conv7 = BatchNormalization()(conv7)
    add7 = Add()([iden7,conv7])

    ###decoding block3
    up8 = UpSampling2D()(add7) #[20 -> 40]
    concat8 = Concatenate(axis=3)([up8,add2])
    iden8 = SeparableConv2D(64, 1, activation=None, padding='same', kernel_initializer = 'he_normal')(concat8)
    conv8 = Dropout(droupout_rate)(concat8)
    conv8 = SeparableConv2D(64, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = Activation(act_func)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Dropout(droupout_rate)(conv8)
    conv8 = SeparableConv2D(64, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = Activation(act_func)(conv8)
    conv8 = BatchNormalization()(conv8)
    add8 = Add()([iden8,conv8])

    ###decoding block4
    up9 = add8#UpSampling2D()(add8) #[40 -> 40]
    #concat9 = Concatenate(axis=3)([up9,add1])
    iden9 = SeparableConv2D(64,1,activation=None, padding='same', kernel_initializer = 'he_normal')(up9) #(concat9)
    conv9 = Dropout(droupout_rate)(up9) #(concat9)
    conv9 = SeparableConv2D(64, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Activation(act_func)(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(droupout_rate)(conv9)
    conv9 = SeparableConv2D(64, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Activation(act_func)(conv9)
    conv9 = BatchNormalization()(conv9)
    add9 = Add()([iden9,conv9])

    ### Few extra layers without activations
    conv9 = Dropout(droupout_rate)(add9)
    conv9 = SeparableConv2D(64, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(droupout_rate)(conv9)
    conv9 = SeparableConv2D(64, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(droupout_rate)(conv9)
    conv9 = SeparableConv2D(64, window_function, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)


    ### Final layer
    conv10 = SeparableConv2D(N_CLASSES, window_function, activation ='linear', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    print (conv10)
    #linear activation as used in Caldeira et al. 18

    model = Model(inputs=inputs, outputs=conv10)

    return model

# Create a MirroredStrategy.
#strategy = tf.distribute.MirroredStrategy()
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
#with strategy.scope():

model = ResUNet((64,64,nchannels))
print (model.summary())

cp_path = "../checkpoints/"+act_func+"/sep_conv2d_"+act_func+"_"+re.sub(".jsonl","",input_fn)+"_cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(cp_path,save_weights_only=True,verbose=1,save_freq='epoch')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_delta=1e-10, cooldown=1, min_lr=0.)

class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model.optimizer.lr.numpy()))

class PrintLoss(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch > 2:
            print('\nLoss for epoch {} is {}'.format(epoch + 1, model.history.history['loss'][-1]))

callbacks_list = [cp_callback, reduce_lr, PrintLR()]


opt = Adam(learning_rate = 1e-5) #1e-5 best
model.compile(optimizer=opt, loss='mean_squared_error',metrics=["mse"])

# Train the model on all available devices.
history = model.fit(df_train, validation_data=df_test, epochs=num_epochs,callbacks=callbacks_list)

print("_______________________")
print("Done fitting model")

ofile = re.sub(".jsonl",".keras",input_fn)
pltfile = re.sub(".jsonl",".png",input_fn)
ofile = "/data6/avharris/ml_sz_clusters/models/sep_conv2d_"+act_func+"_"+ofile
pltfile = "../plots/sep_conv2d_"+act_func+"_"+pltfile
model.save(ofile)

#history= model.history
print(history.history.keys())
fig = plt.figure()  
plt.plot(np.array(history.history['loss']), color='blue', label='Training loss');
plt.yscale('log') #  training sample is 4 times larger
#plot(np.array(history.history['Kappa_model_loss']), color='blue', linestyle='--', label='Kappa Model Training loss')
#plot(np.array(hist ory.history['tSZ_model_loss']), color='blue', linestyle='-.', label='tSZ model Training loss')
plt.plot(history.history['val_loss'], color='orange', label='Validation Loss')
#plot(history.history['val_Kappa_model_loss'], color='orange', linestyle='--', label='Kappa Model Validation Loss')
#plot(history.histo ry['val_tSZ_model_loss'], color='orange', linestyle='-.', label='tSZ model Validation Loss')
plt.ylabel('Loss'); plt.xlabel('# Epochs')
plt.legend()
plt.savefig(pltfile)
plt.close(fig)
print("Done loss plot")

m = re.sub(".keras","",ofile)
m = re.sub("/data6/kaper/ml_sz_clusters/models/","",m)
f = re.sub(".jsonl","",input_fn)
img_dir = "../test_imgs/"+m+"_"+f+"/"

print(m)
print(f)

res = np.deg2rad(0.5 / 60.)

def map_min_max(filename,res):
    shape,wcs = enmap.fullsky_geometry(res=res)
    h_map = hp.read_map(filename).astype(np.float64)
    p_map = reproject.healpix2map(h_map,shape=shape,wcs=wcs)
    min_val = np.min(p_map)
    max_val = np.max(p_map)
    return min_val, max_val

maps_dict = {}

maps_dict = {}
if nchannels == 5:
    files = ["tot_93","tot_145","tot_217","tsz_8192"]
    flabels = {2:"93",3:"145",4:"217"}
    f_arr = [2,3,4,5]
    f_arr2 = [2,3,4]
    n = {2:-4.2840e6,3:-2.7685e6,4:-2.1188e4}
else:
    files = ["tot_93","tsz_8192"]
    flabels = {2:"93"}
    f_arr = [2,5]
    f_arr2 = [2]
    n = {2:-4.2840e6}


for i in range(len(files)):
    ff = "/data6/kaper/ml_sz_clusters/websky/"+ files[i] + ".fits"
    min_val, max_val = map_min_max(ff,res)
    maps_dict[f_arr[i]] = {
        "min" : min_val,
        "max" : max_val,
    }

predictions = model.predict(df_test,batch_size = batch_sz)
i = 0
Cy_min = maps_dict[5]["min"]
Cy_max = maps_dict[5]["max"]
res_flat = []
res_tot = np.zeros((64,64))
if nchannels ==5:
    sub_tot = {2:np.zeros((64,64)),3:np.zeros((64,64)),4:np.zeros((64,64))}
    sub_flat = {2:[],3:[],4:[]}
else:
    sub_tot = {2:np.zeros((64,64))}
    sub_flat = {2:[]}

cmap = "bwr"

for inputs,labels in df_test.map(lambda x,y: (x,y)):
    larr = labels.numpy()
    iarr = inputs.numpy()
    for l in range(larr.shape[0]):
        resfile = img_dir+ str(i)+"_res.png"        
        lab = larr[l].reshape(64,64)
        pred = predictions[i].reshape(64,64)
        res = lab - pred
        res_tot = res_tot + res
        res_flat = np.append(res_flat,res.flatten())
        fig = plt.figure()
        plt.imshow(res,cmap = cmap)
        plt.colorbar()
        plt.clim(-0.03,0.03)
        plt.savefig(resfile)
        plt.close(fig)
        labfile = img_dir + str(i)+"_label.png"
        fig = plt.figure()
        plt.imshow(lab,cmap = cmap)
        plt.colorbar()
        plt.clim(0,0.25)
        plt.savefig(labfile)
        plt.close(fig)
        predfile = img_dir +str(i)+"_predict.png"
        fig = plt.figure()
        plt.imshow(pred,cmap = cmap)
        plt.colorbar()
        plt.clim(0,0.25)
        plt.savefig(predfile)
        plt.close(fig)
        for ff in f_arr2:
            subfile = img_dir + str(i)+"_sub_freq_"+flabels[ff]+"_input.png"
            infile = img_dir + str(i)+"_freq_"+flabels[ff]+"_input.png"
            T_norm = iarr[l,:,:,ff].reshape(64,64)
            fig = plt.figure()
            plt.imshow(T_norm,cmap = cmap)
            plt.colorbar()
            plt.clim(0.3,0.8)
            plt.savefig(infile)
            plt.close(fig)
            T_min = maps_dict[ff]["min"]
            T_max = maps_dict[ff]["max"]
            T_tot = T_norm*(T_max - T_min) + T_min
            Cy = pred*(Cy_max - Cy_min) + Cy_min
            T_cmb = T_tot - n[ff]*Cy
            sub_tot[ff] = sub_tot[ff] + T_cmb
            sub_flat[ff] = np.append(sub_flat[ff],T_cmb.flatten())
            fig = plt.figure()
            plt.imshow(T_cmb,cmap=cmap)
            plt.colorbar()
            plt.clim(-250,250)
            plt.savefig(subfile)
            plt.close(fig)
        i+=1

print("Done input, labels, preds stamps")

fig = plt.figure()
res_avg_flat = res_flat/i
plt.hist(res_avg_flat,range=[-0.006,0.006])
plt.savefig("../plots/"+f+"_"+m+"_stack_res_hist.png")
plt.close(fig)

fig = plt.figure()
plt.imshow(res_tot/i)
plt.colorbar()
plt.savefig("../plots/"+f+"_"+m+"_stack_res.png")
plt.close(fig)

for ff in f_arr2:
    fig = plt.figure()
    plt.imshow(sub_tot[ff]/i)
    plt.colorbar()
    plt.savefig("../plots/"+f+"_"+m+"_stack_sub_f"+flabels[ff]+".png")
    plt.close(fig)
    fig = plt.figure()
    plt.hist(sub_flat[ff]/i)
    plt.savefig("../plots/"+f+"_"+m+"_stack_sub_hist_f"+flabels[ff]+".png")
    plt.close(fig)
print("done")

print("Done stacks of residuals and subtractions")
                   
print("Trained CNN on file "+ f)
print("Num channels "+str(nchannels))
print("Model with SeparableConv2D")
print("Saved model as "+ofile)
print("Loss plot is "+pltfile)
print("Total time in hours: "+str((time.time()-start))*2.77778e-7)
