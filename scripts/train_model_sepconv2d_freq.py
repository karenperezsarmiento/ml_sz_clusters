import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras import datasets as tfdatasets
import datasets
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import Normalization
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
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import argparse as ap
import re

parser = ap.ArgumentParser(description="Train CNN")
parser.add_argument("-if","--input_file",type=str,help="Name of input dataset file")
parser.add_argument("-nc","--num_channels",type=int,help="Number of channels in input cutout     (1 or 3)")
parser.add_argument("-fq","--frequency",type=int,help="Frequency of CMB+Compton-y map (93,145,217)")
parser.add_argument("-af","--activation_func",type=str,help="Activation function in CNN (relu,selu,linear,etc)")
args = parser.parse_args()
input_fn = args.input_file
nchannels = args.num_channels
freq = args.frequency
act_func = args.activation_func
infile = "../data/"+input_fn

data = load_dataset("json",data_files=infile,split="train")

freq_key = "tot_"+str(freq)
train_testvalid = data.train_test_split(test_size=0.1)
df_train = train_testvalid["train"].to_tf_dataset(
    columns=[freq_key],
    label_cols=["tsz_8192"],
    batch_size=2,
    prefetch=False,
    shuffle=True)
df_test = train_testvalid["test"].to_tf_dataset(
    columns=[freq_key],
    label_cols=["tsz_8192"],
    batch_size=2,
    prefetch=False,
    shuffle=False)

N_CLASSES = 1
window_function = 3
droupout_rate = 0.00
def ResUNet(img_shape):

    inputs = Input(shape = img_shape)
    #tabs_input = Input(shape = img_shape)
    print (inputs)

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
opt = Adam(learning_rate = 1e-5)
model.compile(optimizer=opt, loss='mean_squared_error',metrics=["mse"])

# Train the model on all available devices.
history = model.fit(df_train, validation_data=df_test, epochs=300)
ofile = re.sub(".jsonl",".keras",input_fn)
pltfile = re.sub(".jsonl",".png",input_fn)
ofile = "../models/sep_conv2d_"+act_func+"_"+str(freq)+"_"+ofile
pltfile = "../plots/sep_conv2d_"+act_func+"_"+str(freq)+"_"+pltfile
model.save(ofile)

i = 0
for inputs,labels in df_test.map(lambda x,y: (x,y)):
    ifile = "../test_imgs/"+str(i)+"_input_sep_conv2d_"+act_func+"_"+str(freq)+".png"
    outfile = "../test_imgs/"+str(i)+"_label_sep_conv2d_"+act_func+"_"+str(freq)+".png"
    fig = plt.figure()
    larr = labels.numpy()
    plt.imshow(larr.reshape(64,64))
    plt.savefig(outfile)
    plt.close(fig)
    iarr = inputs.numpy() 
    iarr = iarr[0,:,:,0]
    fig = plt.figure()
    plt.imshow(iarr.reshape(64,64))
    plt.savefig(ifile)  
    plt.close(fig)
    i += 1

predictions = model.predict(df_test)

for i in range(predictions.shape[0]):
    outfile = "../test_imgs/"+str(i)+"_predict_sep_conv2d_"+act_func+"_"+str(freq)+".png"
    fig = plt.figure()
    plt.imshow(predictions[i].reshape(64,64)) 
    plt.savefig(outfile)
    plt.close(fig)
                    
history= model.history
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
print("Trained CNN on file "+infile)
print("Num channels "+str(nchannels))
print("Model with SeparableConv2D")
print("Saved model as "+ofile)
print("Loss plot is "+pltfile)

