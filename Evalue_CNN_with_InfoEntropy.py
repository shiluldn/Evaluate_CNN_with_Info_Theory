# -*- coding: utf-8 -*-
"""
    This is the Pyhton code for evaluating the learning process of 
    convolutional neural networks (CNN) using information theoretical measures.
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
    By Xiyu Shi (x.shi@lboro.ac.uk), Varuna De-Silva (v.d.de-silva@lboro.ac.uk)
    
    30 November 2021

"""

#%%
#import glob
import time
#import matplotlib
from matplotlib import pyplot as plt
#import matplotlib.image as mpimg
import numpy as np
import pickle
import pandas as pd
#import imageio as im
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import SpatialDropout2D

from tensorflow.keras.models import load_model

#%% General setups for the testing environmental variables

# CNN kernel size. 3 means 3x3, 5 means 5x5 and 7 means 7x7
kernel_size = 3
cnn_number  = 2
dense_unit  = 512

dropout_rate = 0.0

# Hypo-parameters for model configuration
batch_size = 1024
epochs = 200

# Starting and ending image index for mutual information calculation
image_start = 1000
image_end = 2000

max_cnn = cnn_number

# path of saved epoch models, figures, etc.
path_epoch_models = "./Epoch_models/"

#%%
# Information Theoretic Quantities Calculation Functions
def entropy(Y):
    """
    Also known as Shanon Entropy
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count/len(Y)
    en = np.sum((-1)*prob*np.log2(prob))
    return en


#Joint Entropy
def jEntropy(Y,X):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    """
    YX = np.concatenate((Y,X), axis=0)
    return entropy(YX)

#Conditional Entropy
def cEntropy(Y, X):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    return jEntropy(Y, X) - entropy(X)


#Information Gain (or the Mutual Information)
def gain(Y, X):
    """
    Information Gain, I(Y;X) = H(Y) - H(Y|X)
    Reference: https://en.wikipedia.org/wiki/Information_gain_in_decision_trees#Formal_definition
    """
    return entropy(Y) - cEntropy(Y,X)

#%%
# Network configuration with multiple CNN layers, and followed by maxpooling,
# flatten and fully connected layers. Dropout layers can be added after different
# layers by uncommenting the relevant lines.

def configNetwork():
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation = 'relu'))
    
    # Add more hidden CNN layers to the network
    layer_count = cnn_number - 1
    while layer_count > 0:
        layer_count = layer_count - 1      
        classifier.add(Conv2D(32, (kernel_size, kernel_size), padding='valid', activation='relu'))
    
    # add a dropout layer after the Conv layer. If not needed, uncomment this line
    # classifier.add(SpatialDropout2D(dropout_rate))
    
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    
    # add a dropout layer after the MaxPooling. If not needed, uncomment this line
    classifier.add(Dropout(dropout_rate))
    
    # Flattening
    classifier.add(Flatten())
    
    # add a dropout layer after the Flattening. If not needed, uncomment this line
    # classifier.add(Dropout(dropout_rate))
    
    # Full connection 
    classifier.add(Dense(units = dense_unit, activation = 'relu'))
    
    # add a dropout layer after the Fully Connected layer Dense. If not needed, uncomment this line
    # classifier.add(Dropout(dropout_rate))
    
    # Output layer
    classifier.add(Dense(units = 10, activation = 'softmax'))  
    
    classifier.summary()
    classifier.compile(optimizer = 'rmsprop',
                        loss = 'categorical_crossentropy', 
                        metrics = ['accuracy'])
    return classifier

#%%
# Loading and examing the dataset. Either MNIST or Fashion-MNIST can be used by
# changing the code

def loading_dataset():
    num_classes = 10
    
    # Loading the data, split between train and test sets
    # To use Fashion_MNIST, change the "mnist" to "fashion_mnist" in the following line.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")    
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test  = keras.utils.to_categorical(y_test, num_classes)
    
    return x_train, y_train, x_test, y_test

#%%
# Plotting histogram of randomly selected images (Figures 3a and 3b)

def plotting_histogram():
    a=x_train[image_start + 150]
    b=x_train[image_start + 160]
    unique, count = np.unique(a.flatten(), return_counts=True, axis=0)
    x,y= np.unique(b.flatten(), return_counts=True, axis=0)
    #prob = count/len(a.flatten())
    
    fig2a=plt.figure(dpi=600)
    plt.ylim(0, 700)
    plt.bar(unique,count,width=0.01)
    plt.xlabel("Normalized Pixel Values")
    plt.ylabel("Count")
    plt.tight_layout();
    fig2a.savefig(path_figures + "Figure_3a.png", dpi=600)
    
    fig2b = plt.figure(dpi=600)
    plt.ylim(0, 700)
    plt.bar(x, y, width=0.01)
    plt.xlabel("Normalized Pixel Values")
    plt.ylabel("Count")
    plt.tight_layout();
    plt.show()
    fig2b.savefig(path_figures + "Figure_3b.png", dpi=600)
    
    
    # Printing Information Thoeretic Calculations of Images ( Table 5.5)
    
    print("\nEntropy of Image 1, H(X): ",entropy(a.flatten()))
    print("Entropy of Image 2, H(Y): ",entropy(b.flatten()))
    print("Mutual Information of Images,  I(X;Y): ",gain(a.flatten(),b.flatten()))
    print("Conditional Entropy of Images, H(X|Y) : ",cEntropy(a.flatten(),b.flatten()))
    print("Conditional Entropy of Images, H(Y|X) : ",cEntropy(b.flatten(),a.flatten()))
    print("Joint Entropy of Images, H(X,Y): ",jEntropy(a.flatten(),b.flatten()))

#%%
# Training and Saving Designed Model

def train_and_save_model():
    checkpoint = keras.callbacks.ModelCheckpoint(path_epoch_models + 'Model_{epoch}.h5', save_freq='epoch') 
    
    start_time = time.time()
    history=classifier.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size=batch_size, epochs=epochs,callbacks=[checkpoint])
    print('Training took {} seconds'.format(time.time()-start_time))
    
    # Save training history as a history dictionary file so we can reload late to draw the accuracy and loss curves
    with open(path_figures + 'histroyDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Reload the history as follows
    # history = pickle.load(open(path_figures + 'historyDict', "rb"))
    
    # Save training history as a CSV file
    # Convert the history.history dict to a pandas DataFrame:    
    hist_df = pd.DataFrame(history.history) 
    hist_df.to_csv(path_figures + 'history.csv')
    
    return history

#%%
# Plotting the accuracy and loss curves in model training and validation Curves

def plot_accuracy_loss():
    # plot loss
    fig = plt.figure(dpi=600)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(history.history['loss'], color='blue',label="Train")
    plt.plot(history.history['val_loss'], color='red', label='Test')
    plt.legend()
    plt.tight_layout();
    fig.savefig(path_figures + "loss.png", dpi=600)
    
    # plot accuracy
    fig = plt.figure(dpi=600)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='Train')
    plt.plot(history.history['val_accuracy'], color='red', label='Test')
    plt.legend()
    plt.tight_layout();
    fig.savefig(path_figures + "accuracy.png", dpi=600)
    plt.show()
    
#%%
# Calculating the Mutual Information Between Input and Output I(X;T), and 
# between Output and Labels I(Y;T) for each epoch model

def mutual_info_each_epoch():
#    from tensorflow.keras.models import load_model
    
    # Loading the saved weights of each epoch to the new model variable Model     
    all_epoch_model_list = []
    for i in range(1, epochs+1):
        epoch_model_name = 'Model_' + str(i) + '.h5'
        Model = load_model(path_epoch_models + epoch_model_name)    
        all_epoch_model_list.append(Model)
        
    # Calculating I(X;T) and I(Y;T) for each epoch model
    images=x_train[image_start:image_end]
    foutputs=[]
    start_time= time.time();
    for i, model in enumerate(all_epoch_model_list):
        layer_outputs=[layer.output for layer in model.layers[:]]
        activation_model=models.Model(inputs=model.input,outputs=layer_outputs)
        
        # foutputs[i][j] has the output of every layer j for each predicated image i.
        foutputs.append(activation_model.predict(images))   
        
    print('Testing {} images for every epoch took {} seconds'.format(image_end-image_start, time.time()-start_time))
    
    # I(Y;T) betweeen training lable Y and output T
    I_YT=[] 
    #I(X;T) between input X and output T
    I_XT=[] 
    
    for i in range(len(foutputs)):
        I_YT.append(gain(foutputs[i][total_layers-1].flatten(), y_train[image_start:image_end].flatten()))
        I_XT.append(gain(images.flatten(), foutputs[i][total_layers-1].flatten()))
        
    # Save mutual information to a CSV file
    mInfo_dict = {'I_XT' : I_XT, 'I_YT' : I_YT}
    mInfo_df = pd.DataFrame(mInfo_dict)
    mInfo_df.to_csv(path_figures + 'mutualInfo_epoch.csv')
        
    return I_XT, I_YT, all_epoch_model_list

#%%
# Plotting the mutual information in figures

def plot_all_mutul_info_figures():
    
    # Plotting mutual information I(X;T) between input X and the final trained output T 
    fig = plt.figure(dpi=600)
    #plt.ylim(5, 14);
    plt.plot(range(epochs), I_XT[0:epochs], color="blue", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("Mutual Information I(X;T)")
    plt.tight_layout();
    plt.show()
    fig.savefig(path_figures + "Figure_6a" + figSuffix + "_IXT.png", dpi=600)
    
    # Figure 4: Plot mutual information I(Y;T) between training label Y and final trained output T 
    # here we only plot the first 60 epochs' output - train label mutual info
    fig = plt.figure(dpi=600)
    #plt.ylim(3, 6.5);
    plt.plot(range(epochs), I_YT[0:epochs], color="blue", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("Mutual Information I(Y;T)")
    plt.tight_layout();
    plt.show()
    fig.savefig(path_figures + "Figure_6b" + figSuffix + "_IYT.png", dpi=600)
    
    # Plotting Mutual Information Along the Layers (Figure 7)
    images=x_train[image_start:image_end]
    
    epoch_list = [1, 2, 5, 10, 20, 30, 40, 50, 60, 100]
    epoch_output=[]
    start_time = time.time();
    for i, epoch_index in enumerate(epoch_list): # i:0-9
        epoch_model = all_epoch_model_list[epoch_index - 1]
        layer_outputs=[layer.output for layer in epoch_model.layers[:]]
        activation_model=models.Model(inputs=epoch_model.input, outputs=layer_outputs)
        epoch_output.append(activation_model.predict(images))
    print('Testing {} images for every epoch 1, 2, 5, 10, 20, 30, 40, 50, 60, 100 took {} seconds'.format(image_end-image_start, time.time()-start_time))
    
    epoch_entropy=[]
    start_time = time.time();
    for i, model in enumerate(epoch_output):# i:0-9
        layer_entropy=[]
        for j in range(len(epoch_output[0])): 
            layer_entropy.append(entropy(epoch_output[i][j].flatten()))
        
        epoch_entropy.append(layer_entropy)  
    print('Calculating mutual information between layers of an epoch took {} seconds'.format(time.time()-start_time))
    
    # Save the mutual information of each layer for 10 epochs to a CSV file
    layerInfo_df = pd.DataFrame(epoch_entropy)
    layerInfo_df.to_csv(path_figures + 'entropy_' + str(cnn_number) + 'layer_10Epochs.csv')
    
    x_coordinator = []
    for i in range(total_layers):
        x_coordinator.append(i+1) 
        
    fig = plt.figure(dpi=600)
    for i in range(0, 10):
        plt.plot(x_coordinator, epoch_entropy[i], marker='.', markersize=10)
    
    xticks = []    
    xtick_number = total_layers + 2
    for i in range(xtick_number):
        xticks.append(i+1)
    
    plt.xlabel("Layers")
    plt.ylabel("Mutual Information: I(X:T)")
    ax = plt.gca()
    ax.set_xticks(xticks)
    
    xtick_label_list = []
    for i in range(xtick_number):
        xtick_label_list.append("")
    for i in range(cnn_number):
        xtick_label_list[i] = "Conv" + str(i+1)
        print(i, xtick_label_list[i])
        
    xtick_label_list[i+1] = "Dropout"; xtick_label_list[i+2] = "MaxPool"; 
    xtick_label_list[i+3] = "Flatten"; xtick_label_list[i+4] = "Dense";
    xtick_label_list[i+5] = "Output"
    
    ax.set_xticklabels(xtick_label_list, rotation=90)
    plt.legend(epoch_list, title='Epoch', loc='lower right')
    plt.tight_layout();
    plt.show()
    fig.savefig(path_figures + "Figure_7" + figSuffix + "_layerMutualInfo.png")
        
#%%
# The main code for evaluating the CNN's learning process using Entropy theory

start_time_all = time.time()

for cnn_number in range(1, max_cnn+1):
    # With Dropout layer, We have 5 additional layers: MaxPool, dropout, Faltten, Dense1, Dense2 for Output
    # Without Dropout layer,  We have 4 additional layers: MaxPool, Faltten, Dense1, Dense2 for Output
    # Users need to change the following line to +4 or +5 as needed.
    total_layers = cnn_number + 5 
    path_figures = "./Figures/kernel_" + str(kernel_size) + "x" + str(kernel_size) + "/" + str(cnn_number) + "-layer/"
    figSuffix = chr(ord('a') + cnn_number - 1);
    
    classifier = configNetwork()
    
    x_train, y_train, x_test, y_test = loading_dataset()
       
    plotting_histogram()
    
    # Training and Saving Designed Model
    history = train_and_save_model()    
    plot_accuracy_loss()
    
    # Compute the mutual information I_XT and I_YT
    I_XT, I_YT, all_epoch_model_list = mutual_info_each_epoch()
  
    plot_all_mutul_info_figures()
    
print('\n\nTotal running time: {} seconds'.format(time.time() - start_time_all))
