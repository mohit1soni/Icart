# This is the classifier program written in tensorflow and 
# visualized in tensorboard to classify traffic signs using german traffic sign dataset
#%%
import numpy as np
from utils import read_data as rd
import matplotlib.pyplot as plt
%matplotlib inline
#%%
path=["../data/traffic_signs/train.p","../data/traffic_signs/test.p"]
# While specifying the path make sure where you are runnig the code this path with respect to working in the 
dataset='GTSBR'
a=rd.read_data(path,dataset)
X_train,y_train,X_test,y_test=a.split_traffic_sign_data()
#%%
n_train, n_test = X_train.shape[0], X_test.shape[0]
# What's the shape of an traffic sign image?
image_shape = X_train[0].shape
# How many classes?
n_classes = np.unique(y_train).shape[0]
print("Number of training examples =", n_train)
print("Number of testing examples  =", n_test)
print("Image data shape  =", image_shape)
print("Number of classes =", n_classes)
#%%

#show a random sample from each class of the traffic sign dataset
rows, cols = 4, 12
fig, ax_array = plt.subplots(rows, cols)
plt.suptitle('RANDOM SAMPLES FROM TRAINING SET (one for each class)')
for class_idx, ax in enumerate(ax_array.ravel()):
    if class_idx < n_classes:
        # show a random image of the current class
        cur_X = X_train[y_train == class_idx]
        cur_img = cur_X[np.random.randint(len(cur_X))]
        ax.imshow(cur_img)
        ax.set_title('{:02d}'.format(class_idx))
    else:
        ax.axis('off')
#Hide both x and y ticks
plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
plt.draw()
