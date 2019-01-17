# This is the classifier program written in tensorflow and 
# visualized in tensorboard to classify traffic signs using german traffic sign dataset
#%%
import numpy as np
from utils import read_data as rd
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
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

#%%
train_distribution,test_distribution = np.zeros(n_classes),np.zeros(n_classes)
for c in range(n_classes):
    train_distribution[c] = np.sum(y_train == c)/n_train
    test_distribution[c] = np.sum(y_test == c) /n_test
fig,ax = plt.subplots()
col_width = 0.5
bar_train =ax.bar(np.arange(n_classes),train_distribution,width=col_width,color= 'r') 
bar_test =ax.bar(np.arange(n_classes)+col_width,test_distribution,width=col_width,color= 'b')
ax.set_ylabel('PERCENTAGE OF Presence')
ax.set_xlabel('Class Label')
ax.set_title('Classes distribution in traffic-sign dataset')
ax.set_xticks(np.arange(0,n_classes,5) + col_width)
ax.set_xticklabels(['{:02d}'.format(c) for c in range(0,n_classes,5)])
ax.legend((bar_train[0],bar_test[0]),('train_set','test_set'))
plt.show()

#%%
def preprocess_freatures(X,equalize_hist=True):
    X = np.array([np.expand_dims(cv2.cvtColor(rgb_img,
    cv2.COLOR_RGB2YUV)[:,:,0],2) for rgb_img in X ])

    if equalize_hist:
        X=np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img)),2) 
        for img in X])
    X = np.float32(X)
    X -= np.mean(X,axis=0)
    X /= (np.std(X,axis=0) + np.finfo('float32').eps)
    return X
#%%
X_train_norm = preprocess_freatures(X_train)
X_test_norm = preprocess_freatures(X_test)
#%%
