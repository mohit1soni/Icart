import numpy as np
from sklearn.model_selection import train_test_split
import glob
import cv2
import os
import time
from datetime import datetime

batch_save_path="../../../data/data_drive/"+"/batches/"
label_save_path="../../../data/data_drive/"+"/labels/"

batch_list = glob.glob(batch_save_path+"*.npy")
label_list = glob.glob(label_save_path+"*.npy")

x_train,x_val,y_train,y_val=train_test_split(batch_list,label_list,test_size=0.2,random_state=42,shuffle=False,stratify=None)

# For showing an chosen image form the batch
# data=np.load(x_train[0])
# label=np.load(y_train[0])
# print(label[99])
# while(True):
    # image=data[99,:,:,:].astype("uint8")
    # cv2.imshow("image",image)
    # if cv2.waitKey(25) & 0XFF==ord('f'):
        # break

# Directory defination
root_dir=os.getcwd()
restore_from=os.path.join(root_dir,'saved_model','model_name')
log_dir=os.path.join(root_dir,'log')

# Variable Initialization
height=256
width=455
batch_size=100
nb_epoch=10
saved_model_step=10
max_to_keep=4

# Defining Time_based folder creation for logs
now=datetime.now()
time_stamp=now.strftime("%Y_%m_%d_%H_%M_%S")
log_dir=os.path.join(log_dir,time_stamp)
# os.makedirs(log_dir)
saved_model_dir=os.path.join(root_dir,'saved_model',time_stamp)
# os.makedirs(saved_model_dir)
print(saved_model_dir)


import tensorflow as tf

steering=tf.graph()
print("Genration of Tensorflow Graph")

with steering.as_default():
    global_step=tf.variable(0)
    with tf.name_scope('Inputs'):
        input_data=tf.placeholder(tf.float32,shape=(None,height,width,3),name="images_in")
        input_labels=tf.placeholder(tf.float32,shape=(None),name="steeting_angle")
        is_training=tf.placeholder_with_default(False,shape=())
        condition=tf.cast(is_training,tf.float32)
        pred=tf.less(condition,0.5)
        one_hot=tf.one_hot(input_labels,)
    with tf.name_scope("Driver_Net"):







