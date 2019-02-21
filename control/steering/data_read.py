import glob
import numpy as np
import cv2
import csv
from decimal import *
import matplotlib.pyplot as plt
import pandas as pd

data_path="../../../data/driving_dataset"
batch_size = 100
labels_file=pd.read_csv("../../../data/driving_dataset/data.txt",delimiter=' ')

def assign_bin(decimal_value,min_steer,max_steer):
    int_range=decimal_value
    if int(np.divide(int_range,10)) == 0:
        return np.round(int_range)
    else:
        new_int_range=np.divide(int_range,10)
        # print(new_int_range)
        return np.round(new_int_range)

labels=np.asarray(labels_file['steer'],dtype=np.float32)
min_steer=min(labels)
max_steer=max(labels)
print(type(labels[0]))

new_label=list()
for i in range(labels.shape[0]):
    label=assign_bin(labels[i],min_steer,max_steer)
    new_label.append(label)


# labels =list()
# for row in labels_file:
#     k=row.split(" ")
#     k[0]=data_path+"/"+k[0]
#     labels.append(k[0])
#     int_range = assign_bin(Decimal(k[1]),min_steer,max_steer)
#     labels.append(int_range)


# labels=np.reshape(labels,(int(len(labels)/2),2))

# img_samp=cv2.imread(labels[0,0])
# (img_height,img_width,n_channels)=img_samp.shape


n_batches=int(labels.shape[0]/batch_size)

batch_save_path="../../../data/data_drive/"+"/batches/"
label_save_path="../../../data/data_drive/"+"/labels/"

# #  For the creation of batches
max_steer=max(new_label)
min_steer=min(new_label)

min_index=np.argmin(new_label)
max_index=np.argmax(new_label)

# # Just For Checking Purposes
print(min_steer,max_steer)
print(min_index,max_index)
# print(labels[min_index,1],labels[max_index,1])
# print(type(labels[100,1]))
##  For the visualization of the data

# shorted_list=sorted(new_label,key=int)
# plt.hist(shorted_list,bins=30,histtype="step")
# plt.xlabel("Steering Angle")
# plt.ylabel("No of Occurance")
# plt.title("Data Visualization with Non-linear Binning Function")
# plt.savefig("../../Results/Bining_histogram_steering_Ang.png")
# plt.show()

##  For the generation of data and labels

for i in range(n_batches):
    # batch=np.zeros((batch_size,img_height,img_width,n_channels))
    steering=np.zeros((batch_size),dtype=float)
    steering=new_label[i*batch_size:(i+1)*batch_size]
    # filename=labels[i*batch_size:(i+1)*batch_size,0]
    # for j in range(batch_size):
    #     batch[j,:,:,:]=cv2.imread(filename[j])
    # np.save(batch_save_path+"batch_"+str(i)+".npy",batch)
    print(i,end=" ")
    np.save(label_save_path+"label_"+str(i)+".npy",steering)
    # np.save(label_save_path+"filename_"+str(i)+".npy",filename)


##  For the testing of the generated batches and labels

batch1=np.load(batch_save_path+"batch_0.npy")
label1=np.load(label_save_path+"label_15.npy")
print(label1[10])



# For showing an chosen image form the batch

while(True):
    image=batch1[99,:,:,:].astype("uint8")
    cv2.imshow("image",image)
    if cv2.waitKey(25) & 0XFF==ord('f'):
        break

