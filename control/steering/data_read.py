import glob
import numpy as np
import cv2
import csv

data_path="../../../data/driving_dataset"
batch_size = 100
# labels_file=open("../../../data/driving_dataset/data.txt",'r')

# labels =list()
# for row in labels_file:
#     k=row.split(" ")
#     k[0]=data_path+"/"+k[0]
#     labels.append(k[0])
#     labels.append(float(k[1].replace("/n","")))

# labels=np.reshape(labels,(int(len(labels)/2),2))
# img_samp=cv2.imread(labels[0,0])
# (img_height,img_width,n_channels)=img_samp.shape

# n_batches=int(labels.shape[0]/batch_size)

batch_save_path="../../../data/data_drive/"+"/batches/"
label_save_path="../../../data/data_drive/"+"/labels/"

#  For the creation of batches

# for i in range(n_batches):
#     batch=np.zeros((batch_size,img_height,img_width,n_channels))
#     steering=np.zeros((batch_size),dtype=float)
#     steering=labels[i*batch_size:(i+1)*batch_size,1]
#     filename=labels[i*batch_size:(i+1)*batch_size,0]
#     for j in range(batch_size):
#         batch[j,:,:,:]=cv2.imread(filename[j])
#     np.save(batch_save_path+"batch_"+str(i)+".npy",batch)
#     np.save(label_save_path+"label_"+str(i)+".npy",steering)
#     np.save(label_save_path+"filename_"+str(i)+".npy",filename)
#     print(i,end=" ")

#  For the testing of the generated batches and labels
batch1=np.load(batch_save_path+"batch_0.npy")
label1=np.load(label_save_path+"label_0.npy")
print(label1[99])

# For showing an chosen image form the batch

# while(True):
#     image=batch1[99,:,:,:].astype("uint8")
#     cv2.imshow("image",image)
#     if cv2.waitKey(25) & 0XFF==ord('f'):
#         break


