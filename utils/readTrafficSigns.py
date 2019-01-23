# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#

#%%
import matplotlib.pyplot as plt
import csv
import numpy as np
import os
import shutil
# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
#%%
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    # images = [] # images
    # labels = [] # corresponding labels
    # shapes=[]
    # count_images=list()
    dest = rootpath
    dest = dest.replace("Images","nimage")
    csv.register_dialect('md',delimiter=',',quoting=csv.QUOTE_NONE,skipinitialspace=True)
    # loop over all 42 classes
    i=0
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        # gtFile = open('test.csv')
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader)
        # skip header
        # loop over all images in current annotations file
        # files=os.listdir(prefix)
        # for filename in files:
        #     dst=filename.replace("_0_0.ppm","_"+str(c)+".ppm")
        #     src=prefix+filename
        #     dst=prefix+dst
        #     if ".ppm" in filename:
        #         os.rename(src,dst)

        # with open('train_labels.csv','a',newline='') as f:
        #     writer=csv.writer(f,dialect='md')
        for row in gtReader:
            shutil.copy(prefix+row[0].replace(".ppm","_"+str(c)+".jpg"),dest)
            # images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            # labels.append(row[7]) # the 8th column is the label
            # shapes.append(row[1])
            # shapes.append(row[2])
            # shapes.append(row[3])
            # shapes.append(row[4])
            # shapes.append(row[5])
            # shapes.append(row[6])
            # count_images.append(len(images[i]))
            # i +=1
            # row[0]=row[0].replace(".ppm","_"+str(c)+".ppm")
            # writer.writerow(row)
        gtFile.close()

    # shapes=np.reshape(shapes,(len(labels),6))
    # np.asarray(count_images)
    # return images,labels,shapes,count_images

#%%
path="../../Data/Images"
# images,labels,shapes,count_images = readTrafficSigns(path)
readTrafficSigns(path)
#%%
# print(len(images[0][1]))
#%%
# np.save('../data/traffic_signs/images_train.npy',images
# np.save('../data/traffic_signs/labels_train.npy',labels)
