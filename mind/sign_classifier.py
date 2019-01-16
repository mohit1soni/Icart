# This is the classifier program written in tensorflow and 
# visualized in tensorboard to classify traffic signs using german traffic sign dataset
#%%
import numpy
from utils import read_data as rd

#%%
path=["../data/traffic_signs/train.p","../data/traffic_signs/test.p"]
# While specifying the path make sure where you are runnig the code this path with respect to working in the 
dataset='GTSBR'
a=rd.read_data(path,dataset)
x_train,y_train,x_test,y_test=a.split_traffic_sign_data()
#%%


