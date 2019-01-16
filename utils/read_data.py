#%%
import pickle
import numpy
#%%
class read_data(object):
    def __init__(self,path,dataset,val=False):
        if val:
            self.train_path=path[0]
            self.val_path=path[1]
            self.test_path=path[2]
        else:
            self.train_path=path[0]
            self.test_path=path[1]
        self.dataset_name=dataset

    def read_traffic_sign_data(self):
        with open(self.train_path,mode='rb') as f:
            train=pickle.load(f)
        with open(self.test_path,mode='rb') as f:
            test=pickle.load(f)
        return train,test

    def split_traffic_sign_data(self):
        train,test=self.read_traffic_sign_data()
        x_train,y_train=train['features'],train['labels']
        x_test,y_test=test['features'],test['labels']
        return x_train,y_train,x_test,y_test

