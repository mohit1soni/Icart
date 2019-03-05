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

x_train,x_val,y_train,y_val=train_test_split(batch_list,label_list,test_size=0.2,random_state=10,shuffle=False,stratify=None)

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
restore_from=os.path.join(root_dir,'saved_model','2019_02_22_04_00_50')
log_dir=os.path.join(root_dir,'log')

## Variable Initialization

# Image Vriables
height=256
width=455
batch_size=100
num_classes=16+1+50

# Network Variables
epochs=100
max_to_keep=4
regularizer_scale=0.2
display_step=1
batch_display_step=50
save_model_step=10
num_train_batch=len(x_train)
num_val_batch=len(x_val)
learning_rate=0.0001


# Defining Time_based folder creation for logs
now=datetime.now()
time_stamp=now.strftime("%Y_%m_%d_%H_%M_%S")
log_dir=os.path.join(log_dir,time_stamp)
os.makedirs(log_dir)
saved_model_dir=os.path.join(root_dir,'saved_model')
saved_model_dir=os.path.join(saved_model_dir,time_stamp)
os.makedirs(saved_model_dir)

print(saved_model_dir)


import tensorflow as tf
slim = tf.contrib.slim

steering=tf.Graph()
print("Genration of Tensorflow Graph")

with steering.as_default():
    global_step=tf.Variable(0)
    with tf.name_scope('Inputs'):
        input_data=tf.placeholder(tf.float32,shape=(None,height,width,3),name="images_in")
        input_labels=tf.placeholder(tf.int32,shape=(None),name="steeting_angle")
        is_training=tf.placeholder_with_default(False,shape=())
        condition=tf.cast(is_training,tf.float32)
        pred=tf.less(condition,0.5)
        one_hot=tf.one_hot(input_labels,num_classes,dtype=tf.int32,name="One_hot_labels")
        input_tensor=tf.subtract(tf.multiply(2.0,tf.divide(input_data,255.0)),1.0)

    with tf.name_scope("Driver_Net"):
        regularizer=tf.contrib.layers.l2_regularizer(scale=regularizer_scale)
        net=tf.layers.conv2d(input_tensor,3,5,strides=(2,2),padding='valid',activation=tf.nn.elu,name="layer0",kernel_regularizer=regularizer)
        net=tf.layers.dropout(net,rate=0.2)

        net_mean,net_variances=tf.nn.moments(net,[0])
        net = tf.nn.batch_normalization(net,mean=net_mean,variance=net_variances,offset=None,scale=None,variance_epsilon=0.001)

        net=tf.layers.conv2d(net,24,5,strides=(2,2),padding='valid',activation=tf.nn.elu,name="layer1",kernel_regularizer=regularizer)
        net=tf.layers.dropout(net,rate=0.2)

        net_mean,net_variances=tf.nn.moments(net,[0])
        net = tf.nn.batch_normalization(net,mean=net_mean,variance=net_variances,offset=None,scale=None,variance_epsilon=0.001)

        net=tf.layers.conv2d(net,36,5,strides=(2,2),padding='valid',activation=tf.nn.elu,name="layer2",kernel_regularizer=regularizer)
        net=tf.layers.dropout(net,rate=0.2)

        net_mean,net_variances=tf.nn.moments(net,[0])
        net = tf.nn.batch_normalization(net,mean=net_mean,variance=net_variances,offset=None,scale=None,variance_epsilon=0.001)

        net=tf.layers.conv2d(net,48,5,strides=(2,2),padding='valid',activation=tf.nn.elu,name="layer3",kernel_regularizer=regularizer)
        net=tf.layers.dropout(net,rate=0.2)

        net=tf.layers.conv2d(net,64,3,strides=(1,1),padding='valid',activation=tf.nn.elu,name="layer4")
        net=tf.layers.dropout(net,rate=0.2)

        net=tf.layers.conv2d(net,64,3,strides=(1,1),padding='valid',activation=tf.nn.elu,name="layer5")
        net=tf.layers.dropout(net,rate=0.2)

        net=tf.contrib.layers.flatten(net)

        net=tf.layers.dense(net,100,activation=tf.nn.elu,name='Dense_1')
        net=tf.layers.dropout(net,rate=0.5,training=is_training)

        # net=tf.layers.dense(net,50,activation=tf.nn.elu,name='Dense_2')
        # net=tf.layers.dropout(net,rate=0.5,training=is_training)

        # net=tf.layers.dense(net,10,activation=tf.nn.elu,name='Dense_3')
        # net=tf.layers.dropout(net,rate=0.5,training=is_training)

        output=tf.layers.dense(net,num_classes,name="output_layer")

    with tf.name_scope("Training"):
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=one_hot),name="loss")
        loss=tf.reduce_mean(tf.squared_difference(output,input_labels),name="loss")
        
        optimize=tf.train.AdadeltaOptimizer(learning_rate=learning_rate,name="optimizer").minimize(loss,global_step=global_step)

        #Summery Log
        training_loss_summery=tf.summary.scalar("Training_Loss",loss)
        val_loss_summary=tf.summary.scalar("Val_loss",loss)

    with tf.name_scope("Accuracy"):
        pred_prob=tf.nn.softmax(output)
        pred_labels=tf.argmax(pred_prob,1,name="predicted_labels")
        train_acc=tf.metrics.accuracy(input_labels,pred_labels,name="Accuracy")[1]
        val_acc=tf.metrics.accuracy(input_labels,pred_labels,name="Accuracy")[1]

        # Summery and Logs
        train_acc_summary=tf.summary.scalar('Training_arruracy',train_acc*100)
        val_acc_summary=tf.summary.scalar('Validation_arruracy',val_acc*100)

        train_summary_merged=tf.summary.merge([train_acc_summary,training_loss_summery])
        val_summary_merged=tf.summary.merge([val_acc_summary,val_loss_summary])

    saver=tf.train.Saver(max_to_keep=max_to_keep)

tf.reset_default_graph()

print("Started Running Session .. ")
with tf.Session(graph=steering) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    #  For Restoring The previous model
    if tf.train.latest_checkpoint(restore_from) is not None:
        print("Restoring Model")
        saver.restore(sess,tf.train.latest_checkpoint(restore_from))
    # Defining Regular writer discriptions
    writer=tf.summary.FileWriter(log_dir)
    writer.add_graph(sess.graph)
    writer.flush()

    for epoch in range(epochs):
        epoch_start_time=time.time()
        # Training Start here
        for batch_index in range(num_train_batch):
            batch_start_time=time.time()
            data=np.load(x_train[batch_index])
            label=np.load(y_train[batch_index])
            feed_dict={input_data:data,input_labels:label,is_training:True}
            sess.run(optimize,feed_dict=feed_dict)
            # For the Display Purpose
            if batch_index%(batch_display_step) == 0 and epoch%(display_step)==0:
                print('Epoch: %d <===>'%(epoch),
                      '%.2f Percent Completed'%((batch_index*100.0/num_train_batch)),
                      '[Batch Size--> %d Images <===> Batch Process Time--> %.2f Seconds]'%(data.shape[0],time.time()-batch_start_time))

        if epoch%display_step == 0:
            # Calculating Validation Accuracy
            random_index=np.random.randint(0,num_val_batch)
            sample_val_data=np.load(x_val[random_index])
            sample_val_label=np.load(y_val[random_index])
            feed_dict={input_data:sample_val_data,input_labels:sample_val_label,is_training:False}
            val_accuracy,val_summary=sess.run([val_acc,val_summary_merged],feed_dict)

            # Calculating Training Accuracy
            random_index=np.random.randint(0,num_train_batch)
            sample_train_data=np.load(x_train[random_index])
            sample_train_label=np.load(y_train[random_index])
            feed_dict={input_data:sample_train_data,input_labels:sample_train_label,is_training:False}
            loss_op,train_accuracy,train_summary=sess.run([loss,train_acc,train_summary_merged],feed_dict)

            # Printing the Output
            print('Epoch: %d <===>'%(epoch),'(val accuracy-->%.2f) (training loss-->%.2f) (training accuracy-->%.2f)'%(val_accuracy,loss_op,train_accuracy))
            print('...Writing Logs in %s...'%(time_stamp))
            writer.add_summary(train_summary,epoch)
            writer.flush()
            writer.add_summary(val_summary,epoch)
            writer.flush()
            #writer.add_summary(temp_op,epoch)
            print('[Epoch Process Time--> %.2f Seconds]'%(time.time()-epoch_start_time))
                #--------------------------------------------------------------
        if epoch%save_model_step==0:
            print('...Saving Checkpoint File in %s...'%(time_stamp))
            saver.save(sess, saved_model_dir,global_step=epoch)
    tf.train.write_graph(sess.graph,saved_model_dir,'model.pbtxt')



