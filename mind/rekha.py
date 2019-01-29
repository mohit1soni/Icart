import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.ImageDraw as ImageDraw
import sys
from PIL import Image

sys.path.append("..")
from utils import image_vislaisation_util as visual
class Object_classifier(object):
    "This class is to colaborate all the trained model and make it as a single modules"
    def __init__(self,PATH_TO_MODEL):
        self.PATH_TO_MODEL= PATH_TO_MODEL
        self.detection_graph=tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def=tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_MODEL,'rb') as fid:
                serialized_graph=fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def,name='')
            self.image_tensor=self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes=self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores=self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes=self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d=self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess=tf.Session(graph=self.detection_graph)

    def get_classification(self,img):
        with self.detection_graph.as_default():
            img_expanded=np.expand_dims(img,axis=0)
            (boxes,scores,classes,num) = self.sess.run([self.d_boxes,self.d_scores,self.d_classes,self.num_d],
            feed_dict={self.image_tensor:img_expanded})
        return boxes,scores,classes,num

    def read_from_webcam(self,window_name="objet_detection"):
        "This class is to read frames from camera and continuously passing it to the classifier and bounding box detector "
        cap=cv2.VideoCapture(0)
        while True:
            ret,image_np = cap.read()
            image_np=self.make_bounding_box(image_np)
            cv2.imshow(window_name,cv2.resize(image_np,(800,600)))
            if cv2.waitKey(25) & 0XFF == ord('q'):
                cv2.destroyAllWindows()
                break

    def make_bounding_box(self,img,color='red'):
        """ This method makes the bounding boxes using the trained inference graphs"""
        width=img.shape[0]
        height=img.shape[1]
        self.boxes,self.scores,self.classes,self.num=self.get_classification(img)
        pil_image=Image.fromarray(img)
        index= np.where(self.scores[0] >= 0.5)[0]
        for i in index:
            score_of_Class=self.scores[0,i]
            class_id=self.classes[0,i]
            bbox=self.boxes[0,i]
            top = bbox[0]*height
            left = bbox[1]*width
            bottom = bbox[2]*height
            right = bbox[3]*width
            draw = ImageDraw.Draw(pil_image)
            draw.line([(left, top), (left, bottom), (right, bottom),
                 (right, top), (left, top)], width=4, fill=color)
        pil_image=np.array(pil_image)
        return pil_image

def main():
    PATH_TO_MODEL='../Trained_model/frozen_inference_graph_coco.pb'
    traffic_signs=Object_classifier(PATH_TO_MODEL)
    traffic_signs.read_from_webcam()

    # img_path="../../Data/test_images/image1.jpg"
    # img=Image.open(img_path)

    # print(boxes, "Boxes")
    # img.save("new_image.jpg")
    # print(scores)
    # print(classes)
    # print(num)

if __name__ == "__main__":
    main()