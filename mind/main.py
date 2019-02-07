import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.ImageDraw as ImageDraw
import sys
from PIL import Image
import id_class_map as idmap

sys.path.append("..")
from utils import image_vislaisation_util as visual
class Object_classifier(object):
    "This class is to colaborate all the trained model and make it as a single modules"
    def __init__(self,PATH_TO_MODEL,model_name="coco"):
        self.PATH_TO_MODEL= PATH_TO_MODEL
        self.detection_graph=tf.Graph()
        self.id_map=idmap.id_map(model_name)
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
        # cap=cv2.VideoCapture('../../data/video1.mp4')
        cap=cv2.VideoCapture(0)
        # cap.set(cv2.CV_CAP_PROP_FPS, 15)
        # print("hello")
        # cap.set(3,800)
        # cap.set(4,600)
        while True:
            ret,image_np = cap.read()
            image_np=self.make_bounding_box(image_np)
            cv2.imshow(window_name,cv2.resize(image_np,(1280,720)))
            # cv2.imshow(window_name,image_np)
            # cv2.waitKey(50)
            if cv2.waitKey(25) & 0XFF == ord('q'):
                cv2.destroyAllWindows()
                break
    def make_bounding_box(self,img,color='blue'):
        """ This method makes the bounding boxes using the trained inference graphs"""
        # width=1200
        # height=786
        self.boxes,self.scores,self.classes,self.num=self.get_classification(img)
        pil_image=Image.fromarray(img)
        draw = ImageDraw.Draw(pil_image)
        width,height=pil_image.size

        index= np.where(self.scores[0] >= 0.5)[0]
        class_ids=list()
        class_ids.append([int(self.classes[0,j]) for j in index])
        # print(class_ids[0])
        class_names=self.id_map.map(class_ids[0])
        print(class_names)
        for i in index:
            score_of_Class=self.scores[0,i]
            print(score_of_Class)
            class_id=self.classes[0,i]
            bbox=self.boxes[0,i]
            top = bbox[0]*height
            left = bbox[1]*width
            bottom = bbox[2]*height
            right = bbox[3]*width
            draw.line([(left, top), (left, bottom), (right, bottom),
                 (right, top), (left, top)], width=4, fill=color)
        np.copyto(img,np.array(pil_image))
        return img
    def tracking_dyanmic(self):
        pass

def main():
    PATH_TO_MODEL='../Trained_model/frozen_inference_graph_coco.pb'
    # print("hello")
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