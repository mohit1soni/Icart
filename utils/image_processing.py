import numpy as np
import cv2
import matplotlib.pyplot as plt

class image_processing(object):
    """ This class is for all the image processing tasks
    required to build the Icart.
    Input:
    Functions:
    outputs:
    """
    def __init__(self):
        pass
    def single_img_read(self,img_path,show=True,window_name='Resulting_image'):
        self.image=cv2.imread(img_path)
        if show == True:
            cv2.imshow(window_name,self.image)
            cv2.waitKey(0)
        return self.image

    def gray_img_read(self,show=True,window_name='gray_image'):
        self.gray_image=cv2.cvtColor(self.image,cv2.COLOR_RGB2GRAY)
        if show==True:
            cv2.imshow(window_name,self.gray_image)
            cv2.waitKey(0)

    def apply_gaussian_blur(self):
        pass

    def canny_edge_detect(self):
        self.gray_img_read() # Making the image Gray



    def find_lanes(self):
        pass


def main():
    path='../data/test_images/solidWhiteCurve.jpg'
    pro=image_processing()
    image=pro.single_img_read(path)

if __name__ == '__main__':
    main()