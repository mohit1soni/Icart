import numpy as np
import cv2
import matplotlib.pyplot as plt
# import matplotlib.image as imgplt
class image_processing(object):
    """ This class is for all the image processing tasks
    required to build the Icart.
    Input:
    Functions:
    outputs:
    """
    def __init__(self):
        pass
    def img_show(self,image,title='image',Cmap='',fig_num=1,group_view=False):
        plt.figure(fig_num)
        plt.title(title)
        plt.axis('off')
        if Cmap=='':
            plt.imshow(image)
        else:
            plt.imshow(image,cmap=Cmap)
        if group_view== True:
            plt.show()

    def single_img_read(self,img_path,show=False,window_name='Color_image'):
        self.image=cv2.imread(img_path)
        if show == True:
            # cv2.imshow(window_name,self.image)
            # cv2.waitKey(0)
            self.img_show(self.image,title=window_name,fig_num=1)
        return self.image

    def gray_img_read(self,show=False,window_name='gray_image'):
        self.gray_image=cv2.cvtColor(self.image,cv2.COLOR_RGB2GRAY)
        if show==True:
            # cv2.imshow(window_name,self.gray_image)
            # cv2.waitKey(0)
            self.img_show(self.gray_image,title=window_name,Cmap='gray',fig_num=2,group_view=False)

    def apply_gaussian_blur(self,Gauss_ksize=(5,5),show=False,window_name='gaussian_blured_image'):
        self.blured_image=cv2.GaussianBlur(self.gray_image,Gauss_ksize,0)
        if show==True:
            # cv2.imshow(window_name,self.gray_image)
            # cv2.waitKey(0)
            self.img_show(self.blured_image,title=window_name,Cmap='gray',fig_num=3,group_view=False)

    def canny_edge_detect(self,show=True,Threshold=[50,150],window_name='Canny_Image'):
        self.gray_img_read(show=True) # Making the image Gray
        self.apply_gaussian_blur(show=True)
        self.canny_image=cv2.Canny(self.blured_image,Threshold[0],Threshold[1])
        if show==True:
            # cv2.imshow(window_name,self.gray_image)
            # cv2.waitKey(0)
            self.img_show(self.canny_image,title=window_name,Cmap='gray',fig_num=4,group_view=True)

    def find_lanes(self):
        pass


def main():
    path='../data/test_images/solidWhiteCurve.jpg'
    pro=image_processing()
    image=pro.single_img_read(path,show=True)
    pro.canny_edge_detect()

if __name__ == '__main__':
    main()