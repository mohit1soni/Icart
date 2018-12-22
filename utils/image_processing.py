import numpy as np
import matplotlib.pyplot as plt
import cv2
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
        # plt.axis('off')
        if Cmap=='':
            plt.imshow(image)
        else:
            plt.imshow(image,cmap=Cmap)
        if group_view is True:
            plt.show()

    def single_img_read(self,img_path,show=False,window_name='Color_image'):
        self.image=cv2.imread(img_path)
        if show is True:
            # cv2.imshow(window_name,self.image)
            # cv2.waitKey(0)
            self.img_show(self.image,title=window_name,fig_num=1)
        return self.image

    def gray_img_read(self,show=False,window_name='gray_image'):
        self.gray_image=cv2.cvtColor(self.image,cv2.COLOR_RGB2GRAY)
        if show is True:
            # cv2.imshow(window_name,self.gray_image)
            # cv2.waitKey(0)
            self.img_show(self.gray_image,title=window_name,Cmap='gray',fig_num=2,group_view=False)

    def apply_gaussian_blur(self,Gauss_ksize=(5,5),show=False,window_name='gaussian_blured_image'):
        self.blured_image=cv2.GaussianBlur(self.gray_image,Gauss_ksize,0)
        if show is True:
            # cv2.imshow(window_name,self.gray_image)
            # cv2.waitKey(0)
            self.img_show(self.blured_image,title=window_name,Cmap='gray',fig_num=3,group_view=False)

    def canny_edge_detect(self,show=True,Threshold=[50,150],window_name='Canny_Image'):
        self.gray_img_read(show=True) # Making the image Gray
        self.apply_gaussian_blur(show=True)
        self.canny_image=cv2.Canny(self.blured_image,Threshold[0],Threshold[1])
        if show is True:
            # cv2.imshow(window_name,self.gray_image)
            # cv2.waitKey(0)
            self.img_show(self.canny_image,title=window_name,Cmap='gray',fig_num=4,group_view=False)

    def region_of_interest(self):
        self.image_height=self.canny_image.shape[0]
        polygon= np.array([(170,self.image_height),(913,self.image_height),(481,300)])
        self.mask=np.zeros_like(self.canny_image)
        cv2.fillPoly(self.mask,np.int32([polygon]),255)
        self.masked_image=cv2.bitwise_and(self.canny_image,self.mask)

    def hough_transform(self,show=True,window_name="Track_Image"):
        lines=cv2.HoughLinesP(self.masked_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
        self.line_image=np.zeros_like(self.image)
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2=line.reshape(4)
                cv2.line(self.line_image,(x1,y1),(x2,y2),(255,0,0),10)
        self.track_image=cv2.addWeighted(self.line_image,0.6,self.image,0.4,0.4)
        if show:
            self.img_show(self.track_image,title=window_name,Cmap='',fig_num=6,group_view=True)


    def find_lanes(self,show=True,window_name='Masked Image'):
        self.region_of_interest()
        if show:
            self.img_show(self.masked_image,title=window_name,Cmap='gray',fig_num=5,group_view=False)
        self.hough_transform()


def main():
    path='../data/test_images/solidWhiteCurve.jpg'
    pro=image_processing()
    image=pro.single_img_read(path,show=True)
    pro.canny_edge_detect()
    pro.find_lanes()

if __name__ == '__main__':
    main()