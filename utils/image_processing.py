import numpy as np
import matplotlib.pyplot as plt
import os
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
        self.left_fit=list()
        self.right_fit=list()

    def img_show(self,image,title='image',Cmap='',fig_num=1,group_view=False,save=False):
        plt.figure(fig_num)
        plt.title(title)
        # plt.axis('off')
        if Cmap=='':
            plt.imshow(image)
        else:
            plt.imshow(image,cmap=Cmap)
        if save:
            plt.savefig("../Results/image_processing/Track/"+title+".png")
        if group_view is True:
            plt.show()

    def video_capture(self,video_path,show=False,window_name="video_frame",save=False):
        cap=cv2.VideoCapture(video_path)
        if save:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out=cv2.VideoWriter('../Results/image_processing/Track/'+window_name+'.mp4',fourcc,32,(960,540),1)
        while(cap.isOpened()):
            ret,frame=cap.read()
            if ret==True:
                self.image=frame
                self.canny_edge_detect()
                self.find_lanes(show=False)
                if show:
                    cv2.imshow(window_name,self.track_image)
                    if cv2.waitKey(1) == ord('q'):
                        break
                if save:
                    out.write(self.track_image)
            else:
                break
        if save:
            out.release()
        cv2.destroyAllWindows()

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

    def canny_edge_detect(self,show=False,Threshold=[50,150],window_name='Canny_Image'):
        self.gray_img_read(show=False) # Making the image Gray
        self.apply_gaussian_blur(show=False)
        self.canny_image=cv2.Canny(self.blured_image,Threshold[0],Threshold[1])
        if show is True:
#             cv2.imshow(window_name,self.gray_image)
#             cv2.waitKey(0)
           self.img_show(self.canny_image,title=window_name,Cmap='gray',fig_num=4,group_view=False)

    def region_of_interest(self):
        self.image_height=self.canny_image.shape[0]
        polygon= np.array([(200,self.image_height),(1100,self.image_height),(550,250)])
        #This polygon drawing has to be adaptive so that it will always detect lines
        self.mask=np.zeros_like(self.canny_image)
        cv2.fillPoly(self.mask,np.int32([polygon]),255)
        self.masked_image=cv2.bitwise_and(self.canny_image,self.mask)

    def hough_transform(self,show=False,window_name="Track_Image"):
        lines=cv2.HoughLinesP(self.masked_image,2,np.pi/180,10,np.array([]),minLineLength=40,maxLineGap=5)
        self.line_image=np.zeros_like(self.image)
        left_fit=list()
        right_fit=list()

        if lines is not None:
            for line in lines:
                x1,y1,x2,y2=line.reshape(4)
                parameters=np.polyfit((x1,x2),(y1,y2),1)
                slope=parameters[0]
                intercept=parameters[1]
                if slope < 0 :
                    left_fit.append((slope,intercept))
                else:
                    right_fit.append((slope,intercept))


        if len(left_fit)==0:
            left_fit=self.left_fit
        else:
            self.left_fit=left_fit

        left_fit_average=np.average(left_fit,axis=0)
        right_fit_average=np.average(right_fit,axis=0)

        y1=self.image.shape[0]
        y2=int(y1*3/5)

        print(left_fit_average)

        left_slope=left_fit_average[0]
        left_intercept=left_fit_average[1]

        right_slope=right_fit_average[0]
        right_intercept=right_fit_average[1]


        x1_left=int((y1-left_intercept)/left_slope)
        x2_left=int((y2-left_intercept)/left_slope)

        x1_right=int((y1-right_intercept)/right_slope)
        x2_right=int((y2-right_intercept)/right_slope)

        cv2.line(self.line_image,(x1_left,y1),(x2_left,y2),(255,0,0),10)
        cv2.line(self.line_image,(x1_right,y1),(x2_right,y2),(255,0,0),10)

        self.track_image=cv2.addWeighted(self.line_image,0.6,self.image,0.4,0.4)
        if show:
            self.img_show(self.track_image,title=window_name,Cmap='',fig_num=6,group_view=False,save=False)


    def find_lanes(self,show=False,window_name='Masked Image'):
        self.region_of_interest()
        if show:
            cv2.imshow(window_name,self.masked_image)
            cv2.waitKey(1)
#            self.img_show(self.masked_image,title=window_name,Cmap='gray',fig_num=5,group_view=False)
        self.hough_transform()
    def make_video_from_images(self,images_path,video_file_name):
        images = [img for img in os.listdir(images_path) if img.endswith('.png')]
        frame=cv2.imread(os.path.join(images_path,images[0]))
        height,width,layers = frame.shape
        video=cv2.VideoWriter(video_file_name,0,5,(width,height))
        for image in images:
            video.write(cv2.imread(os.path.join(images_path,image)))
        cv2.destroyAllWindows()
        video.release()


def main():
    # path='../data/test_images/solidWhiteCurve.jpg'
    # video_path='../data/test_videos/test2.mp4'
    images_path='../../Results/lane'
    pro=image_processing()
    video_name='lane_object_detect.avi'
    pro.make_video_from_images(images_path,video_name)
    # pro.video_capture(video_path,show=True,save=True)
#    image=pro.single_img_read(path,show=True)
#    pro.canny_edge_detect()
#    pro.find_lanes()

if __name__ == '__main__':
    main()