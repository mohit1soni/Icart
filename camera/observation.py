import numpy as np
import cv2
import glob
import imutils
# global observation_space
global img_width
global img_height

images=glob.glob("..\\data\\observation\\*.jpg")
img=cv2.imread(images[0])
# image=imutils.rotate(img,90)
# print(image[:,0,:])

(img_height,img_width,n_channels)=img.shape
print(img.shape)

observation_space=np.zeros((2*img_height,img_width+2*img_height,3),dtype="uint8")
print(observation_space.shape)

def create_observation_space(image,observation_space,angle=0):
    if angle==0:
        observation_space[img_height:2*img_height,img_height:img_height+img_width,:] = image
        while True:
            cv2.imshow("black",observation_space)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    elif angle == 90:
        image=imutils.rotate_bound(image,90)
        observation_space[img_height-int(img_width/2)-1:img_height-int(img_width/2)+img_width-1,img_height+img_width:,:] = image
        while True:
            cv2.imshow("black",observation_space)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    elif angle == 180:
        image=imutils.rotate_bound(image,180)
        observation_space[0:img_height,img_height:img_height+img_width,:] = image
        while True:
            cv2.imshow("black",observation_space)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    elif angle == 270:
        image=imutils.rotate_bound(image,270)
        observation_space[img_height-int(img_width/2)-1:img_height-int(img_width/2)+img_width-1,0:img_height,:] = image
        while True:
            cv2.imshow("black",observation_space)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



angle=0
for image in images:
    img=cv2.imread(image)
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    create_observation_space(img,observation_space,angle=angle)
    angle = angle + 90

cv2.imwrite("../data/observation/observation_space.jpg",observation_space)
cv2.destroyAllWindows()
