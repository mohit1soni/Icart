from pykinect import nui
import thread
import pygame
from pygame.color import THECOLORS
from pygame.locals import *
import numpy
import time

class Depth_Image(object):
    def __init__(self):
        self.depth_window_size=640,480

    def frame_ready(self):
        """  For async skeltion tracking """
        with nui.Runtime() as kinect :
            kinect.skeleton_engine.enabled = True
            frame = kinect.skeleton_engine.get_next_frame()
            for skeltion in frame.SkeletonData:
                print(skeltion)
                if skeltion.eTrackingState == nui.SkeletonTrackingState.TRACKED:
                    print(skeltion)
            kinect.skeleton_frame_ready += self.frame_ready
            time.clock()

    def instantiate(self):
        """  For sync skeltion tracking """
        with nui.Runtime() as kinect :
            kinect.skeleton_engine.enabled = True
            while True:
                frame = kinect.skeleton_engine.get_next_frame()
                for skeltion in frame.SkeletonData:
                    if skeltion.eTrackingState == nui.SkeletonTrackingState.TRACKED:
                        print(skeltion)

    def depth_frame_ready(self):
        pass
    
    def video_frame_ready(self):
        pass



def main():
    kinect1=Depth_Image()

if __name__ == "__main__":
    main()