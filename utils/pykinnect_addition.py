from pykinect import nui
import numpy

class Depth_Image(object):
    def __init__(self):
        pass
    def instantiate(self):
        with nui.Runtime() as kinect :
            kinect.skeleton_engine.enabled = True
            while True:
                frame = kinect.skeleton_engine.get_next_frame()
                for skeltion in frame.SkeletonData:
                    if skeltion.eTrackState == nui.SkeletonTrackingState.TRACKED:
                        print skeltion

def main():
    kinect=Depth_Image()
    kinect.instantiate()

if __name__ == "__main__":
    main()