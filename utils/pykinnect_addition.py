from pykinect import nui
import thread
import pygame
import numpy
import time
class Depth_Image(object):
    def __init__(self):
        self.depth_window_size=640,480

    def allocate_thread(self):
        # self.screen_lock=thread.allocate()
        # self.screen=None
        self.temp_surface=pygame.Surface(self.depth_window_size,0,16)

    def depth_frame_ready(self,frame):
        self.allocate_thread()
        # with self.screen_lock:
        frame.image.copy_bits(self.temp_surface._pixels_address)
        arr2D=(pygame.surfarray.pixels2d(self.temp_surface)>>7)& 255
        pygame.surfarray.blit_array(self.screen,arr2D)
        del(arr2D)
        pygame.display.update()

    def play_game(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.depth_window_size,0,8)
        self.screen.set_palette(tuple([(i,i,i) for i in range(256)]))
        pygame.display.set_caption('Depth_Imgage_data')

        with nui.Runtime() as self.kinect:
            self.kinect.depth_frame_ready += self.depth_frame_ready
            # self.video_frmae_ready += video_frmae_ready
            self.kinect.depth_stream.open(nui.ImageStreamType.Depth,2,nui.ImageResolution.Resolution640x480,nui.ImageType.Depth)
            # self.kinect.video_stream.open(nui.ImageStreamType.Video,2,nui.ImageResolution.Resolution640x480,nui.ImageType.Video)
            while True:
                self.event = pygame.event.wait()
                if self.event.type == pygame.QUIT:
                    break


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

def main():
    kinect1=Depth_Image()
    kinect1.play_game()
    # kinect1.frame_ready()
    # kinect.instantiate()
if __name__ == "__main__":
    main()