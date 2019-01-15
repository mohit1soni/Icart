# For Anaconda Activation for vscode terminal "C:/Users/MOHIT SONI/Anaconda2/Scripts/activate.bat"
import thread
import itertools
import ctypes

import pykinect
from pykinect import nui
from pykinect.nui import JointId

import pygame
from pygame.color import THECOLORS
from pygame.locals import *

KINECTEVENT = pygame.USEREVENT
depth_window_size=640,480
video_window_size=640,480

pygame.init()

if hasattr(ctypes.pythonapi, 'Py_InitModule4'):
    Py_ssize_t = ctypes.c_int
elif hasattr(ctypes.pythonapi, 'Py_InitModule4_64'):
    Py_ssize_t = ctypes.c_int64
else:
    raise TypeError("Cannot determine type of Py_ssize_t")
_PyObject_AsWriteBuffer = ctypes.pythonapi.PyObject_AsWriteBuffer
_PyObject_AsWriteBuffer.restype = ctypes.c_int
_PyObject_AsWriteBuffer.argtypes = [ctypes.py_object,
                        ctypes.POINTER(ctypes.c_void_p),
                        ctypes.POINTER(Py_ssize_t)]

def surface_to_array(screen):
    buffer_interface=screen.get_buffer()
    address=ctypes.c_void_p()
    size=Py_ssize_t()
    _PyObject_AsWriteBuffer(buffer_interface,ctypes.byref(address),ctypes.byref(size))
    bytes = (ctypes.c_byte * size.value).from_address(address.value)
    bytes.object = buffer_interface
    return bytes

def depth_frame_ready(frame):
    if video_display:
        return
    with screen_lock:
        address = surface_to_array(screen)
        frame.image.copy_bits(address)
        del address
        pygame.display.update()

def video_frame_ready(frame):
    if not video_display:
        return
    with screen_lock:
        address=surface_to_array(screen)
        frame.image.copy_bits(address)
        del address
        pygame.display.update()


if __name__ == "__main__":
    """ This is the main loop for playing the game  """
    video_display=False
    screen_lock=thread.allocate()
    screen=pygame.display.set_mode(depth_window_size,0,16)
    pygame.display.set_caption("Kinect Connected")
    screen.fill(THECOLORS["blue"])
    kinect=nui.Runtime()
    kinect.skeleton_engine.enabled = True
    kinect.depth_frame_ready += depth_frame_ready
    kinect.video_frame_ready += video_frame_ready
    kinect.video_stream.open(nui.ImageStreamType.Video,2,nui.ImageResolution.Resolution640x480,nui.ImageType.Color)
    kinect.depth_stream.open(nui.ImageStreamType.Depth,2,nui.ImageResolution.Resolution640x480,nui.ImageType.Depth)
    done = False
    while not done:
        e=pygame.event.wait()
        dispInfo = pygame.display.Info()
        if e.type == pygame.QUIT:
            done = True
            break
        elif e.type == KEYDOWN:
            if e.key == K_ESCAPE:
                done = True
                break
            elif e.key == K_d:
                with screen_lock:
                    screen = pygame.display.set_mode(depth_window_size,0,16)
                    video_display=False
            elif e.key == K_v:
                with screen_lock:
                    screen = pygame.display.set_mode(video_window_size,0,32)
                    video_display=True
            elif e.key == K_u:
                kinect.camera.elevation_angle = kinect.camera.elevation_angle + 2
            elif e.key == K_j:
                kinect.camera.elevation_angle = kinect.camera.elevation_angle - 2
            elif e.key == K_x:
                kinect.camera.elevation_angle = 2
