# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:19:09 2019

@author: Mohit Kumar Soni
"""
from multiprocessing import Process
import sys

rocket = 0

def func1():
    """ Write function 1 here"""


def func2():
    """ Write function 2 here """
    

if __name__=='__main__':
    p1 = Process(target = func1)
    p1.start()
    p2 = Process(target = func2)
    p2.start()
    # This is where I had to add the join() function.
    p1.join()
    p2.join()