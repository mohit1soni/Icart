from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import sys

sys.path.append("../../models/research")

from PIL import Image
from object_detection.utils import dataset_util
for collections import namedtuple,OrderedDict




def create_tf_example(label_and_source_info):
    # TODO START: Populate the following variable in your example
    height = None
    width = None
    filename = None
    encoded_image_data = None
    image_fromat = None # 'jpeg' or 'png'

    xmins=[]
    xmaxs=[]
    y_mins=[]
    y_maxs=[]
    classes_text = []
    classes = []
    # TODO END

    tf_label_and_data = tf.train.Example(feature)

    pass