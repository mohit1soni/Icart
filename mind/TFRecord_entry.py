import tensorflow as tf
from object_detection.utils import dataset_utils

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