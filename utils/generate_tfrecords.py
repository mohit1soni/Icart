"""
Usage:

# Create train data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record

# Create test data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/test_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/test.record
"""

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
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
# flags.DEFINE_string('label', '', 'Name of class label')
# if your image has more labels input them as
# flags.DEFINE_string('label0', '0', 'speed_limit_20')
# flags.DEFINE_string('label1', '1', 'speed_limit_30')
# flags.DEFINE_string('label2', '2', 'speed_limit_50')
# flags.DEFINE_string('label3', '3', 'speed_limit_60')
# flags.DEFINE_string('label4', '4', 'speed_limit_70')
# flags.DEFINE_string('label5', '5', 'speed_limit_80')
# flags.DEFINE_string('label6', '6', 'restrection_ends_80')
# flags.DEFINE_string('label7', '7', 'speed_limit_100')
# flags.DEFINE_string('label8', '8', 'speed_limit_120')
# flags.DEFINE_string('label9', '9', 'no_overtaking')
# flags.DEFINE_string('label10', '10', 'no_overtaking_trucks')
# flags.DEFINE_string('label11', '11', 'priority_at_next_intersection')
# flags.DEFINE_string('label12', '12', 'priority_road')
# flags.DEFINE_string('label13', '13', 'give_way')
# flags.DEFINE_string('label14', '14', 'stop')
# flags.DEFINE_string('label15', '15', 'no_traffic_both_ways')
# flags.DEFINE_string('label16', '16', 'no_trucks')
# flags.DEFINE_string('label17', '17', 'no_entry')
# flags.DEFINE_string('label18', '18', 'danger')
# flags.DEFINE_string('label19', '19', 'bend_left')
# flags.DEFINE_string('label20', '20', 'bend_right')
# flags.DEFINE_string('label21', '21', 'bend')
# flags.DEFINE_string('label22', '22', 'uneven_road')
# flags.DEFINE_string('label23', '23', 'slippery_road')
# flags.DEFINE_string('label24', '24', 'road_narrows')
# flags.DEFINE_string('label25', '25', 'construction')
# flags.DEFINE_string('label26', '26', 'traffic_signal')
# flags.DEFINE_string('label27', '27', 'pedestrian_crossing')
# flags.DEFINE_string('label28', '28', 'school_crossing')
# flags.DEFINE_string('label29', '29', 'cycles_crossing')
# flags.DEFINE_string('label30', '30', 'snow')
# flags.DEFINE_string('label31', '31', 'animals_ahead')
# flags.DEFINE_string('label32', '32', 'restrection_ends')
# flags.DEFINE_string('label33', '33', 'go_right')
# flags.DEFINE_string('label34', '34', 'go_left')
# flags.DEFINE_string('label35', '35', 'go_straight')
# flags.DEFINE_string('label36', '36', 'go_right_or_straight')
# flags.DEFINE_string('label37', '37', 'go_left_or_straight')
# flags.DEFINE_string('label38', '38', 'keep_right')
# flags.DEFINE_string('label39', '39', 'keep_left')
# flags.DEFINE_string('label40', '40', 'roundabout')
# flags.DEFINE_string('label41', '41', 'restrection_ends_overtaking')
# flags.DEFINE_string('label42', '42', 'restrection_ends_overtaking_trucks')

# and so on.
flags.DEFINE_string('img_path', '', 'Path to images')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
# for multiple labels add more else if statements
# if row_label == FLAGS.label:  # 'ship':
    #     return 1
    # comment upper if statement and uncomment these statements for multiple labelling
def class_text_to_int(row_label):
    if row_label == 'speed_limit_20':
        return 0
    elif row_label == 'speed_limit_30':
        return 1
    elif row_label == 'speed_limit_50':
        return 2
    elif row_label == 'speed_limit_60':
        return 3
    elif row_label == 'speed_limit_70':
        return 4
    elif row_label == 'speed_limit_80':
        return 5
    elif row_label == 'restriction_ends_80':
        return 6
    elif row_label == 'speed_limit_100':
        return 7
    elif row_label == 'speed_limit_120':
        return 8
    elif row_label == 'no_overtaking':
        return 9
    elif row_label == 'no_overtaking_trucks':
        return 10
    elif row_label == 'priority_at_next_intersection':
        return 11
    elif row_label == 'priority_road':
        return 12
    elif row_label == 'give_way':
        return 13
    elif row_label == 'stop':
        return 14
    elif row_label == 'no_traffic_both_ways':
        return 15
    elif row_label == 'no_trucks':
        return 16
    elif row_label == 'no_entry':
        return 17
    elif row_label == 'danger':
        return 18
    elif row_label == 'bend_left':
        return 19
    elif row_label == 'bend_right':
        return 20
    elif row_label == 'bend_danger':
        return 21
    elif row_label == 'uneven_road':
        return 22
    elif row_label == 'slippery_road':
        return 23
    elif row_label == 'road_narrows':
        return 24
    elif row_label == 'construction':
        return 25
    elif row_label == 'traffic_signal':
        return 26
    elif row_label == 'pedestrain_crossing':
        return 27
    elif row_label == 'school_crossing':
        return 28
    elif row_label == 'cycles_crossing':
        return 29
    elif row_label == 'snow':
        return 30
    elif row_label == 'animals_ahead':
        return 31
    elif row_label == 'restriction_ends':
        return 32
    elif row_label == 'go_right':
        return 33
    elif row_label == 'go_left':
        return 34
    elif row_label == 'go_straight':
        return 35
    elif row_label == 'go_right_or_straight':
        return 36
    elif row_label == 'go_left_or_straight':
        return 37
    elif row_label == 'keep_right':
        return 38
    elif row_label == 'keep_left':
        return 39
    elif row_label == 'roundabout':
        return 40
    elif row_label == 'restriction_ends_overtaking':
        return 41
    elif row_label == 'restriction_ends_overtaking_trucks':
        return 42


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    # check if the image format is matching with your images.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))
        print(classes)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    print(os.getcwd())
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), FLAGS.img_path)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()