from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import matplotlib

import matplotlib.pyplot as plt
import math
import numpy as np

import os


import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = '%s-*'

SPLITS_TO_SIZES = {'train': 821, 'validation': 36}

_NUM_CLASSES = 3

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading cifar10.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)

data_dir = '/home/elements/Desktop/datasets/flags_tfrecords/'

def test_dataset():
  with tf.Graph().as_default(): 
      dataset = get_split('train', data_dir)
      data_provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset, common_queue_capacity=32, common_queue_min=1)
      image, label = data_provider.get(['image', 'label'])
      
      with tf.Session() as sess:    
          with slim.queues.QueueRunners(sess):
              for i in xrange(4):
                  np_image, np_label = sess.run([image, label])
                  height, width, _ = np_image.shape
                  class_name = name = dataset.labels_to_names[np_label]
                  
                  plt.figure()
                  plt.imshow(np_image)
                  plt.title('%s, %d x %d' % (name, height, width))
                  plt.axis('off')
                  plt.show()


if __name__ == '__main__':
    test_dataset()
