import dataset_utils
import numpy as np
import os
import tensorflow as tf
url = "http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz"
checkpoints_dir = '/tmp/checkpoints'

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)
