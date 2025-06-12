from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

#anapho_op_library = tf.load_op_library("./anapho_kernels.so")
anapho_op_library = tf.load_op_library("./anapho_kernels_gold_mentions.so")

extract_spans = anapho_op_library.extract_spans
tf.no_gradient("ExtractSpans")
