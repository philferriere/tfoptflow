"""
mixed_precision.py

Helpers to train a model using mixed-precision training.

Modified by Phil Ferriere

Modifications licensed under the MIT License (see LICENSE for details)

Based on:
    - https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Classification/imagenet/nvcnn_hvd.py
    Written by The TensorFlow Authors, Copyright 2016 The TensorFlow Authors. All Rights Reserved.
    Licensed under the Apache License 2.0

    - 5.6.2. TensorFlow Example
    https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#example_tensorflow
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf


def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable
