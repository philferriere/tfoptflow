"""
multi_gpus.py

Helpers to train a model using multi-GPU in-graph replication with synchronous updates.
We create one copy of the model (aka, a tower) per device and instruct it to compute forward and backward passes.
The gradients are then averaged and applied on the controller device where all the model’s variables reside.
The controller device is the CPU, meaning that all variables live on the CPU and are copied to the GPUs in each step.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Based on:
    - https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
    Written by The TensorFlow Authors, Copyright 2015 The TensorFlow Authors. All Rights Reserved.
    Licensed under the Apache License 2.0

    - TensorFlow - Multi GPU Computation
    http://blog.s-schoener.com/2017-12-15-parallel-tensorflow-intro/
    Written by Sebastian Schöner, License unknown

    - Tensorflow Multi-GPU VAE-GAN implementation
    https://timsainb.github.io/multi-gpu-vae-gan-in-tensorflow.html
    Written by Sebastian Schöner, License unknown
"""

from __future__ import absolute_import, division, print_function
from tensorflow.python.client import device_lib
import tensorflow as tf

PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable',
          'MutableHashTable', 'MutableHashTableOfTensors', 'MutableDenseHashTable']


def assign_to_device(ops_device, var_device):
    """Returns a function to place variables on the var_device.
    If var_device is not set then the variables will be placed on the default device.
    The best device for shared variables depends on the platform as well as the model.
    Start with CPU:0 and then test GPU:0 to see if there is an improvement.
    Args:
        ops_device: Device for everything but variables. Sample values are /device:GPU:0 and /device:GPU:1.
        var_device: Device to put the variables on. Sample values are /device:CPU:0 or /device:GPU:0.
    Ref:
        - Placing Variables on the cpu using `tf.contrib.layers` functions
        https://github.com/tensorflow/tensorflow/issues/9517
    """

    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return var_device
        else:
            return ops_device

    return _assign


def get_available_gpus():
    """Returns a list of the identifiers of all visible GPUs.
    Ref:
        - How to get current available GPUs in tensorflow?
        https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers. A tower is the name used to describe
    a copy of the model on a device. Note that average_gradients() provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list is over individual gradients. The
        inner list is over the gradient calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Each grad_and_vars looks like the following: ((grad0_gpu0, var0_gpu0),..., (grad0_gpuN, var0_gpuN))
        grads = [grad for grad, _ in grad_and_vars]
        grad = tf.reduce_mean(grads, 0)
        # Keep in mind that the Variables are redundant because they are shared across towers. So, we only need to
        # return the first tower's pointer to the Variable.
        var = grad_and_vars[0][1]
        average_grads.append((grad, var))
    return average_grads
