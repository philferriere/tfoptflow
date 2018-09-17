"""
lr.py

Adaptive learning rate utility functions.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Based on:
    - The learning rate scheme of the FlowNet2 paper
    https://github.com/NVlabs/PWC-Net/blob/master/Caffe/model/solver.prototxt
    Copyright (C) 2018 NVIDIA Corporation. All rights reserved. Licensed under the CC BY-NC-SA 4.0 license

    - Add support for Cyclic Learning Rate #20785
    https://github.com/tensorflow/tensorflow/pull/20785/commits/e1b30b2c50776fc1e660503d07451a6f169a7ff9
    Written by Mahmoud Aslan, Copyright (c) 2018 Mahmoud Aslan
    MIT License
"""
from __future__ import absolute_import, division, print_function
import tensorflow as tf


def lr_multisteps_long(g_step_op, boundaries=None, values=None):
    """Setup the S<sub>long</sub> learning rate schedule introduced in E. Ilg et al.'s "FlowNet 2.0:
        Evolution of optical flow estimation with deep networks"
        Note that we tailor this schedule to the batch size and number of GPUs.
        If the number of GPUs is one and batch_size is 8, then we use S<sub>long</sub>.
        For every additional GPU, we divide the length of the schedule by that number.
        For every additional 8 samples in the batch size, we divide the length of the schedule by 2.
    Args:
        g_step_op: Global step op
        boundaries: Learning rate boundary changes
        values: Learning rate values after boundary changes
    Based on:
        - https://github.com/NVlabs/PWC-Net/blob/master/Caffe/model/solver.prototxt
        Copyright (C) 2018 NVIDIA Corporation. All rights reserved. Licensed under the CC BY-NC-SA 4.0 license

        # use the learning rate scheme as the FlowNet2 paper
        net: "../model/train.prototxt"
        snapshot_prefix: "flow"
        base_lr: 0.0001
        lr_policy: "multistep"
        gamma: 0.5
        stepvalue: 400000
        stepvalue: 600000
        stepvalue: 800000
        stepvalue: 1000000
        stepvalue: 1200000
        momentum: 0.9
        weight_decay: 0.0004
        display: 100
        max_iter: 1200000
        snapshot: 20000
        solver_mode: GPU
        solver_type: ADAM
        momentum2: 0.999
    Ref:
        Per page 5 of paper, section "Implementation details," we first train the models using the FlyingChairs
        dataset using the S<sub>long</sub> learning rate schedule, starting from 0.0001 and reducing the learning
        rate by half at 0.4M, 0.6M, 0.8M, and 1M iterations.
    """
    if boundaries is None and values is None:
        boundaries = [400000, 600000, 800000, 1000000, 1200000]
        values = [0.0001 / (2 ** boundary) for boundary in range(len(boundaries) + 1)]
    return tf.train.piecewise_constant(g_step_op, boundaries, values, 'lr_multisteps')


def lr_multisteps_fine(g_step_op, boundaries=None, values=None):
    """Setup the S<sub>fine</sub> learning rate schedule introduced in E. Ilg et al.'s "FlowNet 2.0:
    Evolution of optical flow estimation with deep networks"
    Args:
        g_step_op: Global step op
        boundaries: Learning rate boundary changes
        values: Learning rate values after boundary changes
    """
    if boundaries is None and values is None:
        boundaries = [1400000, 1500000, 1600000, 1700000]
        values = [0.00001 / (2 ** boundary) for boundary in range(len(boundaries) + 1)]
    return tf.train.piecewise_constant(g_step_op, boundaries, values, 'lr_multisteps')


def lr_cyclic_long(g_step_op, base_lr=None, max_lr=None, step_size=None):
    """Setup a cyclic learning rate for long pre-training
    Args:
        g_step_op: Global step op
        base_lr: Initial learning rate and minimum bound of the cycle.
        max_lr:  Maximum learning rate bound.
        step_size: Number of iterations in half a cycle.
    """
    if base_lr is None and max_lr is None and step_size is None:
        base_lr = 0.00001
        max_lr = 0.0001
        step_size = 10000
    return _lr_cyclic(g_step_op, base_lr, max_lr, step_size, op_name='lr_cyclic')


def lr_cyclic_fine(g_step_op, base_lr=None, max_lr=None, step_size=None):
    """Setup a cyclic learning rate for fine-tuning
    Args:
        g_step_op: Global step op
        base_lr: Initial learning rate and minimum bound of the cycle.
        max_lr:  Maximum learning rate bound.
        step_size: Number of iterations in half a cycle.
    """
    if base_lr is None and max_lr is None and step_size is None:
        base_lr = 0.000001
        max_lr = 0.00001
        step_size = 10000
    return _lr_cyclic(g_step_op, base_lr, max_lr, step_size, op_name='lr_cyclic')


def _lr_cyclic(g_step_op, base_lr=None, max_lr=None, step_size=None, gamma=0.99994, mode='triangular2', op_name=None):
    """Computes a cyclic learning rate, based on L.N. Smith's "Cyclical learning rates for training neural networks."
    [https://arxiv.org/pdf/1506.01186.pdf]

    This method lets the learning rate cyclically vary between the minimum (base_lr) and the maximum (max_lr)
    achieving improved classification accuracy and often in fewer iterations.

    This code returns the cyclic learning rate computed as:

    ```python
    cycle = floor( 1 + global_step / ( 2 * step_size ) )
    x = abs( global_step / step_size – 2 * cycle + 1 )
    clr = learning_rate + ( max_lr – learning_rate ) * max( 0 , 1 - x )
    ```

    Policies:
        'triangular': Default, linearly increasing then linearly decreasing the learning rate at each cycle.

        'triangular2': The same as the triangular policy except the learning rate difference is cut in half at the end
        of each cycle. This means the learning rate difference drops after each cycle.

        'exp_range': The learning rate varies between the minimum and maximum boundaries and each boundary value
        declines by an exponential factor of: gamma^global_step.

    Args:
        global_step: Session global step.
        base_lr: Initial learning rate and minimum bound of the cycle.
        max_lr:  Maximum learning rate bound.
        step_size: Number of iterations in half a cycle. The paper suggests 2-8 x training iterations in epoch.
        gamma: Constant in 'exp_range' mode gamma**(global_step)
        mode: One of {'triangular', 'triangular2', 'exp_range'}. Default 'triangular'.
        name: String.  Optional name of the operation.  Defaults to 'CyclicLearningRate'.
    Returns:
        The cyclic learning rate.
    """
    assert (mode in ['triangular', 'triangular2', 'exp_range'])
    lr = tf.convert_to_tensor(base_lr, name="learning_rate")
    global_step = tf.cast(g_step_op, lr.dtype)
    step_size = tf.cast(step_size, lr.dtype)

    # computing: cycle = floor( 1 + global_step / ( 2 * step_size ) )
    double_step = tf.multiply(2., step_size)
    global_div_double_step = tf.divide(global_step, double_step)
    cycle = tf.floor(tf.add(1., global_div_double_step))

    # computing: x = abs( global_step / step_size – 2 * cycle + 1 )
    double_cycle = tf.multiply(2., cycle)
    global_div_step = tf.divide(global_step, step_size)
    tmp = tf.subtract(global_div_step, double_cycle)
    x = tf.abs(tf.add(1., tmp))

    # computing: clr = learning_rate + ( max_lr – learning_rate ) * max( 0, 1 - x )
    a1 = tf.maximum(0., tf.subtract(1., x))
    a2 = tf.subtract(max_lr, lr)
    clr = tf.multiply(a1, a2)

    if mode == 'triangular2':
        clr = tf.divide(clr, tf.cast(tf.pow(2, tf.cast(cycle - 1, tf.int32)), tf.float32))
    if mode == 'exp_range':
        clr = tf.multiply(tf.pow(gamma, global_step), clr)

    return tf.add(clr, lr, name=op_name)
