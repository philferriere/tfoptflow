"""
losses.py

Loss functions.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Based on:
    - https://github.com/NVlabs/PWC-Net/blob/master/Caffe/model/train.prototxt
        Copyright (C) 2018 NVIDIA Corporation. All rights reserved. Licensed under the CC BY-NC-SA 4.0 license
    - https://github.com/daigo0927/PWC-Net_tf/blob/master/losses.py
        Written by Daigo Hirooka, Copyright (c) 2018 Daigo Hirooka
        MIT License

Ref:
    Per page 4 of paper, section "Training loss," the loss function used in regular training mode is the same as the
    one used in Dosovitskiy et al's "FlowNet: Learning optical flow with convolutional networks" paper (multiscale
    training loss). For fine-tuning, the loss function used is described at the top of page 5 (robust training loss).
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf


def pwcnet_loss(y, y_hat_pyr, opts):
    """Adds the L2-norm or L1-norm losses at all levels of the pyramid.
    In regular training mode, the L2-norm is used to compute the multiscale loss.
    In fine-tuning mode, the L1-norm is used to compute the robust loss.
    Note that the total loss returned is not regularized.
    Args:
        y: Optical flow groundtruths in [batch_size, H, W, 2] format
        y_hat_pyr: Pyramid of optical flow predictions in list([batch_size, H, W, 2]) format
        opts: options (see below)
        Options:
            pyr_lvls: Number of levels in the pyramid
            alphas: Level weights (scales contribution of loss at each level toward total loss)
            epsilon: A small constant used in the computation of the robust loss, 0 for the multiscale loss
            q: A q<1 gives less penalty to outliers in robust loss, 1 for the multiscale loss
            mode: Training mode, one of ['multiscale', 'robust']
    Returns:
        Loss tensor opp
    Ref:
        - https://github.com/NVlabs/PWC-Net/blob/master/Caffe/model/train.prototxt
    """
    # Use a different norm based on the training mode we're in (training vs fine-tuning)
    norm_order = 2 if opts['loss_fn'] == 'loss_multiscale' else 1

    with tf.name_scope(opts['loss_fn']):
        total_loss = 0.
        _, gt_height, _, _ = tf.unstack(tf.shape(y))

        # Add individual pyramid level losses to the total loss
        for lvl in range(opts['pyr_lvls'] - opts['flow_pred_lvl'] + 1):
            _, lvl_height, lvl_width, _ = tf.unstack(tf.shape(y_hat_pyr[lvl]))

            # Scale the full-size groundtruth to the correct lower res level
            scaled_flow_gt = tf.image.resize_bilinear(y, (lvl_height, lvl_width))
            scaled_flow_gt /= tf.cast(gt_height / lvl_height, dtype=tf.float32)

            # Compute the norm of the difference between scaled groundtruth and prediction
            if opts['use_mixed_precision'] is False:
                y_hat_pyr_lvl = y_hat_pyr[lvl]
            else:
                y_hat_pyr_lvl = tf.cast(y_hat_pyr[lvl], dtype=tf.float32)
            norm = tf.norm(scaled_flow_gt - y_hat_pyr_lvl, ord=norm_order, axis=3)
            level_loss = tf.reduce_mean(tf.reduce_sum(norm, axis=(1, 2)))

            # Scale total loss contribution of the loss at each individual level
            total_loss += opts['alphas'][lvl] * tf.pow(level_loss + opts['epsilon'], opts['q'])

        return total_loss
