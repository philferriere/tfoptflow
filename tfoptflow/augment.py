"""
augment.py

Augmentation utility functions and classes.
Uses numpy, to be run on CPU while GPU learns model params.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

To look at later:
    https://github.com/Johswald/Bayesian-FlowNet/blob/master/flownet.py (reproduces original FlowNet aug in np)
    https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/augment.py
    https://github.com/sampepose/flownet2-tf/blob/master/src/dataloader.py
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import cv2
import random  # so we don't interfere with the use of np.random in the dataset loader

from utils import scale

_DBG_AUG_SET = -1

_DEFAULT_AUG_OPTIONS = {
    'aug_type': 'heavy',  # in ['basic', 'heavy']
    'aug_labels': True,  # If True, augment both images and labels; otherwise, only augment images
    'fliplr': 0.5,  # Horizontally flip 50% of images
    'flipud': 0.5,  # Vertically flip 50% of images
    # Translate 50% of images by a value between -5 and +5 percent of original size on x- and y-axis independently
    'translate': (0.5, 0.05),
    'scale': (0.5, 0.05),  # Scale 50% of images by a factor between 95 and 105 percent of original size
    'random_seed': 1969,
}


class Augmenter(object):
    """Augmenter class.
    """

    def __init__(self, options=_DEFAULT_AUG_OPTIONS):
        """Initialize the Augmenter object.
        In 'basic' mode, we only consider 'fliplr' and 'flipud'.
        In 'heavy' mode, we also consider 'translate' and 'scale'.
        Args:
            options: see _DEFAULT_AUG_OPTIONS comments
        """
        self.opts = options
        assert (self.opts['aug_type'] in ['basic', 'heavy'])
        random.seed(self.opts['random_seed'])

    ###
    # Augmentation
    ###
    def augment(self, images, labels=None, as_tuple=False):
        """Augment training samples.
        Args:
            images: Image pairs in format [N, 2, H, W, 3] or list(((H, W, 3),(H, W, 3)))
            labels: Optical flows in format [N, H, W, 2] or list((H, W, 2))
            as_tuple: If True, return image pair tuple; otherwise, return np array in [2, H, W, 3] format
        Returns:
            aug_images: list or array of augmented image pairs.
            aug_labels: list or array of augmented optical flows.
        """
        # Augment image pairs
        assert(isinstance(images, list) or isinstance(images, np.ndarray))
        if labels is not None:
            assert(isinstance(labels, list) or isinstance(labels, np.ndarray))
        do_labels = True if self.opts['aug_labels'] and labels is not None else False

        aug_images, aug_labels = [], []
        for idx in range(len(images)):
            img_pair = images[idx]
            assert(len(img_pair[0].shape) == 3 and (img_pair[0].shape[2] == 1 or img_pair[0].shape[2] == 3))
            assert(len(img_pair[1].shape) == 3 and (img_pair[1].shape[2] == 1 or img_pair[1].shape[2] == 3))

            aug_img_pair = [np.copy(img_pair[0]), np.copy(img_pair[1])]
            if do_labels:
                aug_flow = np.copy(labels[idx])

            # Flip horizontally?
            if self.opts['fliplr'] > 0.:
                rand = random.random()
                if rand < self.opts['fliplr']:
                    aug_img_pair = [np.fliplr(aug_img_pair[0]), np.fliplr(aug_img_pair[1])]
                    if do_labels:
                        aug_flow = np.fliplr(aug_flow)
                        aug_flow[:, :, 0] *= -1

            # Flip vertically?
            if self.opts['flipud'] > 0.:
                rand = random.random()
                if rand < self.opts['flipud']:
                    aug_img_pair = [np.flipud(aug_img_pair[0]), np.flipud(aug_img_pair[1])]
                    if do_labels:
                        aug_flow = np.flipud(aug_flow)
                        aug_flow[:, :, 1] *= -1

            if self.opts['aug_type'] == 'heavy':
                # Translate?
                if self.opts['translate'][0] > 0.:
                    rand = random.random()
                    if rand < self.opts['translate'][0]:
                        h, w, _ = aug_img_pair[0].shape
                        tw = int(random.uniform(-self.opts['translate'][1], self.opts['translate'][1]) * w)
                        th = int(random.uniform(-self.opts['translate'][1], self.opts['translate'][1]) * h)
                        translation_matrix = np.float32([[1, 0, tw], [0, 1, th]])
                        aug_img_pair[1] = cv2.warpAffine(aug_img_pair[1], translation_matrix, (w, h))
                        aug_flow[:, :, 0] += tw
                        aug_flow[:, :, 1] += th

                # Scale? (clipped, so that the result has the same size as the input)
                if self.opts['scale'][0] > 0.:
                    rand = random.random()
                    if rand < self.opts['scale'][0]:
                        ratio = random.uniform(1.0 - self.opts['scale'][1], 1.0 + self.opts['scale'][1])
                        aug_img_pair[0] = scale(aug_img_pair[0], ratio)
                        aug_img_pair[1] = scale(aug_img_pair[1], ratio)
                        if do_labels:
                            aug_flow = scale(aug_flow, ratio)
                            aug_flow *= ratio

            aug_images.append((aug_img_pair[0], aug_img_pair[1]))
            if do_labels:
                aug_labels.append(aug_flow)

        # Return using image in the same format as the input
        if isinstance(images, np.ndarray):
            aug_images = np.asarray(aug_images)
        if do_labels:
            if isinstance(labels, np.ndarray):
                aug_labels = np.asarray(aug_labels)

        if do_labels:
            return aug_images, aug_labels
        else:
            return aug_images

    ###
    # Debug utils
    ###
    def print_config(self):
        """Display configuration values."""
        print("\nAugmenter Configuration:")
        for k, v in self.options.items():
            print("  {:20} {}".format(k, v))
