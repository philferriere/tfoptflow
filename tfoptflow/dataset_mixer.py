"""
dataset_mixer.py

Mixer optical flow dataset class.
Will generate mixed samples from a list of dataset objects (e.g., FlyingChairs and FlyingThings3DHalfRes).
Dataset options useb by the individual datasets and the mixed dataset must be compatible.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""

from __future__ import absolute_import, division, print_function
import numpy as np

from augment import Augmenter
from dataset_base import OpticalFlowDataset, _DEFAULT_DS_TRAIN_OPTIONS

class MixedDataset(OpticalFlowDataset):
    """Mixed optical flow dataset.
    """
    def __init__(self, mode='train_with_val', datasets=None, options=_DEFAULT_DS_TRAIN_OPTIONS):
        """Initialize the MixedDataset object
        Args:
            mode: Possible options: 'train_noval', 'val', 'train_with_val' or 'test'
            datasets: List of dataset objects to mix samples from
            options: see _DEFAULT_DS_TRAIN_OPTIONS comments
        """
        # Only options supported in this initial implementation
        assert (mode in ['train_noval', 'val', 'train_with_val', 'test'])
        self.mode = mode
        self.opts = options

        # Combine dataset fields
        self._train_IDs, self._val_IDs, self._test_IDs = [], [], []
        self._train_IDs_simplified, self._val_IDs_simplified, self._test_IDs_simplified = [], [], []
        self._images_train_path, self._images_val_path, self._images_test_path = [], [], []
        self._labels_train_path, self._labels_val_path, self._pred_labels_val_path, self._pred_labels_test_path = [], [], [], []
        self.min_flow_mag = np.finfo(np.float32).max
        self.avg_flow_mag = []
        self.max_flow_mag = 0.
        for ds in datasets:
            if ds._train_IDs is not None: self._train_IDs.extend(ds._train_IDs)
            if ds._val_IDs is not None: self._val_IDs.extend(ds._val_IDs)
            if ds._test_IDs is not None: self._test_IDs.extend(ds._test_IDs)
            if ds._train_IDs_simplified is not None: self._train_IDs_simplified.extend(ds._train_IDs_simplified)
            if ds._val_IDs_simplified is not None: self._val_IDs_simplified.extend(ds._val_IDs_simplified)
            if ds._test_IDs_simplified is not None: self._test_IDs_simplified.extend(ds._test_IDs_simplified)
            if ds._images_train_path is not None: self._images_train_path.extend(ds._images_train_path)
            if ds._images_val_path is not None: self._images_val_path.extend(ds._images_val_path)
            if ds._images_test_path is not None: self._images_test_path.extend(ds._images_test_path)
            if ds._labels_train_path is not None: self._labels_train_path.extend(ds._labels_train_path)
            if ds._labels_val_path is not None: self._labels_val_path.extend(ds._labels_val_path)
            if ds._pred_labels_val_path is not None: self._pred_labels_val_path.extend(ds._pred_labels_val_path)
            if ds._pred_labels_test_path is not None: self._pred_labels_test_path.extend(ds._pred_labels_test_path)
            self.min_flow_mag = min(self.min_flow_mag, ds.min_flow_mag)
            self.avg_flow_mag.append(ds.avg_flow_mag)
            self.max_flow_mag = max(self.max_flow_mag, ds.max_flow_mag)

        self.avg_flow_mag = np.mean(self.avg_flow_mag) # yes, this is only an approximation of the average...

        # Load all data in memory, if requested
        if self.opts['in_memory']:
             self._preload_all_samples()

        # Shuffle the data and set trackers
        np.random.seed(self.opts['random_seed'])
        if self.mode in ['train_noval', 'train_with_val']:
            # Train over the original training set, in the first case
            self._train_ptr = 0
            self.train_size = len(self._train_IDs)
            self._train_idx = np.arange(self.train_size)
            np.random.shuffle(self._train_idx)
            if self.mode == 'train_with_val':
                # Train over the training split, validate over the validation split, in the second case
                self._val_ptr = 0
                self.val_size = len(self._val_IDs)
                self._val_idx = np.arange(self.val_size)
                np.random.shuffle(self._val_idx)
            if self.opts['tb_test_imgs'] is True:
                # Make test images available to model in training mode
                self._test_ptr = 0
                self.test_size = len(self._test_IDs)
                self._test_idx = np.arange(self.test_size)
                np.random.shuffle(self._test_idx)
            # Instantiate augmenter, if requested
            if self.opts['aug_type'] is not None:
                assert (self.opts['aug_type'] in ['basic', 'heavy'])
                self._aug = Augmenter(self.opts)

        elif self.mode == 'val':
            # Validate over the validation split
            self._val_ptr = 0
            self.val_size = len(self._val_IDs)
            self._val_idx = np.arange(self.val_size)
            np.random.shuffle(self._val_idx)

        else:
            # Test over the entire testing set
            self._test_ptr = 0
            self.test_size = len(self._test_IDs)
            self._test_idx = np.arange(self.test_size)
            np.random.shuffle(self._test_idx)

