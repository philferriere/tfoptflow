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
        self._trn_IDs, self._val_IDs, self._tst_IDs = [], [], []
        self._trn_IDs_simpl, self._val_IDs_simpl, self._tst_IDs_simpl = [], [], []
        self._img_trn_path, self._img_val_path, self._img_tst_path = [], [], []
        self._lbl_trn_path, self._lbl_val_path, self._pred_lbl_val_path, self._pred_lbl_tst_path = [], [], [], []
        self.min_flow = np.finfo(np.float32).max
        self.avg_flow = []
        self.max_flow = 0.
        for ds in datasets:
            if ds._trn_IDs is not None:
                self._trn_IDs.extend(ds._trn_IDs)
            if ds._val_IDs is not None:
                self._val_IDs.extend(ds._val_IDs)
            if ds._tst_IDs is not None:
                self._tst_IDs.extend(ds._tst_IDs)
            if ds._trn_IDs_simpl is not None:
                self._trn_IDs_simpl.extend(ds._trn_IDs_simpl)
            if ds._val_IDs_simpl is not None:
                self._val_IDs_simpl.extend(ds._val_IDs_simpl)
            if ds._tst_IDs_simpl is not None:
                self._tst_IDs_simpl.extend(ds._tst_IDs_simpl)
            if ds._img_trn_path is not None:
                self._img_trn_path.extend(ds._img_trn_path)
            if ds._img_val_path is not None:
                self._img_val_path.extend(ds._img_val_path)
            if ds._img_tst_path is not None:
                self._img_tst_path.extend(ds._img_tst_path)
            if ds._lbl_trn_path is not None:
                self._lbl_trn_path.extend(ds._lbl_trn_path)
            if ds._lbl_val_path is not None:
                self._lbl_val_path.extend(ds._lbl_val_path)
            if ds._pred_lbl_val_path is not None:
                self._pred_lbl_val_path.extend(ds._pred_lbl_val_path)
            if ds._pred_lbl_tst_path is not None:
                self._pred_lbl_tst_path.extend(ds._pred_lbl_tst_path)
            self.min_flow = min(self.min_flow, ds.min_flow)
            self.avg_flow.append(ds.avg_flow)
            self.max_flow = max(self.max_flow, ds.max_flow)

        self.avg_flow = np.mean(self.avg_flow)  # yes, this is only an approximation of the average...

        # Load all data in memory, if requested
        if self.opts['in_memory']:
            self._preload_all_samples()

        # Shuffle the data and set trackers
        np.random.seed(self.opts['random_seed'])
        if self.mode in ['train_noval', 'train_with_val']:
            # Train over the original training set, in the first case
            self._trn_ptr = 0
            self.trn_size = len(self._trn_IDs)
            self._trn_idx = np.arange(self.trn_size)
            np.random.shuffle(self._trn_idx)
            if self.mode == 'train_with_val':
                # Train over the training split, validate over the validation split, in the second case
                self._val_ptr = 0
                self.val_size = len(self._val_IDs)
                self._val_idx = np.arange(self.val_size)
                np.random.shuffle(self._val_idx)
            if self.opts['tb_test_imgs'] is True:
                # Make test images available to model in training mode
                self._tst_ptr = 0
                self.tst_size = len(self._tst_IDs)
                self._tst_idx = np.arange(self.tst_size)
                np.random.shuffle(self._tst_idx)
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
            self._tst_ptr = 0
            self.tst_size = len(self._tst_IDs)
            self._tst_idx = np.arange(self.tst_size)
            np.random.shuffle(self._tst_idx)
