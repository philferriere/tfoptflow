"""
dataset_base.py

Optical flow dataset base class.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from skimage.io import imread
import cv2

from augment import Augmenter
from optflow import flow_read

if sys.platform.startswith("win"):
    _DATASET_ROOT = 'E:/datasets/'
else:
    _DATASET_ROOT = '/media/EDrive/datasets/'

_DBG_TRAIN_VAL_TEST_SETS = -1  # 128 # -1

_DEFAULT_DS_TRAIN_OPTIONS = {
    'verbose': False,
    'in_memory': False,  # True loads all samples upfront, False loads them on-demand
    'crop_preproc': (384, 448),  # None or (h, w), use (384, 768) for FlyingThings3D
    'scale_preproc': None,  # None or (h, w),
    # 'type': 'final',  # ['clean' | 'final'] for MPISintel, ['noc' | 'occ'] for KITTI, 'into_future' for FlyingThings3D
    'tb_test_imgs': False,  # If True, make test images available to model in training mode
    # Sampling and split options
    'random_seed': 1969,  # random seed used for sampling
    'val_split': 0.03,  # portion of data reserved for the validation split
    # Augmentation options
    'aug_type': 'heavy',  # in [None, 'basic', 'heavy'] to add augmented data to training set
    'aug_labels': True,  # If True, augment both images and labels; otherwise, only augment images
    'fliplr': 0.5,  # Horizontally flip 50% of images
    'flipud': 0.5,  # Vertically flip 50% of images
    # Translate 50% of images by a value between -5 and +5 percent of original size on x- and y-axis independently
    'translate': (0.5, 0.05),
    'scale': (0.5, 0.05),  # Scale 50% of images by a factor between 95 and 105 percent of original size
}

_DEFAULT_DS_TUNE_OPTIONS = {
    'verbose': False,
    'in_memory': False,  # True loads all samples upfront, False loads them on-demand
    'crop_preproc': (384, 768),  # None or (h, w), use (384, 768) for FlyingThings3D
    'scale_preproc': None,  # None or (h, w),
    # ['clean' | 'final'] for MPISintel, ['noc' | 'occ'] for KITTI, 'into_future' for FlyingThings3D
    'type': 'into_future',
    'tb_test_imgs': False,  # If True, make test images available to model in training mode
    # Sampling and split options
    'random_seed': 1969,  # random seed used for sampling
    'val_split': 0.03,  # portion of data reserved for the validation split
    # Augmentation options
    'aug_type': 'heavy',  # in [None, 'basic', 'heavy'] to add augmented data to training set
    'aug_labels': True,  # If True, augment both images and labels; otherwise, only augment images
    'fliplr': 0.5,  # Horizontally flip 50% of images
    'flipud': 0.5,  # Vertically flip 50% of images
    # Translate 50% of images by a value between -5 and +5 percent of original size on x- and y-axis independently
    'translate': (0.5, 0.05),
    'scale': (0.5, 0.05),  # Scale 50% of images by a factor between 95 and 105 percent of original size
}

_DEFAULT_DS_VAL_OPTIONS = {
    'verbose': False,
    'in_memory': False,  # True loads all samples upfront, False loads them on-demand
    'crop_preproc': None,  # None or (h, w),
    'scale_preproc': None,  # None or (h, w),
    'type': 'final',  # ['clean' | 'final'] for MPISintel, ['noc' | 'occ'] for KITTI, 'into_future' for FlyingThings3D
    # Sampling and split options
    'random_seed': 1969,  # random seed used for sampling
    'val_split': 0.03,  # portion of data reserved for the validation split
    # Augmentation options
    'aug_type': None,  # in [None, 'basic', 'heavy'] to add augmented data to training set
}

_DEFAULT_DS_TEST_OPTIONS = {
    'verbose': False,
    'in_memory': False,  # True loads all samples upfront, False loads them on-demand
    'crop_preproc': None,  # None or (h, w),
    'scale_preproc': None,  # None or (h, w),
    'type': 'final',  # ['clean' | 'final'] for MPISintel, ['noc' | 'occ'] for KITTI, 'into_future' for FlyingThings3D
    # Sampling and split options
    'random_seed': 1969,  # random seed used for sampling
    'val_split': 0.03,  # portion of data reserved for the validation split
    # Augmentation options
    # 'aug_type': None,  # in [None, 'basic', 'heavy'] to add augmented data to training set
}


class OpticalFlowDataset(object):
    """Optical flow dataset.
    """

    def __init__(self, mode='train_with_val', ds_root=_DATASET_ROOT, options=_DEFAULT_DS_TRAIN_OPTIONS):
        """Initialize the OFDataset object
        Args:
            mode: Possible values:
                'train_noval', the entire dataset will be used for training (no data set aside for validation)
                'train_with_val', the dataset will be split between a training and a holdout validation set
                'val', the holdout validation set will be used for evaluation
                'val_notrain', the entire dataset will be used to validate the performance of a model
                'test', the dataset will be used to generate predictions on a test set that has no groundtruths
            ds_root: Path to the root of the dataset
            options: see _DEFAULT_DS_TRAIN_OPTIONS comments
        """
        # Only options supported in this initial implementation
        assert (mode in ['train_noval', 'train_with_val', 'val', 'val_notrain', 'test'])
        self.mode = mode
        self.opts = options

        # Setup train/val/test paths and file names
        self._ds_root = ds_root

        # Set the train, val, test, label and prediction label folders
        self.set_folders()

        # Set the names of the train/val/test files that will hold the list of sample/label IDs
        self.set_IDs_filenames()

        self._trn_IDs = self._val_IDs = self._tst_IDs = None
        self._trn_IDs_simpl = self._val_IDs_simpl = self._tst_IDs_simpl = None
        self._images_train = self._labels_train = self._images_val = self._labels_val = self._images_test = None
        self._img_trn_path = self._lbl_trn_path = self._img_val_path = self._lbl_val_path = None
        self._pred_lbl_val_path = self._pred_lbl_tst_path = self._img_tst_path = None

        # Load ID files
        if not self._load_ID_files():
            self.prepare()

        # Collect flow stats - the below data members MUST be set in any class that
        # derives from this base class BEFORE calling this constructor!
        if self.min_flow is None and self.avg_flow is None and self.max_flow is None:
            self._get_flow_stats()

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
            # Instantiate augmenter, if requested
            if self.opts['aug_type'] is not None:
                assert (self.opts['aug_type'] in ['basic', 'heavy'])
                self._aug = Augmenter(self.opts)

        elif self.mode in ['val', 'val_notrain']:
            # Validate over the validation split or turn the whole dataset into an evaluation dataset
            self._val_ptr = 0
            self.val_size = len(self._val_IDs)
            self._val_idx = np.arange(self.val_size)
            # np.random.shuffle(self._val_idx)

        else:
            # Test over the entire testing set
            self._tst_ptr = 0
            self.tst_size = len(self._tst_IDs)
            self._tst_idx = np.arange(self.tst_size)

    ###
    # training/val/test ID files
    ###
    def set_folders(self):
        """Set the train, val, test, label and prediction label folders.
        Override this for each dataset, if necessary.
        Called by the base class on init.
        """
        self._tst_dir = self._val_dir = self._trn_dir = self._ds_root
        self._lbl_dir = f"{self._ds_root}/flow"
        self._pred_lbl_dir = f"{self._ds_root}/flow_pred"

    def set_IDs_filenames(self):
        """Set the names of the train/val/test files that will hold the list of sample/label IDs
        Override this for each dataset, if necessary.
        Called by the base class on init.
        """
        self._trn_IDs_file = f"{self._ds_root}/train_{self.opts['val_split']}split.txt"
        self._val_IDs_file = f"{self._ds_root}/val_{self.opts['val_split']}split.txt"
        self._tst_IDs_file = f"{self._ds_root}/test.txt"

    def prepare(self):
        """Do all the preprocessing needed before training/val/test samples can be used.
        """
        if self.opts['verbose']:
            print("Preparing dataset (one-time operation)...")
        # Create paths files and load them back in
        self._build_ID_sets()
        self._create_ID_files()
        self._load_ID_files()
        if self.opts['verbose']:
            print("... done with preparing the dataset.")

    def _build_ID_sets(self):
        """Build the list of samples and their IDs, split them in the proper datasets.
         Each ID is a tuple of the form (image1, image2, flow).
         This method must be overriden.
        """
        raise NotImplementedError

    def _get_flow_stats(self):
        """Get the min, avg, max flow of the training data according to OpenCV.
        This will allow us to normalize the rendering of flows to images across the entire dataset. Why?
        Because low magnitude flows should appear lighter than high magnitude flows when rendered as images.
        """
        flow_mags = []
        desc = "Collecting training flow stats"
        num_flows = len(self._lbl_trn_path)
        with tqdm(total=num_flows, desc=desc, ascii=True, ncols=100) as pbar:
            for flow_path in self._lbl_trn_path:
                pbar.update(1)
                flow = flow_read(flow_path)
                flow_magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                nans = np.isnan(flow_magnitude)
                if np.any(nans):
                    nans = np.where(nans)
                    flow_magnitude[nans] = 0.
                flow_mags.append(flow_magnitude)
        self.min_flow, self.max_flow = np.min(flow_mags), np.max(flow_mags)
        self.avg_flow = np.mean(flow_mags)
        print(
            f"train flow min={self.min_flow}, avg={self.avg_flow}, max={self.max_flow} ({num_flows} flows)")

    def _create_ID_files(self):
        """Create the ID files for each split of the dataset
        """
        for file, IDs in [(self._trn_IDs_file, self._trn_IDs), (self._val_IDs_file,
                                                                self._val_IDs), (self._tst_IDs_file, self._tst_IDs)]:
            with open(file, 'w') as f:
                f.write('\n'.join('{}###{}###{}'.format(ID[0], ID[1], ID[2]) for ID in IDs))

    def _load_ID_files(self):
        """Load the ID files and build the full file paths associated with those IDs
        Returns:
              True if ID files were loaded, False if ID files weren't found
        """
        if self.mode in ['train_noval', 'train_with_val']:
            if not os.path.exists(self._trn_IDs_file) or not os.path.exists(self._val_IDs_file):
                return False

            with open(self._trn_IDs_file, 'r') as f:
                self._trn_IDs = f.readlines()
                self._trn_IDs = [tuple(ID.rstrip().split("###")) for ID in self._trn_IDs]

            with open(self._val_IDs_file, 'r') as f:
                self._val_IDs = f.readlines()
                self._val_IDs = [tuple(ID.rstrip().split("###")) for ID in self._val_IDs]

            self._img_trn_path = [(self._trn_dir + '/' + ID[0], self._trn_dir + '/' + ID[1]) for ID in self._trn_IDs]
            self._lbl_trn_path = [self._trn_lbl_dir + '/' + ID[2] for ID in self._trn_IDs]

            if self.mode == 'train_noval':
                # Train over the original training set (no validation split)
                self._trn_IDs += self._val_IDs
                for ID in self._val_IDs:
                    self._img_trn_path.append((self._val_dir + '/' + ID[0], self._val_dir + '/' + ID[1]))
                    self._lbl_trn_path.append(self._val_lbl_dir + '/' + ID[2])
            else:
                # Train over the training split, validate over the validation split
                self._img_val_path, self._lbl_val_path, self._pred_lbl_val_path = [], [], []
                for ID in self._val_IDs:
                    self._img_val_path.append((self._val_dir + '/' + ID[0], self._val_dir + '/' + ID[1]))
                    self._lbl_val_path.append(self._val_lbl_dir + '/' + ID[2])
                    lbl_id = ID[2].replace('.pfm', '.flo').replace('.png', '.flo')
                    self._pred_lbl_val_path.append(self._val_pred_lbl_dir + '/' + lbl_id)

            if self.opts['tb_test_imgs'] is True:
                # Make test images available to model in training mode
                if not os.path.exists(self._tst_IDs_file):
                    return False

                with open(self._tst_IDs_file, 'r') as f:
                    self._tst_IDs = f.readlines()
                    self._tst_IDs = [tuple(ID.rstrip().split("###")) for ID in self._tst_IDs]

                self._img_tst_path, self._pred_lbl_tst_path = [], []
                for ID in self._tst_IDs:
                    self._img_tst_path.append((self._tst_dir + '/' + ID[0], self._tst_dir + '/' + ID[1]))
                    self._pred_lbl_tst_path.append(self._tst_pred_lbl_dir + '/' + ID[2])

        elif self.mode in ['val', 'val_notrain']:
            # Validate over the validation split
            if not os.path.exists(self._val_IDs_file):
                return False

            with open(self._val_IDs_file, 'r') as f:
                self._val_IDs = f.readlines()
                self._val_IDs = [tuple(ID.rstrip().split("###")) for ID in self._val_IDs]

            if self.mode == 'val_notrain':
                with open(self._trn_IDs_file, 'r') as f:
                    self._trn_IDs = f.readlines()
                    self._trn_IDs = [tuple(ID.rstrip().split("###")) for ID in self._trn_IDs]
                self._val_IDs += self._trn_IDs

            self._img_val_path, self._lbl_val_path, self._pred_lbl_val_path = [], [], []
            for ID in self._val_IDs:
                self._img_val_path.append((self._val_dir + '/' + ID[0], self._val_dir + '/' + ID[1]))
                self._lbl_val_path.append(self._val_lbl_dir + '/' + ID[2])
                lbl_id = ID[2].replace('.pfm', '.flo').replace('.png', '.flo')
                self._pred_lbl_val_path.append(self._val_pred_lbl_dir + '/' + lbl_id)

        else:
            # Test over the entire testing set
            if not os.path.exists(self._tst_IDs_file):
                return False

            with open(self._tst_IDs_file, 'r') as f:
                self._tst_IDs = f.readlines()
                self._tst_IDs = [tuple(ID.rstrip().split("###")) for ID in self._tst_IDs]

            self._img_tst_path, self._pred_lbl_tst_path = [], []
            for ID in self._tst_IDs:
                self._img_tst_path.append((self._tst_dir + '/' + ID[0], self._tst_dir + '/' + ID[1]))
                self._pred_lbl_tst_path.append(self._tst_pred_lbl_dir + '/' + ID[2])

        # Build a list of simplified IDs for Tensorboard logging
        if self._trn_IDs is not None:
            self._trn_IDs_simpl = self.simplify_IDs(self._trn_IDs)
        if self._val_IDs is not None:
            self._val_IDs_simpl = self.simplify_IDs(self._val_IDs)
        if self._tst_IDs is not None:
            self._tst_IDs_simpl = self.simplify_IDs(self._tst_IDs)

        if _DBG_TRAIN_VAL_TEST_SETS != -1:  # Debug mode only
            if self._trn_IDs is not None:
                self._trn_IDs = self._trn_IDs[0:_DBG_TRAIN_VAL_TEST_SETS]
            if self._img_trn_path is not None:
                self._img_trn_path = self._img_trn_path[0:_DBG_TRAIN_VAL_TEST_SETS]
            if self._lbl_trn_path is not None:
                self._lbl_trn_path = self._lbl_trn_path[0:_DBG_TRAIN_VAL_TEST_SETS]
            if self._val_IDs is not None:
                self._val_IDs = self._val_IDs[0:_DBG_TRAIN_VAL_TEST_SETS]
            if self._img_val_path is not None:
                self._img_val_path = self._img_val_path[0:_DBG_TRAIN_VAL_TEST_SETS]
            if self._lbl_val_path is not None:
                self._lbl_val_path = self._lbl_val_path[0:_DBG_TRAIN_VAL_TEST_SETS]
            if self._pred_lbl_val_path is not None:
                self._pred_lbl_val_path = self._pred_lbl_val_path[0:_DBG_TRAIN_VAL_TEST_SETS]
            if self._tst_IDs is not None:
                self._tst_IDs = self._tst_IDs[0:_DBG_TRAIN_VAL_TEST_SETS]
            if self._img_tst_path is not None:
                self._img_tst_path = self._img_tst_path[0:_DBG_TRAIN_VAL_TEST_SETS]
            if self._pred_lbl_tst_path is not None:
                self._pred_lbl_tst_path = self._pred_lbl_tst_path[0:_DBG_TRAIN_VAL_TEST_SETS]

        return True

    ###
    # Batch Management
    ###
    def _preload_all_samples(self):
        """Preload all samples (input image pairs + associated flows) in memory.
        """
        if self.mode in ['train_noval', 'train_with_val']:

            self._images_train, self._labels_train = [], []
            desc = "Loading train image pairs & flows"
            with tqdm(total=len(self._img_trn_path), desc=desc, ascii=True, ncols=100) as pbar:
                for n, image_path in enumerate(self._img_trn_path):
                    pbar.update(1)
                    label_path = self._lbl_trn_path[n]
                    image, label = self._load_sample(image_path, label_path)
                    self._labels_train.append(label)
                    self._images_train.append(image)

            if self.mode == 'train_with_val':
                self._images_val, self._labels_val = [], []
                desc = "Loading val image pairs & flows"
                with tqdm(total=len(self._img_val_path), desc=desc, ascii=True, ncols=100) as pbar:
                    for n, image_path in enumerate(self._img_val_path):
                        pbar.update(1)
                        label_path = self._lbl_val_path[n]
                        image, label = self._load_sample(image_path, label_path, preprocess=False)
                        self._labels_val.append(label)
                        self._images_val.append(image)

            if self.opts['tb_test_imgs'] is True:
                self._images_test = []
                desc = "Loading test samples"
                with tqdm(total=len(self._img_tst_path), desc=desc, ascii=True, ncols=100) as pbar:
                    for image_path in self._img_tst_path:
                        pbar.update(1)
                        self._images_test.append(self._load_sample(image_path, preprocess=False))

        elif self.mode in ['val', 'val_notrain']:

            self._images_val, self._labels_val = [], []
            desc = "Loading val image pairs & flows"
            with tqdm(total=len(self._img_val_path), desc=desc, ascii=True, ncols=100) as pbar:
                for n, image_path in enumerate(self._img_val_path):
                    pbar.update(1)
                    label_path = self._lbl_val_path[n]
                    image, label = self._load_sample(image_path, label_path, preprocess=False)
                    self._labels_val.append(label)
                    self._images_val.append(image)

        elif self.mode == 'test':
            self._images_test = []
            desc = "Loading test samples"
            with tqdm(total=len(self._img_tst_path), desc=desc, ascii=True, ncols=100) as pbar:
                for image_path in self._img_tst_path:
                    pbar.update(1)
                    self._images_test.append(self._load_sample(image_path, preprocess=False))

    def next_batch(self, batch_size, split='train'):
        """Get next batch of samples and labels (input image pairs + associated flows)
        In '*_with_pred_paths' mode, also return a destination folder where to save predicted flows.
        Args:
            batch_size: Size of the batch
            split: 'train', 'val', 'val_with_preds', 'val_with_pred_paths', 'test', or 'test_with_pred_paths'
        In training and validation mode, returns:
            images: Batch of image pairs in format [N, 2, H, W, 3]
            labels: Batch of optical flows in format [N, H, W, 2], or file paths to predicted label
        In testing mode, returns:
            images: Batch of RGB images in format [N, 2, H, W, 3]
            pred_folder: List of output folders where to save the predicted instance masks
        """
        assert(split in ['train', 'val', 'val_with_preds', 'val_with_pred_paths', 'test', 'test_with_pred_paths'])

        # Come up with list of indices to load
        if split == 'train':
            assert(self.mode in ['train_noval', 'train_with_val'])
            if self._trn_ptr + batch_size < self.trn_size:
                idx = np.array(self._trn_idx[self._trn_ptr:self._trn_ptr + batch_size])
                new_ptr = self._trn_ptr + batch_size
            else:
                old_idx = np.array(self._trn_idx[self._trn_ptr:])
                np.random.shuffle(self._trn_idx)
                new_ptr = (self._trn_ptr + batch_size) % self.trn_size
                idx = np.concatenate((old_idx, np.array(self._trn_idx[:new_ptr])))

        elif split in ['val', 'val_with_preds', 'val_with_pred_paths']:
            assert(self.mode in ['val', 'val_notrain', 'train_with_val'])
            if self._val_ptr + batch_size < self.val_size:
                idx = np.array(self._val_idx[self._val_ptr:self._val_ptr + batch_size])
                new_ptr = self._val_ptr + batch_size
            else:
                old_idx = np.array(self._val_idx[self._val_ptr:])
                # np.random.shuffle(self._val_idx)
                new_ptr = (self._val_ptr + batch_size) % self.val_size
                idx = np.concatenate((old_idx, np.array(self._val_idx[:new_ptr])))

        elif split in ['test', 'test_with_pred_paths']:
            assert (self.mode == 'test')
            if self._tst_ptr + batch_size < self.tst_size:
                new_ptr = self._tst_ptr + batch_size
                idx = list(range(self._tst_ptr, self._tst_ptr + batch_size))
            else:
                new_ptr = (self._tst_ptr + batch_size) % self.tst_size
                idx = list(range(self._tst_ptr, self.tst_size)) + list(range(0, new_ptr))

        # Move pointers forward
        if split == 'train':
            self._trn_ptr = new_ptr
        elif split in ['val', 'val_with_preds', 'val_with_pred_paths']:
            self._val_ptr = new_ptr
        elif split in ['test', 'test_with_pred_paths']:
            self._tst_ptr = new_ptr

        # Return samples and labels
        return self.get_samples(idx=idx, split=split, as_list=False, simple_IDs=True)

    ###
    # Sample loaders and getters
    ###

    def _load_sample(self, image_path=None, label_path=None, preprocess=True, as_tuple=False, is_training=False):
        """Load a properly formatted sample (image pair + optical flow)
        Args:
            image_path: Pair of image paths, if any
            label_path: Path to optical flow, if any
            preprocess: If True, apply preprocessing steps (cropping + scaling); otherwise, don't
            as_tuple: If true, return image pair as a tuple; otherwise, return a np array in [2, H, W, 3] format
            is_training: If true, sample is a training sample subject to data augmentation
        Returns:
            image: Image pair in format [2, H, W, 3] or ([H, W, 3],[H, W, 3]), if any
            label: Optical flow in format [W, H, 2], if any
        """
        # Read in RGB image, if any
        if image_path:
            image1, image2 = imread(image_path[0]), imread(image_path[1])
            assert(len(image1.shape) == 3 and image1.shape[2] == 3 and len(image2.shape) == 3 and image2.shape[2] == 3)

        # Read in label, if any
        if label_path:
            label = flow_read(label_path)
            assert (len(label.shape) == 3 and label.shape[2] == 2)
        else:
            label = None

        # Return image and/or label
        if label_path:
            if image_path:
                if as_tuple:
                    return (image1, image2), label
                else:
                    return np.array([image1, image2]), label
            else:
                return label
        else:
            if image_path:
                if as_tuple:
                    return (image1, image2)
                else:
                    return np.array([image1, image2])

    def _augment_sample(self, image=None, label=None, as_tuple=False):
        """Augment sample (input image pair + associated flow, if any)
        Args:
            image: Image pair
            label: optical flow, if any
            as_tuple: If True, return image pair tuple; otherwise, return np array in [2, H, W, 3] format
        Returns:
            image: Augmented image pair in format ([H, W, 3],[H, W, 3]) or [2, H, W, 3]?
            label: Augmented label in format [H, W, 2], if any label
        """
        # Use augmentation, if requested
        if label is None:
            assert(self._aug.opts['aug_labels'] is False)
            # TODO Test this code path with basic horizontal flipping
            aug_image = self._aug.augment([image], None, as_tuple)
            image = aug_image[0]
        else:
            aug_image, aug_label = self._aug.augment([image], [label], as_tuple)
            image, label = aug_image[0], aug_label[0]

        # Return image and label
        if as_tuple:
            return (image[0], image[1]), label
        else:
            return np.array([image[0], image[1]]), label

    def _get_train_samples(self, idx, as_tuple=False, simple_IDs=False):
        """Get training images with associated labels
        Args:
            idx: List of sample indices to return
            as_tuple: If True, return image pairs as tuples; otherwise, return them as np arrays in [2, H, W, 3] format
            simple_IDs: If True, return concatenated IDs; otherwise, return individual image and label IDs
        Returns:
            images: List of RGB images in format list(([H, W, 3],[H, W, 3])) or list([2, H, W, 3])
            labels: List of labels in format list([H, W, 2])
            IDs: List of IDs in format list(str) or list([str,str,str])
        """
        images, labels, IDs = [], [], []
        for l in idx:
            if self.opts['in_memory']:
                image = self._images_train[l]
                label = self._labels_train[l]
            else:
                image, label = self._load_sample(self._img_trn_path[l], self._lbl_trn_path[l],
                                                 is_training=True)

            # Crop images and/or labels to a fixed size, if requested
            if self.opts['crop_preproc'] is not None:
                h, w = image[0].shape[:2]
                h_max, w_max = self.opts['crop_preproc']
                assert (h >= h_max and w >= w_max)
                max_y_offset, max_x_offset = h - h_max, w - w_max
                if max_y_offset > 0 or max_x_offset > 0:
                    y_offset = np.random.randint(max_y_offset + 1)
                    x_offset = np.random.randint(max_x_offset + 1)
                    # The following assumes the image pair is in [2,H,W,3] format
                    image = image[:, y_offset:y_offset + h_max, x_offset:x_offset + w_max, :]
                    label = label[y_offset:y_offset + h_max, x_offset:x_offset + w_max, :]

            # Scale images and/or labels to a fixed size, if requested
            if self.opts['scale_preproc'] is not None:
                scale_shape = (int(self.opts['scale_preproc'][0]), int(self.opts['scale_preproc'][1]))
                image[0] = cv2.resize(image[0], scale_shape)
                image[1] = cv2.resize(image[1], scale_shape)
                label = cv2.resize(label, scale_shape) * scale_shape[0] / image[0].shape[0]

            # Augment the samples, if requested
            if self.opts['aug_type'] is not None:
                image, label = self._augment_sample((image[0], image[1]), label, as_tuple)

            # Don't move augmentation to _load_sample() otherwise, if the samples are in-memory they will have been
            # augmented once and that's it, the first time they were loaded. The augmentation code needs to be
            # here so that, no matter the memory mode, every sample is augmented differently with every batch

            images.append(image)
            labels.append(label)
            if simple_IDs is True:
                IDs.append(self._trn_IDs_simpl[l])
            else:
                IDs.append(self._trn_IDs[l])

        return images, labels, IDs

    def _get_val_samples(self, idx, as_tuple=False, simple_IDs=False):
        """Get validation images with associated labels
        Args:
            idx: List of sample indices to return
            as_tuple: If True, return image pairs as tuples; otherwise, return them as np arrays in [2, H, W, 3] format
            simple_IDs: If True, return concatenated IDs; otherwise, return individual image and label IDs
        Returns:
            images: List of RGB images in format list(([H, W, 3],[H, W, 3])) or list([2, H, W, 3])
            labels: List of labels in format list([H, W, 2])
            IDs: List of IDs in format list(str) or list([str,str,str])
        """
        images, labels, IDs = [], [], []
        for l in idx:
            if self.opts['in_memory']:
                image = self._images_val[l]
                label = self._labels_val[l]
            else:
                image, label = self._load_sample(
                    self._img_val_path[l], self._lbl_val_path[l], preprocess=False, as_tuple=as_tuple)

            # We also must crop validation images and labels to a fixed size during online evaluation.
            # Why do this with validation data? Because we are currently stuck with having the same
            # batch size between the training and validation data sets.  Since our images all have
            # different sizes, if we want to batch validation samples, they must all have the same size...
            # Crop images and/or labels to a fixed size, if requested
            if self.opts['crop_preproc'] is not None:
                h, w = image[0].shape[:2]
                h_max, w_max = self.opts['crop_preproc']
                assert (h >= h_max and w >= w_max)
                max_y_offset, max_x_offset = h - h_max, w - w_max
                if max_y_offset > 0 or max_x_offset > 0:
                    y_offset = np.random.randint(max_y_offset + 1)
                    x_offset = np.random.randint(max_x_offset + 1)
                    # The following assumes the image pair is in [2,H,W,3] format
                    image = image[:, y_offset:y_offset + h_max, x_offset:x_offset + w_max, :]
                    label = label[y_offset:y_offset + h_max, x_offset:x_offset + w_max, :]

            # Scale images and/or labels to a fixed size, if requested
            if self.opts['scale_preproc'] is not None:
                scale_shape = (int(self.opts['scale_preproc'][0]), int(self.opts['scale_preproc'][1]))
                image[0] = cv2.resize(image[0], scale_shape)
                image[1] = cv2.resize(image[1], scale_shape)
                label = cv2.resize(label, scale_shape) * scale_shape[0] / image[0].shape[0]

            images.append(image)
            labels.append(label)
            if simple_IDs is True:
                IDs.append(self._val_IDs_simpl[l])
            else:
                IDs.append(self._val_IDs[l])

        return images, labels, IDs

    def _get_val_samples_with_preds(self, idx, as_tuple=False, simple_IDs=False):
        """Get validation images with associated labels and predictions
        Args:
            idx: List of sample indices to return
            as_tuple: If True, return image pairs as tuples; otherwise, return them as np arrays in [2, H, W, 3] format
            simple_IDs: If True, return concatenated IDs; otherwise, return individual image and label IDs
        Returns:
            images: List of RGB images in format list(([H, W, 3],[H, W, 3])) or list([2, H, W, 3])
            labels: List of labels in format list([H, W, 2])
            pred_labels: List of predicted labels in format list([H, W, 2])
            IDs: List of IDs in format list(str) or list([str,str,str])
        """
        images, labels, IDs = self._get_val_samples(idx, as_tuple=as_tuple, simple_IDs=simple_IDs)

        pred_labels = []
        for l in idx:
            if os.path.exists(self._pred_lbl_val_path[l]):
                pred_label = self._load_sample(None, self._pred_lbl_val_path[l], preprocess=False, as_tuple=as_tuple)
                pred_labels.append(pred_label)

        return images, labels, pred_labels, IDs

    def _get_val_samples_with_pred_paths(self, idx, as_tuple=False, simple_IDs=False):
        """Get validation images with associated labels and prediction paths
        Args:
            as_tuple: If True, return image pairs as tuples; otherwise, return them as np arrays in [2, H, W, 3] format
            idx: List of sample indices to return
            simple_IDs: If True, return concatenated IDs; otherwise, return individual image and label IDs
        Returns:
            images: List of RGB images in format list(([H, W, 3],[H, W, 3])) or list([2, H, W, 3])
            labels: List of labels in format list([H, W, 2])
            pred_label_paths: List of paths to the predicted labels
            IDs: List of IDs in format list(str) or list([str,str,str])
        """
        images, labels, IDs = self._get_val_samples(idx, as_tuple=as_tuple, simple_IDs=simple_IDs)

        pred_label_paths = []
        for l in idx:
            pred_label_paths.append(self._pred_lbl_val_path[l])

        return images, labels, pred_label_paths, IDs

    def _get_test_samples_with_preds(self, idx, as_tuple=False, simple_IDs=False):
        """Get test images with predicted labels
        Args:
            idx: List of sample indices to return
            as_tuple: If True, return image pairs as tuples; otherwise, return them as np arrays in [2, H, W, 3] format
            simple_IDs: If True, return concatenated IDs; otherwise, return individual image and label IDs
        Returns:
            images: List of RGB images in format list(([H, W, 3],[H, W, 3])) or list([2, H, W, 3])
            pred_labels: List of predicted labels in format list([H, W, 2])
            IDs: List of IDs in format list(str) or list([str,str,str])
        """
        images, pred_labels, IDs = [], [], []
        for l in idx:
            if self.opts['in_memory']:
                image = self._images_test[l]
            else:
                image = self._load_sample(self._img_tst_path[l], preprocess=False, as_tuple=as_tuple)
            images.append(image)
            if os.path.exists(self._pred_lbl_tst_path[l]):
                pred_label = self._load_sample(
                    None,
                    self._pred_lbl_tst_path[l],
                    preprocess=False,
                    as_tuple=as_tuple)
                pred_labels.append(pred_label)
            if simple_IDs is True:
                IDs.append(self._tst_IDs_simpl[l])
            else:
                IDs.append(self._tst_IDs[l])

        return images, pred_labels, IDs

    def _get_test_samples_with_pred_paths(self, idx, as_tuple=False, simple_IDs=False):
        """Get test images with predicted labels
        Args:
            idx: List of sample indices to return
            as_tuple: If True, return image pairs as tuples; otherwise, return them as np arrays in [2, H, W, 3] format
            simple_IDs: If True, return concatenated IDs; otherwise, return individual image and label IDs
        Returns:
            images: List of RGB images in format list(([H, W, 3],[H, W, 3])) or list([2, H, W, 3])
            pred_label_paths: List of paths to the predicted labels
            IDs: List of IDs in format list(str) or list([str,str,str])
        """
        images, pred_label_paths, IDs = [], [], []
        for l in idx:
            if self.opts['in_memory']:
                image = self._images_test[l]
            else:
                image = self._load_sample(self._img_tst_path[l], preprocess=False, as_tuple=as_tuple)
            images.append(image)
            pred_label_paths.append(self._pred_lbl_tst_path[l])
            if simple_IDs is True:
                IDs.append(self._tst_IDs_simpl[l])
            else:
                IDs.append(self._tst_IDs[l])

        return images, pred_label_paths, IDs

    def get_samples(
            self,
            num_samples=0,
            idx=None,
            split='val',
            as_list=True,
            deterministic=False,
            as_tuple=False,
            simple_IDs=False):
        """Get a few (or all) random (or ordered) samples from the dataset.
        Used for debugging purposes (testing how the model is improving over time, for instance).
        If sampling from the training/validation set, there is a label; otherwise, there isn't.
        Note that this doesn't return a valid np array if the images don't have the same size.
        Args:
            num_samples: Number of samples to return (used only if idx is set to None)
            idx: Specific list of indices to pull from the dataset split (no need to set num_samples in this case)
            split: 'train','val','val_with_preds','val_with_pred_paths','test','test_with_preds','test_with_pred_paths'
            as_list: Return as list or np array?
            return_IDs: If True, also return ID of the sample
            deterministic: If True, return first num_samples samples, otherwise, sample randomly
            as_tuple: If True, return image pairs as tuples; otherwise, return them as np arrays in [2, H, W, 3] format
            simple_IDs: If True, return concatenated IDs; otherwise, return individual image and label IDs
        In training and validation mode, returns:
            images: Batch of image pairs in format [num_samples, 2, H, W, 3] or list([2, H, W, 3])
            labels: Batch of optical flows in format [num_samples, H, W, 2] or list([H, W, 2])
            IDs: List of ID strings in list or np.ndarray format
        In testing mode, returns:
            images: Batch of image pairs in format [num_samples, 2, H, W, 3]
            output_files: List of output file names that match the input file names
            IDs: List of ID strings in list or np.ndarray format
        """
        assert(idx is not None or num_samples > 0)

        if split == 'train':
            assert(self.mode in ['train_noval', 'train_with_val'])
            if idx is None:
                if deterministic:
                    idx = self._trn_idx[0:num_samples]
                else:
                    idx = np.random.choice(self._trn_idx, size=num_samples, replace=False)

            images, labels, IDs = self._get_train_samples(idx, as_tuple=as_tuple, simple_IDs=simple_IDs)

            if as_list:
                return images, labels, IDs
            else:
                return map(np.asarray, (images, labels, IDs))

        elif split == 'val':
            assert(self.mode in ['val', 'val_notrain', 'train_with_val'])
            if idx is None:
                if deterministic:
                    idx = self._val_idx[0:num_samples]
                else:
                    idx = np.random.choice(self._val_idx, size=num_samples, replace=False)

            images, gt_labels, IDs = self._get_val_samples(idx, as_tuple=as_tuple, simple_IDs=simple_IDs)

            if as_list:
                return images, gt_labels, IDs
            else:
                return map(np.asarray, (images, gt_labels, IDs))

        elif split == 'val_with_preds':
            assert(self.mode in ['val', 'val_notrain', 'train_with_val'])
            if idx is None:
                if deterministic:
                    idx = self._val_idx[0:num_samples]
                else:
                    idx = np.random.choice(self._val_idx, size=num_samples, replace=False)

            images, gt_labels, pred_labels, IDs = self._get_val_samples_with_preds(
                idx, as_tuple=as_tuple, simple_IDs=simple_IDs)

            if as_list:
                return images, gt_labels, pred_labels, IDs
            else:
                return map(np.asarray, (images, gt_labels, pred_labels, IDs))

        elif split == 'val_with_pred_paths':
            assert(self.mode in ['val', 'val_notrain', 'train_with_val'])
            if idx is None:
                if deterministic:
                    idx = self._val_idx[0:num_samples]
                else:
                    idx = np.random.choice(self._val_idx, size=num_samples, replace=False)

            images, gt_labels, pred_label_paths, IDs = self._get_val_samples_with_pred_paths(
                idx, as_tuple=as_tuple, simple_IDs=simple_IDs)

            if as_list:
                return images, gt_labels, pred_label_paths, IDs
            else:
                return map(np.asarray, (images, gt_labels, pred_label_paths, IDs))

        elif split == 'test':
            if idx is None:
                if deterministic:
                    idx = self._tst_idx[0:num_samples]
                else:
                    idx = np.random.choice(self._tst_idx, size=num_samples, replace=False)

            images, IDs = [], []
            for l in idx:
                if self.opts['in_memory']:
                    image = self._images_test[l]
                else:
                    image = self._load_sample(self._img_tst_path[l], preprocess=False, as_tuple=as_tuple)
                images.append(image)
                if simple_IDs is True:
                    IDs.append(self._tst_IDs_simpl[l])
                else:
                    IDs.append(self._tst_IDs[l])

            if as_list:
                return images, IDs
            else:
                return map(np.asarray, (images, IDs))

        elif split == 'test_with_preds':
            if idx is None:
                if deterministic:
                    idx = self._tst_idx[0:num_samples]
                else:
                    idx = np.random.choice(self._tst_idx, size=num_samples, replace=False)

            images, pred_labels, IDs = self._get_test_samples_with_preds(idx, as_tuple=as_tuple, simple_IDs=simple_IDs)

            if as_list:
                return images, pred_labels, IDs
            else:
                return map(np.asarray, (images, pred_labels, IDs))

        elif split == 'test_with_pred_paths':
            if idx is None:
                if deterministic:
                    idx = self._tst_idx[0:num_samples]
                else:
                    idx = np.random.choice(self._tst_idx, size=num_samples, replace=False)

            images, pred_label_paths, IDs = self._get_test_samples_with_pred_paths(
                idx, as_tuple=as_tuple, simple_IDs=simple_IDs)

            if as_list:
                return images, pred_label_paths, IDs
            else:
                return map(np.asarray, (images, pred_label_paths, IDs))

        else:
            return None, None

    def get_samples_by_flow_ID(self, flow_IDs, split='val', as_list=True, as_tuple=False):
        """Get specific samples from the dataset, looking them up by flow ID.
        Used for error analysis purposes (seeing which predicted flows have the lowest/highest EPE, for instance).
        If sampling from the training/validation set, there is a label; otherwise, there isn't.
        Doesn't return a valid np array if all the image pairs don't have the same size (use as_list=True instead).
        Args:
            idx: Specific list of flow IDs to pull from the dataset split
            split: 'train','val','val_with_preds','val_with_pred_paths','test','test_with_preds','test_with_pred_paths'
            as_list: Return as list or np array?
            return_IDs: If True, also return ID of the sample
            as_tuple: If True, return image pairs as tuples; otherwise, return them as np arrays in [2, H, W, 3] format
        In training and validation mode, returns:
            images: Batch of image pairs in format [num_samples, 2, H, W, 3] or list([2, H, W, 3])
            labels: Batch of optical flows in format [num_samples, H, W, 2]
        In testing mode, returns:
            images: Batch of image pairs in format [num_samples, 2, H, W, 3]
            output_files: List of output file names that match the input file names
        """
        # Which split of IDs should we look into?
        if split == 'train':
            IDs_to_search = self._trn_IDs_simpl
        elif split in ['val', 'val_with_preds', 'val_with_pred_paths']:
            IDs_to_search = self._val_IDs_simpl
        elif split in ['test', 'test_with_preds', 'test_with_pred_paths']:
            IDs_to_search = self._tst_IDs_simpl
        else:
            raise ValueError

        # Build the list of indices to look up
        indices = []
        for flow_ID in flow_IDs:
            for idx, test_ID in enumerate(IDs_to_search):
                if flow_ID in test_ID:
                    indices.append(idx)
                    break

        # Return the samples based on their indices
        if len(indices) > 0:
            return self.get_samples(idx=indices, split=split, as_list=as_list, as_tuple=as_tuple)
        else:
            raise ValueError

    ###
    # Various utility functions
    ###
    def simplify_IDs(self, IDs):
        """Simplify list of ID ID string tuples.
        This is dataset specific and needs override.
        Go from ('video_path/frame_0019.png', 'video_path/frame_0020.png', 'video_path/frame_0019.flo/')
        to 'video_path/frames_0019_0020
        Args:
            IDs: List of ID string tuples to simplify
        Returns:
            IDs: Simplified IDs
        """
        raise NotImplementedError

    ###
    # Debug utils
    ###
    def print_config(self):
        """Display configuration values."""
        print("\nDataset Configuration:")
        for k, v in self.opts.items():
            print(f"  {k:20} {v}")
        print(f"  {'mode':20} {self.mode}")
        if self.mode in ['train_noval', 'train_with_val']:
            print(f"  {'train size':20} {self.trn_size}")
        if self.mode in ['train_with_val', 'val']:
            print(f"  {'val size':20} {self.val_size}")
        if self.mode == 'test':
            print(f"  {'test size':20} {self.tst_size}")

    ###
    # tf.data helpers
    ###
    def _train_stub(self, idx):
        """tf.py_func stub for _get_train_samples()
        Args:
            idx: Index of unique sample to return
        Returns:
            x: Image pair in [2, H, W, 3] format
            y: Groundtruth flow in [H, W, 2] format
            ID: string, sample ID
        """
        x, y, ID = self.get_samples(idx=[idx], split='train', as_list=False, simple_IDs=True)
        return np.squeeze(x), np.squeeze(y), ID[0]

    def _val_stub(self, idx):
        """tf.py_func stub for _get_val_samples()
        Args:
            idx: Index of unique sample to return
        Returns:
            x: Image pair in [2, H, W, 3] format
            y: Groundtruth flow in [H, W, 2] format
            path: string, destination path where to save the predicted flow
            ID: string, sample ID
        """
        x, y, path, ID = self.get_samples(idx=[idx], split='val_with_pred_paths', as_list=False, simple_IDs=True)
        return np.squeeze(x), np.squeeze(y), path[0], ID[0]

    def _test_stub(self, idx):
        """tf.py_func stub for _get_test_samples()
        Args:
            idx: Index of unique sample to return
        Returns:
            x: Image pair in [2, H, W, 3] format
            path: string, destination path where to save the predicted flow
            ID: string, sample ID
        """
        x, path, ID = self.get_samples(idx=[idx], split='test_with_pred_paths', as_list=False, simple_IDs=True)
        return np.squeeze(x), path[0], ID[0]

    def get_tf_ds(self, batch_size=1, num_gpus=1, split='train', sess=None):
        """Get a tf.data.Dataset "view" or the dataset
        Args:
            batch_size: Size of the batch
            split: 'train', 'val', 'val_with_preds', 'val_with_pred_paths', 'test', or 'test_with_pred_paths'
        In training and validation mode, returns:
            images: Batch of image pairs in format [N, 2, H, W, 3]
            labels: Batch of optical flows in format [N, H, W, 2], or file paths to predicted label
        In testing mode, returns:
            images: Batch of RGB images in format [N, 2, H, W, 3]
            pred_folder: List of output folders where to save the predicted instance masks
        Refs:
            - Modules: tf.data and tf.contrib.data
            https://www.tensorflow.org/api_docs/python/tf/data
            https://www.tensorflow.org/api_docs/python/tf/contrib/data
            - Also:
            ttps://cs230-stanford.github.io/tensorflow-input-data.html
            http://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/
            https://github.com/kratzert/finetune_alexnet_with_tensorflow/blob/master/datagenerator.py
            https://jhui.github.io/2017/11/21/TensorFlow-Importing-data/
            https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_DataManagement/tensorflow_dataset_api.py
            https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
        """
        assert(split in ['train', 'val', 'test'])
        threads = min(os.cpu_count(), 12)  # os.cpu_count() returns 20 on SERVERP

        # Use the train/val/test indices as the elements of the tf.data.Dataset
        if split == 'train':
            assert(self.mode in ['train_noval', 'train_with_val'])
            tf_ds = tf.data.Dataset.from_tensor_slices(self._trn_idx)
            tf_ds = tf_ds.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=len(self._trn_idx), count=-1))
            tf_ds = tf_ds.apply(tf.contrib.data.map_and_batch(
                map_func=lambda idx: tf.py_func(self._train_stub, [idx], [tf.uint8, tf.float32, tf.string]),
                batch_size=batch_size * num_gpus, num_parallel_batches=threads))

        elif split == 'val':
            assert(self.mode in ['val', 'val_notrain', 'train_noval', 'train_with_val'])
            tf_ds = tf.data.Dataset.from_tensor_slices(self._val_idx)
            tf_ds = tf_ds.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=len(self._val_idx), count=-1))
            tf_ds = tf_ds.apply(tf.contrib.data.map_and_batch(
                map_func=lambda idx: tf.py_func(self._val_stub, [idx], [tf.uint8, tf.float32, tf.string, tf.string]),
                batch_size=batch_size * num_gpus, num_parallel_batches=threads))

        else:  # if split == 'test':
            tf_ds = tf.data.Dataset.from_tensor_slices(self._tst_idx)
            tf_ds = tf_ds.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=len(self._tst_idx), count=-1))
            tf_ds = tf_ds.apply(tf.contrib.data.map_and_batch(
                map_func=lambda idx: tf.py_func(self._test_stub, [idx], [tf.uint8, tf.string, tf.string]),
                batch_size=batch_size * num_gpus, num_parallel_batches=threads))

        # Return tf dataset
        return tf_ds

    ###
    # To look at later (TFRecords support):
    ###
    # https://github.com/sampepose/flownet2-tf/blob/master/src/dataloader.py for both TFRecords support and aug
    # https://github.com/fperazzi/davis-2017/blob/master/python/lib/davis/dataset/base.py
    # https://github.com/fperazzi/davis-2017/blob/master/python/lib/davis/dataset/loader.py
    # https://github.com/kwotsin/create_tfrecords
    # https://kwotsin.github.io/tech/2017/01/29/tfrecords.html
    # http://yeephycho.github.io/2016/08/15/image-data-in-tensorflow/
    # E:\repos\models-master\research\inception\inception\data\build_imagenet_data.py
    # E:\repos\models-master\research\object_detection\dataset_tools\create_kitti_tf_record.py
    # https://github.com/ferreirafabio/video2tfrecords/blob/master/video2tfrecords.py
    # http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
    # https://github.com/linchuming/ImageSR-Tensorflow/blob/master/data_loader.py
    ###
    def _load_from_tfrecords(self):
        pass

    def _write_to_tfrecords(self):
        pass
