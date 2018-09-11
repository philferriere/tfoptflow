"""
dataset_flyingchairs.py

FlyingChairs optical flow dataset class.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""

from __future__ import absolute_import, division, print_function
import os, random
from sklearn.model_selection import train_test_split

from dataset_base import OpticalFlowDataset, _DATASET_ROOT, _DEFAULT_DS_TRAIN_OPTIONS

_FLYINGCHAIRS_ROOT = _DATASET_ROOT + 'FlyingChairs_release'

class FlyingChairsDataset(OpticalFlowDataset):
    """FlyingChairs optical flow dataset.
    """
    def __init__(self, mode='train_with_val', ds_root=_FLYINGCHAIRS_ROOT, options=_DEFAULT_DS_TRAIN_OPTIONS):
        """Initialize the FlyingChairsDataset object
        Args:
            mode: Possible options: 'train_noval', 'val', 'train_with_val' or 'test'
            ds_root: Path to the root of the dataset
            options: see base class documentation
        """
        self.min_flow_mag = 0.
        self.avg_flow_mag = 11.113031387329102
        self.max_flow_mag = 300.007568359375
        super().__init__(mode, ds_root, options)

    def set_folders(self):
        """Set the train, val, test, label and prediction label folders.
        Overriden by each dataset. Called by the base class on init.
        Sample results:
            self._train_folder           = 'E:/datasets/FlyingChairs_release/data'
            self._train_label_folder     = 'E:/datasets/FlyingChairs_release/data'
            self._val_folder             = 'E:/datasets/FlyingChairs_release/data'
            self._val_label_folder       = 'E:/datasets/FlyingChairs_release/data'
            self._val_pred_label_folder  = 'E:/datasets/FlyingChairs_release/flow_pred'
            self._test_folder            = 'E:/datasets/FlyingChairs_release/data'
            self._test_pred_label_folder = 'E:/datasets/FlyingChairs_release/flow_pred'
        """
        self._train_folder = f"{self._ds_root}/data"
        self._val_folder = self._train_folder
        self._test_folder = self._train_folder

        self._train_label_folder = self._train_folder
        self._val_label_folder = self._train_label_folder
        self._val_pred_label_folder = f"{self._ds_root}/flow_pred"
        self._test_pred_label_folder = self._val_pred_label_folder

    def set_IDs_filenames(self):
        """Set the names of the train/val/test files that will hold the list of sample/label IDs
        Called by the base class on init.
        Typical ID filenames:
            'E:/datasets/MPI-Sintel/final_train.txt'
            'E:/datasets/MPI-Sintel/final_val.txt'
            'E:/datasets/MPI-Sintel/final_test.txt'
        """
        if os.path.exists(self._ds_root + '/FlyingChairs_train_val.txt'):
            self._train_IDs_file = f"{self._ds_root}/train.txt"
            self._val_IDs_file = f"{self._ds_root}/val.txt"
            self._test_IDs_file = f"{self._ds_root}/test.txt"
        else:
            super().set_IDs_filenames()

    def _build_ID_sets(self):
        """Build the list of samples and their IDs, split them in the proper datasets.
         Each ID is a tuple.
         For the training/val/test datasets, they look like ('12518_img1.ppm', '12518_img2.ppm', '12518_flow.flo')
         The original dataset has 22872 image pairs. Using FlyingChairs_train_val.txt, the samples are split between
         22232 training samples (97.2%) and 640 validation samples (2.8%).
        """
        # Search the train folder for the samples, create string IDs for them
        frames = sorted(os.listdir(self._train_folder))
        self._IDs, idx = [], 0
        while idx < len(frames) - 1:
            self._IDs.append((frames[idx+1], frames[idx+2], frames[idx]))
            idx += 3

        # Build the train/val datasets
        if os.path.exists(self._ds_root + '/FlyingChairs_train_val.txt'):
            with open(self._ds_root + '/FlyingChairs_train_val.txt', 'r') as f:
                train_val_IDs = f.readlines()
            train_val_IDs = list(map(int, train_val_IDs))
            train_indices = [idx for idx, value in enumerate(train_val_IDs) if value == 1]
            self._train_IDs = [self._IDs[idx] for idx in train_indices]
            val_indices = [idx for idx, value in enumerate(train_val_IDs) if value == 2]
            self._val_IDs = [self._IDs[idx] for idx in val_indices]
            random.seed(self.opts['random_seed'])
            random.shuffle(self._train_IDs)
            random.shuffle(self._val_IDs)
        elif self.opts['val_split'] > 0.:
            self._train_IDs, self._val_IDs = train_test_split(self._IDs, test_size=self.opts['val_split'],
                                                                  random_state=self.opts['random_seed'])
        else:
            self._train_IDs, self._val_IDs = self._IDs, None

        # Build the test dataset.
        # Note that we're only using the FlyingChairs dataset to pre-train our network. Since, we don't really need a
        # final unbiased estimate after hyper-param tuning, so we set the test set to the val set.
        self._test_IDs = self._val_IDs.copy()

        # Build a list of simplified IDs for Tensorboard logging
        self._train_IDs_simplified = self.simplify_IDs(self._train_IDs)
        self._val_IDs_simplified = self.simplify_IDs(self._val_IDs)
        self._test_IDs_simplified = self.simplify_IDs(self._test_IDs)

    def simplify_IDs(self, IDs):
        """Simplify list of ID string tuples.
        Go from ('video_path/frame_0019.png', 'video_path/frame_0020.png', 'video_path/frame_0019.flo/')
        to 'video_path/frames_0019_0020
        Args:
            IDs: List of ID string tuples to simplify
        Returns:
            IDs: Simplified IDs
        """
        simple_IDs = [f"pair_{ID[0][0:5]}" for ID in IDs]
        return simple_IDs
