"""
dataset_mpisintel.py

MPI-Sintel (436x1024) optical flow dataset class.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""

from __future__ import absolute_import, division, print_function
import os
from sklearn.model_selection import train_test_split

from dataset_base import OpticalFlowDataset, _DATASET_ROOT, _DEFAULT_DS_TRAIN_OPTIONS

_MPISINTEL_ROOT = _DATASET_ROOT + 'MPI-Sintel'


class MPISintelDataset(OpticalFlowDataset):
    """MPI-Sintel optical flow dataset.
    """

    def __init__(self, mode='train_with_val', ds_root=_MPISINTEL_ROOT, options=_DEFAULT_DS_TRAIN_OPTIONS):
        """Initialize the MPISintelDataset object
        Args:
            mode: Possible options: 'train_noval', 'val', 'train_with_val' or 'test'
            ds_root: Path to the root of the dataset
            options: see base class documentation
        """
        self.min_flow = 0.
        self.avg_flow = 13.495569229125977
        self.max_flow = 455.44061279296875
        super().__init__(mode, ds_root, options)
        assert(self.opts['type'] in ['clean', 'final'])

    def set_folders(self):
        """Set the train, val, test, label and prediction label folders.
        Overriden by each dataset. Called by the base class on init.
        Sample results:
            self._trn_dir          = 'E:/datasets/MPI-Sintel/training/final'
            self._trn_lbl_dir      = 'E:/datasets/MPI-Sintel/training/flow'
            self._val_dir          = 'E:/datasets/MPI-Sintel/training/final'
            self._val_lbl_dir      = 'E:/datasets/MPI-Sintel/training/flow'
            self._val_pred_lbl_dir = 'E:/datasets/MPI-Sintel/training/final_flow_pred'
            self._tst_dir          = 'E:/datasets/MPI-Sintel/test/final'
            self._tst_pred_lbl_dir = 'E:/datasets/MPI-Sintel/test/final_flow_pred'
        """
        self._trn_dir = f"{self._ds_root}/training/{self.opts['type']}"
        self._val_dir = self._trn_dir
        self._tst_dir = f"{self._ds_root}/test/{self.opts['type']}"

        self._trn_lbl_dir = f"{self._ds_root}/training/flow"
        self._val_lbl_dir = self._trn_lbl_dir
        self._val_pred_lbl_dir = f"{self._ds_root}/training/{self.opts['type']}_flow_pred"
        self._tst_pred_lbl_dir = f"{self._ds_root}/test/{self.opts['type']}_flow_pred"

    def set_IDs_filenames(self):
        """Set the names of the train/val/test files that will hold the list of sample/label IDs
        Called by the base class on init.
        Typical ID filenames:
            'E:/datasets/MPI-Sintel/final_train.txt'
            'E:/datasets/MPI-Sintel/final_val.txt'
            'E:/datasets/MPI-Sintel/final_test.txt'
        """
        self._trn_IDs_file = f"{self._ds_root}/{self.opts['type']}_train_{self.opts['val_split']}split.txt"
        self._val_IDs_file = f"{self._ds_root}/{self.opts['type']}_val_{self.opts['val_split']}split.txt"
        self._tst_IDs_file = f"{self._ds_root}/{self.opts['type']}_test.txt"

    def _build_ID_sets(self):
        """Build the list of samples and their IDs, split them in the proper datasets.
        Called by the base class on init.
        Each ID is a tuple.
        For the training/val/test datasets, they look like:
            ('alley_1/frame_0001.png', 'alley_1/frame_0002.png', 'alley_1/frame_0001.flo')
        """
        # Search the train folder for the samples, create string IDs for them
        self._IDs = []
        for video in os.listdir(self._trn_dir):  # video: 'alley_1'
            frames = sorted(os.listdir(self._trn_dir + '/' + video))
            for idx in range(len(frames) - 1):
                frame1_ID = f'{video}/{frames[idx]}'
                frame2_ID = f'{video}/{frames[idx+1]}'
                flow_ID = f'{video}/{frames[idx].replace(".png", ".flo")}'
                self._IDs.append((frame1_ID, frame2_ID, flow_ID))

        # Build the train/val datasets
        if self.opts['val_split'] > 0.:
            self._trn_IDs, self._val_IDs = train_test_split(self._IDs, test_size=self.opts['val_split'],
                                                            random_state=self.opts['random_seed'])
        else:
            self._trn_IDs, self._val_IDs = self._IDs, None

        # Build the test dataset
        self._tst_IDs = []
        for video in os.listdir(self._tst_dir):  # video: 'ambush_1'
            frames = sorted(os.listdir(self._tst_dir + '/' + video))
            for idx in range(len(frames) - 1):
                frame1_ID = f'{video}/{frames[idx]}'
                frame2_ID = f'{video}/{frames[idx+1]}'
                flow_ID = f'{video}/{frames[idx].replace(".png", ".flo")}'
                self._tst_IDs.append((frame1_ID, frame2_ID, flow_ID))

        # Build a list of simplified IDs for Tensorboard logging
        self._trn_IDs_simpl = self.simplify_IDs(self._trn_IDs)
        self._val_IDs_simpl = self.simplify_IDs(self._val_IDs)
        self._tst_IDs_simpl = self.simplify_IDs(self._tst_IDs)

    def simplify_IDs(self, IDs):
        """Simplify list of ID ID string tuples.
        Go from ('video_path/frame_0019.png', 'video_path/frame_0020.png', 'video_path/frame_0019.flo/')
        to 'video_path/frames_0019_0020
        Args:
            IDs: List of ID string tuples to simplify
        Returns:
            IDs: Simplified IDs
        """
        simple_IDs = []
        for ID in IDs:
            pos = ID[0].find('frame_')
            simple_IDs.append(f"{ID[0][:pos]}frames_{ID[0][pos + 6:pos + 10]}_{ID[1][pos + 6:pos + 10]}")
        return simple_IDs
