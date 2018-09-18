"""
dataset_kitti.py

KITTI optical flow dataset class.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""

from __future__ import absolute_import, division, print_function
import os
import numpy as np
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split

from dataset_base import OpticalFlowDataset, _DATASET_ROOT, _DEFAULT_DS_TRAIN_OPTIONS
from optflow import flow_read

_KITTI2012_ROOT = _DATASET_ROOT + 'KITTI12'
_KITTI2015_ROOT = _DATASET_ROOT + 'KITTI15'
_KITTI_ROOT = _KITTI2015_ROOT


class KITTIDataset(OpticalFlowDataset):
    """KITTI optical flow dataset.
    List of adapted KITTI12 sizes: [(1, 375, 1242, 2), (1, 370, 1226, 2), (1, 376, 1241, 2), (1, 374, 1238, 2)]
    List of adapted KITTI15 sizes: [(1, 375, 1242, 2), (1, 370, 1224, 2), (1, 376, 1241, 2), (1, 374, 1238, 2)]
    """

    def __init__(self, mode='train_with_val', ds_root=_KITTI_ROOT, options=_DEFAULT_DS_TRAIN_OPTIONS):
        """Initialize the KITTIDataset object
        Args:
            mode: Possible options: 'train_noval', 'val', 'train_with_val' or 'test'
            ds_root: Path to the root of the dataset
            options: see base class documentation
        Flow stats:
            KITTI2012: training flow mag min=0.0, avg=6.736172669242419, max=232.20108032226562 (194 flows)
            KITTI2015: raining flow mag min=0.0, avg=4.7107220490319, max=256.4881896972656 (200 flows)
        """
        self.min_flow = 0.
        self.avg_flow = 4.7107220490319
        self.max_flow = 256.4881896972656
        super().__init__(mode, ds_root, options)
        assert(self.opts['type'] in ['noc', 'occ'])

    def set_folders(self):
        """Set the train, val, test, label and prediction label folders.
        Overriden by each dataset. Called by the base class on init.
        Sample results:
            self._trn_dir          = 'E:/datasets/KITTI12/training/colored_0'
            self._trn_lbl_dir      = 'E:/datasets/KITTI12/training/flow_occ'
            self._val_dir          = 'E:/datasets/KITTI12/training/colored_0'
            self._val_lbl_dir      = 'E:/datasets/KITTI12/training/flow_occ'
            self._val_pred_lbl_dir = 'E:/datasets/KITTI12/training/flow_occ_pred'
            self._tst_dir          = 'E:/datasets/KITTI12/testing/colored_0'
            self._tst_pred_lbl_dir = 'E:/datasets/KITTI12/testing/flow_occ_pred'
        """
        if os.path.exists(self._ds_root + '/training/colored_0'):
            self._trn_dir = self._ds_root + '/training/colored_0'
            self._tst_dir = self._ds_root + '/testing/colored_0'
        elif os.path.exists(self._ds_root + '/training/image_2'):
            self._trn_dir = self._ds_root + '/training/image_2'
            self._tst_dir = self._ds_root + '/testing/image_2'
        else:
            raise IOError
        self._val_dir = self._trn_dir

        self._trn_lbl_dir = self._ds_root + '/training/flow_' + self.opts['type']
        self._val_lbl_dir = self._trn_lbl_dir
        self._val_pred_lbl_dir = self._ds_root + '/training/flow_' + self.opts['type'] + '_pred'
        self._tst_pred_lbl_dir = self._ds_root + '/testing/flow_' + self.opts['type'] + '_pred'

    def _build_ID_sets(self):
        """Build the list of samples and their IDs, split them in the proper datasets.
        Called by the base class on init.
        Each ID is a tuple.
        For the training/val datasets, they look like ('000065_10.png', '000065_11.png', '000065_10.png')
         -> gt flows are stored as 48-bit PNGs
        For the test dataset, they look like ('000000_10.png', '00000_11.png', '000000_10.flo')
        """
        # Search the train folder for the samples, create string IDs for them
        frames = sorted(os.listdir(self._trn_dir))
        self._IDs, idx = [], 0
        while idx < len(frames) - 1:
            self._IDs.append((frames[idx], frames[idx + 1], frames[idx]))
            idx += 2

        # Build the train/val datasets
        if self.opts['val_split'] > 0.:
            self._trn_IDs, self._val_IDs = train_test_split(self._IDs, test_size=self.opts['val_split'],
                                                            random_state=self.opts['random_seed'])
        else:
            self._trn_IDs, self._val_IDs = self._IDs, None

        # Build the test dataset
        self._tst_IDs, idx = [], 0
        frames = sorted(os.listdir(self._tst_dir))
        while idx < len(frames) - 1:
            flow_ID = frames[idx].replace('.png', '.flo')
            self._tst_IDs.append((frames[idx], frames[idx + 1], flow_ID))
            idx += 2

        self._trn_IDs_simpl = self.simplify_IDs(self._trn_IDs)
        self._val_IDs_simpl = self.simplify_IDs(self._val_IDs)
        self._tst_IDs_simpl = self.simplify_IDs(self._tst_IDs)

    def simplify_IDs(self, IDs):
        """Simplify list of ID string tuples.
        Go from ('000065_10.png', '000065_11.png', '000065_10.png') to 'frames_000065_10_11
        Args:
            IDs: List of ID string tuples to simplify
        Returns:
            IDs: Simplified IDs
        """
        simple_IDs = []
        for ID in IDs:
            simple_IDs.append(f"frames_{ID[0][:-4]}_{ID[1][-6:-4]}")
        return simple_IDs

    def _get_flow_stats(self):
        """Get the min, avg, max flow of the training data according to OpenCV.
        This will allow us to normalize the rendering of flows to images across the entire dataset. Why?
        Because low magnitude flows should appear lighter than high magnitude flows when rendered as images.
        We need to override the base class implementation that assumes images in the dataset all have the same size.
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

        # Get the max width and height and combine the flow magnitudes into one masked array
        h_max = np.max([flow_mag.shape[0] for flow_mag in flow_mags])
        w_max = np.max([flow_mag.shape[1] for flow_mag in flow_mags])
        masked = np.ma.empty((h_max, w_max, len(flow_mags)), dtype=np.float32)
        masked.mask = True
        for idx, flow_mag in enumerate(flow_mags):
            masked[:flow_mag.shape[0], :flow_mag.shape[1], idx] = flow_mag
        self.min_flow, self.avg_flow, self.max_flow = np.min(masked), np.mean(masked), np.max(masked)
        print(
            f"training flow min={self.min_flow}, avg={self.avg_flow}, max={self.max_flow} ({num_flows} flows)")
