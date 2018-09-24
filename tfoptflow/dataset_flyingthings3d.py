"""
dataset_flyingthings3d.py

FlyingThnigs3D (540x960) and FlyingThinbs3DHalfRes (270x480) optical flow dataset class.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""

from __future__ import absolute_import, division, print_function
import os
import warnings
from tqdm import tqdm
from skimage.io import imread, imsave
import cv2
from sklearn.model_selection import train_test_split

from dataset_base import OpticalFlowDataset, _DATASET_ROOT, _DEFAULT_DS_TRAIN_OPTIONS
from optflow import flow_read, flow_write

_FLYINGTHINGS3D_ROOT = '//naspro/devt/datasets/FlyingThings3D'
_FLYINGTHINGS3DHALFRES_ROOT = _DATASET_ROOT + 'FlyingThings3D_HalfRes'
_FLYINGTHINGS3DHALFRES_ROOTS = (_FLYINGTHINGS3D_ROOT, _FLYINGTHINGS3DHALFRES_ROOT)


class FlyingThings3DDataset(OpticalFlowDataset):
    """FlyingThings3D optical flow dataset.
    """

    def __init__(self, mode='train_with_val', ds_root=_FLYINGTHINGS3D_ROOT, options=_DEFAULT_DS_TRAIN_OPTIONS):
        """Initialize the FlyingThings3DDataset object
        Args:
            mode: Possible options: 'train_noval', 'val', 'train_with_val' or 'test'
            ds_root: Path to the root of the dataset
            options: see base class documentation
        """
        self.min_flow = 0.
        self.avg_flow = 37.99934768676758
        self.max_flow = 1714.8948974609375
        super().__init__(mode, ds_root, options)
        if 'type' not in self.opts:
            self.opts['type'] = 'into_future'

    def set_folders(self):
        """Set the train, val, test, label and prediction label folders.
        Overriden by each dataset. Called by the base class on init.
        Sample results:
            self._trn_dir          = 'E:/datasets/FlyingThings3D/frames_cleanpass/TRAIN'
            self._trn_lbl_dir      = 'E:/datasets/FlyingThings3D/optical_flow/TRAIN'
            self._val_dir          = 'E:/datasets/FlyingThings3D/frames_cleanpass/TRAIN'
            self._val_lbl_dir      = 'E:/datasets/FlyingThings3D/optical_flow/TRAIN'
            self._val_pred_lbl_dir = 'E:/datasets/FlyingThings3D/flow_pred'
            self._tst_dir          = 'E:/datasets/FlyingThings3D/frames_cleanpass/TEST'
            self._tst_pred_lbl_dir = 'E:/datasets/FlyingThings3D/flow_pred'
        """
        self._trn_dir = f"{self._ds_root}/frames_cleanpass/TRAIN"
        self._val_dir = self._trn_dir
        self._tst_dir = f"{self._ds_root}/frames_cleanpass/TEST"

        self._trn_lbl_dir = f"{self._ds_root}/optical_flow/TRAIN"
        self._val_lbl_dir = self._trn_lbl_dir
        self._val_pred_lbl_dir = f"{self._ds_root}/flow_pred"
        self._tst_pred_lbl_dir = self._val_pred_lbl_dir

    def _build_ID_sets(self):
        """Build the list of samples and their IDs, split them in the proper datasets.
         Each ID is a tuple.
         For the training/val/test datasets, they look like ('12518_img1.ppm', '12518_img2.ppm', '12518_flow.flo')
         The original dataset has 22872 image pairs. Using FlyingChairs_train_val.txt, the samples are split between
         22232 training samples (97.2%) and 640 validation samples (2.8%).
        """
        # Load the exclusion file, if it is present
        exclusion_file = self._ds_root + '/frames_cleanpass/all_unused_files.txt'
        exclusion_list = None
        if os.path.exists(exclusion_file):
            with open(exclusion_file, 'r') as f:
                exclusion_list = f.readlines()
                exclusion_list = [line.rstrip() for line in exclusion_list]

        # Search the train folder for the samples, create string IDs for them
        self._IDs = []
        for subset in sorted(os.listdir(self._trn_dir)):  # subset: 'A'
            scenes = sorted(os.listdir(f"{self._trn_dir}/{subset}"))
            for scene in scenes:  # scene: '0000'
                frames = sorted(os.listdir(f"{self._trn_dir}/{subset}/{scene}/left"))
                for idx in range(len(frames) - 1):
                    frame1_ID = f'{subset}/{scene}/left/{frames[idx]}'
                    frame2_ID = f'{subset}/{scene}/left/{frames[idx+1]}'
                    frame = frames[idx].split('.')[0]
                    flow_ID = f"{subset}/{scene}/{self.opts['type']}/left/OpticalFlowIntoFuture_{frame}_L.pfm"
                    if exclusion_list is None or f"TRAIN/{frame1_ID}" not in exclusion_list:
                        self._IDs.append((frame1_ID, frame2_ID, flow_ID))

        # Build the train/val datasets
        if self.opts['val_split'] > 0.:
            self._trn_IDs, self._val_IDs = train_test_split(self._IDs, test_size=self.opts['val_split'],
                                                            random_state=self.opts['random_seed'])
        else:
            self._trn_IDs, self._val_IDs = self._IDs, None

        # Build the test dataset
        self._tst_IDs = []
        for subset in sorted(os.listdir(self._tst_dir)):  # subset: 'A'
            scenes = sorted(os.listdir(f"{self._tst_dir}/{subset}"))
            for scene in scenes:  # scene: '0000'
                frames = sorted(os.listdir(f"{self._tst_dir}/{subset}/{scene}/left"))
                for idx in range(len(frames) - 1):
                    frame1_ID = f'{subset}/{scene}/left/{frames[idx]}'
                    frame2_ID = f'{subset}/{scene}/left/{frames[idx+1]}'
                    frame = frames[idx].split('.')[0]
                    flow_ID = f"{subset}/{scene}/{self.opts['type']}/left/OpticalFlowIntoFuture_{frame}_L.flo"
                    if exclusion_list is None or f"TEST/{frame1_ID}" not in exclusion_list:
                        self._tst_IDs.append((frame1_ID, frame2_ID, flow_ID))

        # Build a list of simplified IDs for Tensorboard logging
        self._trn_IDs_simpl = self.simplify_IDs(self._trn_IDs)
        self._val_IDs_simpl = self.simplify_IDs(self._val_IDs)
        self._tst_IDs_simpl = self.simplify_IDs(self._tst_IDs)

    def simplify_IDs(self, IDs):
        """Simplify list of ID ID string tuples.
        Go from ('A/0005/left/0006.png', 'A/0005/left/0007.png', 'A/0005/left/0006.flo')
        to 'A_0005_left_0006_0007'
        Args:
            IDs: List of ID string tuples to simplify
        Returns:
            IDs: Simplified IDs
        """
        simple_IDs = [ID[0].replace('/', '_').split('.')[0] + '_' + ID[1][-8:-4] for ID in IDs]
        return simple_IDs


class FlyingThings3DHalfResDataset(FlyingThings3DDataset):
    """FlyingThings3D half-res optical flow dataset.
    """

    def __init__(self, mode='train_with_val', ds_root=_FLYINGTHINGS3DHALFRES_ROOTS, options=_DEFAULT_DS_TRAIN_OPTIONS):
        """Initialize the FlyingThings3DDataset object
        Args:
            mode: Possible options: 'train_noval', 'val', 'train_with_val' or 'test'
            ds_root: Path to the root of the dataset
            options: see base class documentation
        """
        # Generate the half-res version of the dataset, if it doesn't already exit
        if isinstance(ds_root, list) or isinstance(ds_root, tuple):
            self.fullres_root, self.halfres_root = ds_root
            if os.path.exists(self.halfres_root):
                self.generate_files = False
                ds_root = self.halfres_root
            else:
                self.generate_files = True
                ds_root = self.halfres_root
                # ds_root = self.fullres_root
        super().__init__(mode, ds_root, options)
        self.min_flow = 0.
        self.avg_flow = 18.954660415649414
        self.max_flow = 856.9017944335938
        if 'type' not in self.opts:
            self.opts['type'] = 'into_future'

    def set_folders(self):
        """Set the train, val, test, label and prediction label folders.
        Overriden by each dataset. Called by the base class on init.
        Sample results:
            self._trn_dir          = 'E:/datasets/FlyingThings3D_HalfRes/frames_cleanpass/TRAIN'
            self._trn_lbl_dir      = 'E:/datasets/FlyingThings3D_HalfRes/optical_flow/TRAIN'
            self._val_dir          = 'E:/datasets/FlyingThings3D_HalfRes/frames_cleanpass/TRAIN'
            self._val_lbl_dir      = 'E:/datasets/FlyingThings3D_HalfRes/optical_flow/TRAIN'
            self._val_pred_lbl_dir = 'E:/datasets/FlyingThings3D_HalfRes/flow_pred'
            self._tst_dir          = 'E:/datasets/FlyingThings3D_HalfRes/frames_cleanpass/TEST'
            self._tst_pred_lbl_dir = 'E:/datasets/FlyingThings3D_HalfRes/flow_pred'
        """
        self._trn_dir = f"{self._ds_root}/frames_cleanpass/TRAIN"
        self._val_dir = self._trn_dir
        self._tst_dir = f"{self._ds_root}/frames_cleanpass/TEST"

        self._trn_lbl_dir = f"{self._ds_root}/optical_flow/TRAIN"
        self._val_lbl_dir = self._trn_lbl_dir
        self._val_pred_lbl_dir = f"{self._ds_root}/flow_pred"
        self._tst_pred_lbl_dir = self._val_pred_lbl_dir

    def _build_ID_sets(self):
        """Build the list of samples and their IDs, split them in the proper datasets.
         Each ID is a tuple that looks like ('12518_img1.ppm', '12518_img2.ppm', '12518_flow.flo')
        """
        # Load the exclusion file, if it is present
        if self.generate_files is True:
            ds_root = self.fullres_root
            trn_dir = self._trn_dir.replace(self.halfres_root, self.fullres_root)
            trn_lbl_dir = self._trn_lbl_dir.replace(self.halfres_root, self.fullres_root)
            tst_dir = self._tst_dir.replace(self.halfres_root, self.fullres_root)
        else:
            ds_root = self.halfres_root
            trn_dir = self._trn_dir
            trn_lbl_dir = self._trn_lbl_dir
            tst_dir = self._tst_dir

        # Load the exclusion file, if it is present
        exclusion_file = ds_root + '/frames_cleanpass/all_unused_files.txt'
        exclusion_list = None
        if os.path.exists(exclusion_file):
            with open(exclusion_file, 'r') as f:
                exclusion_list = f.readlines()
                exclusion_list = [line.rstrip() for line in exclusion_list]

        # Search the train folder for the samples, create string IDs for them
        self._IDs = []
        for subset in sorted(os.listdir(trn_dir)):  # subset: 'A'
            scenes = sorted(os.listdir(f"{trn_dir}/{subset}"))
            for scene in scenes:  # scene: '0000'
                frames = sorted(os.listdir(f"{trn_dir}/{subset}/{scene}/left"))
                for idx in range(len(frames) - 1):
                    frame1_ID = f'{subset}/{scene}/left/{frames[idx]}'
                    frame2_ID = f'{subset}/{scene}/left/{frames[idx+1]}'
                    frame = frames[idx].split('.')[0]
                    flow_ID = f"{subset}/{scene}/{self.opts['type']}/left/OpticalFlowIntoFuture_{frame}_L.pfm"
                    if exclusion_list is None or f"TRAIN/{frame1_ID}" not in exclusion_list:
                        self._IDs.append((frame1_ID, frame2_ID, flow_ID))

        # Create half-res version of the train/val dataset, if it doesn't exist yet
        if self.generate_files is True:
            if not os.path.exists(self.halfres_root):
                os.makedirs(self.halfres_root)
            with tqdm(total=len(self._IDs), desc="Downsampling train images & flows", ascii=True, ncols=100) as pbar:
                for n, ID in enumerate(self._IDs):
                    pbar.update(1)
                    fullres_path = f'{trn_lbl_dir}/{ID[2]}'
                    halfres_path = fullres_path.replace(self.fullres_root, self.halfres_root).replace('.pfm', '.flo')
                    if not os.path.exists(halfres_path):
                        flow = flow_read(fullres_path)
                        flow = cv2.resize(flow, (int(flow.shape[1] * 0.5), int(flow.shape[0] * 0.5))) * 0.5
                        flow_write(flow, halfres_path)
                    for frame_ID in [ID[0], ID[1]]:
                        frame_path_fullres = f'{trn_dir}/{frame_ID}'
                        frame_path_halfres = frame_path_fullres.replace(self.fullres_root, self.halfres_root)
                        if not os.path.exists(frame_path_halfres):
                            frame = imread(frame_path_fullres)
                            frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                folder = os.path.dirname(frame_path_halfres)
                                if not os.path.exists(folder):
                                    os.makedirs(folder)
                                imsave(frame_path_halfres, frame)
                    self._IDs[n] = (ID[0], ID[1], ID[2].replace('.pfm', '.flo'))

        # Build the train/val datasets
        if self.opts['val_split'] > 0.:
            self._trn_IDs, self._val_IDs = train_test_split(self._IDs, test_size=self.opts['val_split'],
                                                            random_state=self.opts['random_seed'])
        else:
            self._trn_IDs, self._val_IDs = self._IDs, None

        # Build the test dataset
        self._tst_IDs = []
        for subset in sorted(os.listdir(tst_dir)):  # subset: 'A'
            scenes = sorted(os.listdir(f"{tst_dir}/{subset}"))
            for scene in scenes:  # scene: '0000'
                frames = sorted(os.listdir(f"{tst_dir}/{subset}/{scene}/left"))
                for idx in range(len(frames) - 1):
                    frame1_ID = f'{subset}/{scene}/left/{frames[idx]}'
                    frame2_ID = f'{subset}/{scene}/left/{frames[idx+1]}'
                    frame = frames[idx].split('.')[0]
                    flow_ID = f"{subset}/{scene}/{self.opts['type']}/left/OpticalFlowIntoFuture_{frame}_L.flo"
                    if exclusion_list is None or f"TEST/{frame1_ID}" not in exclusion_list:
                        self._tst_IDs.append((frame1_ID, frame2_ID, flow_ID))

        # Create half-res version of the test dataset, if it doesn't exist yet
        if self.generate_files is True:
            with tqdm(total=len(self._tst_IDs), desc="Downsampling test images", ascii=True, ncols=100) as pbar:
                for ID in self._tst_IDs:
                    pbar.update(1)
                    for frame_ID in [ID[0], ID[1]]:
                        frame_path_fullres = f'{tst_dir}/{frame_ID}'
                        frame_path_halfres = frame_path_fullres.replace(self.fullres_root, self.halfres_root)
                        if not os.path.exists(frame_path_halfres):
                            frame = imread(frame_path_fullres)
                            frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                folder = os.path.dirname(frame_path_halfres)
                                if not os.path.exists(folder):
                                    os.makedirs(folder)
                                imsave(frame_path_halfres, frame)

        # Build a list of simplified IDs for Tensorboard logging
        self._trn_IDs_simpl = self.simplify_IDs(self._trn_IDs)
        self._val_IDs_simpl = self.simplify_IDs(self._val_IDs)
        self._tst_IDs_simpl = self.simplify_IDs(self._tst_IDs)
