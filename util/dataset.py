# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# pylint: disable=no-member

from typing import Dict, Union

import os
from glob import glob
import shutil
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torch.utils.data
from torchvision import transforms
from scipy import ndimage
import skimage
import pandas as pd


def load_scan(path):
    """
    Load a CT scan in the format of .h5 file.

    Args:
        path (str): path to the .h5 file.

    Returns:
        (numpy.array): CT scan.
    """
    with h5py.File(path, "r") as f:
        return np.array(f.get("img"))


def show_sequence(sequence):
    """
    Show the sequence of CT slices.

    Args:
        sequence (numpy.array): sequence of CT slices.
    """
    fig = plt.figure()
    ims = []
    for i in range(sequence.shape[0]):
        im = plt.imshow(sequence[i], animated=True, cmap="gray")
        plt.axis("off")
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False)
    plt.show()


def show_two_sequences(sequence1, sequence2):
    """
    Show two sequences of CT slices side by side.

    Args:
        sequence1 (numpy.array): sequence of CT slices.
        sequence2 (numpy.array): sequence of CT slices.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    assert sequence1.shape[0] == sequence2.shape[0]
    ims = []
    for i in range(sequence1.shape[0]):
        im1 = axs[0].imshow(sequence1[i], animated=True, cmap="gray")
        im2 = axs[1].imshow(sequence2[i], animated=True, cmap="gray")
        axs[0].axis("off")
        axs[1].axis("off")
        ims.append([im1, im2])
    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat=False)
    plt.show()


def simple_bodymask(img):
    """
    Simple body mask to remove regions, e.g., bed, that are not related to the patient.

    Args:
        img (numpy.array): CT slice.

    Returns:
        (numpy.array): body mask.
    """
    maskthreshold = -500
    oshape = img.shape
    img = ndimage.zoom(img, 128 / np.asarray(img.shape), order=0)
    bodymask = img > maskthreshold
    bodymask = ndimage.binary_closing(bodymask)
    bodymask = ndimage.binary_fill_holes(bodymask, structure=np.ones((3, 3))).astype(
        int
    )
    bodymask = ndimage.binary_erosion(bodymask, iterations=2)
    bodymask = skimage.measure.label(bodymask.astype(int), connectivity=1)
    regions = skimage.measure.regionprops(bodymask.astype(int))
    if len(regions) > 0:
        max_region = np.argmax(list(map(lambda x: x.area, regions))) + 1
        bodymask = bodymask == max_region
        bodymask = ndimage.binary_dilation(bodymask, iterations=2)
    real_scaling = np.asarray(oshape) / 128
    return ndimage.zoom(bodymask, real_scaling, order=0)


def get_bbox(mask):
    """
    Get the bounding box of the body mask.

    Args:
        mask (numpy.array): body mask (#sequences, height, width)

    Returns:
        (numpy.array): bounding box of the body mask.
    """
    indcs = np.where(mask)
    ymin, ymax = np.min(indcs[1]), np.max(indcs[1])
    xmin, xmax = np.min(indcs[2]), np.max(indcs[2])
    return np.asarray([ymin, ymax, xmin, xmax])


def pad_sequence(sequence):
    """
    Pad the shorter side of the sequence to keep the aspect ratio.

    Args:
        sequence (numpy.array): sequence of CT slices.

    Returns:
        (numpy.array): padded sequence.
    """
    pad_val = sequence.min()
    if sequence.shape[1] > sequence.shape[2]:
        pad = (sequence.shape[1] - sequence.shape[2]) // 2
        sequence = np.pad(
            sequence,
            ((0, 0), (0, 0), (pad, pad)),
            mode="constant",
            constant_values=pad_val,
        )
    elif sequence.shape[1] < sequence.shape[2]:
        pad = (sequence.shape[2] - sequence.shape[1]) // 2
        sequence = np.pad(
            sequence,
            ((0, 0), (pad, pad), (0, 0)),
            mode="constant",
            constant_values=pad_val,
        )
    assert sequence.shape[1] == sequence.shape[2]
    return sequence


def apply_windowing(seq, min_hu=-1350, max_hu=150, mean=0, std=1):
    """
    - Window the image to [min_hu, max_hu] = [-1350, 150]. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7120362/) -- equvalent to w=1500, l=-600.
    - Normalize the image to [0,1]
    - Normalizing the image to zero mean and unit variance (precomputed from the training data for each fold, and saved in img_stats)
    """
    min_hu = np.maximum(min_hu, -1024)  # -1024 should be the minimum HU value
    seq = np.clip(seq, min_hu, max_hu)
    seq = (seq - seq.min()) / (seq.max() - seq.min())

    seq -= mean
    seq /= std
    return seq


def preprocess_sequence(sequence, min_hu=-1350, max_hu=150, mean=0, std=1):
    """
    Simple body mask to remove regions, e.g., bed, that are not related to the patient.
    Windowing
    Normalization

    Args:
        sequence (numpy.array): sequence of CT slices.

    Returns:
        (numpy.array): body mask.
    """
    bodymask = np.zeros(sequence.shape)
    for i in range(sequence.shape[0]):
        bodymask[i] = simple_bodymask(sequence[i])
    bbox = get_bbox(bodymask)  # ymin, ymax, xmin, xmax
    sequence = sequence[:, bbox[0] : bbox[1], bbox[2] : bbox[3]]
    bodymask = bodymask[:, bbox[0] : bbox[1], bbox[2] : bbox[3]]
    # we need to keep the aspect ratio, so we pad the shorter side
    sequence = pad_sequence(sequence)
    bodymask = pad_sequence(bodymask)

    sequence = apply_windowing(sequence, min_hu, max_hu, mean, std)
    res = np.zeros((sequence.shape[0], 256, 256))
    msk = np.zeros((sequence.shape[0], 256, 256), dtype=bool)
    for i in range(sequence.shape[0]):
        res[i] = ndimage.zoom(sequence[i], 256 / np.asarray(sequence[i].shape), order=3)
        msk[i] = ndimage.zoom(bodymask[i], 256 / np.asarray(sequence[i].shape), order=0)
    return res, msk


class ToTensor:
    """
    Convert numpy arrays to PyTorch tensors.

    Args:
        sample (Dict): A dictionary containing the sample data.
        The values should be either numpy arrays or strings.

    Returns:
        Dict: Updated sample with PyTorch tensors instead of numpy arrays.
    """

    def __call__(
        self, sample: Dict[str, Union[np.ndarray, str]]
    ) -> Dict[str, Union[torch.Tensor, str]]:
        for key, value in sample.items():
            if self._is_string(value):
                continue
            sample[key] = self._convert_to_tensor(value)
        return sample

    @staticmethod
    def _is_string(value: Union[np.ndarray, str]) -> bool:
        return isinstance(value, str)

    @staticmethod
    def _convert_to_tensor(value: np.ndarray) -> torch.Tensor:
        if value.ndim in [3, 4]:
            return torch.from_numpy(value[None]).type(torch.FloatTensor)
        return torch.from_numpy(value).type(torch.FloatTensor)


class CTData(torch.utils.data.Dataset):
    """
    CT scan loader. Construct the CT scan loader, then sample sequece of slices from the CT scan. For training and validation, a single sequence is randomly sampled from every CT scan. For testing, multiple sequences are uniformly sampled from every CT scan.
    """

    def __init__(
        self,
        path_to_data_dir,
        path_to_df,
        mode,
        num_slices=16,
        mean=(0),  # FIXME: compute mean and std
        std=(1),
        store_data_to_tmpfs=False,
        n=-1,
    ) -> None:
        """
        Construct the CT scan loader.

        Args:
            path_to_data_dir (string): path to the data directory that contains the CT scans.
            path_to_df (string): path to the csv file that contains the clinical information of the CT scans.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data from the train or val set, and sample one sequence per CT scan.
                For the test mode, the data loader will take data from test set, and sample multiple sequences per CT scan to cover the whole CT scan.
            num_slices (int): number of slices per sequence.
            mean (float): mean of the training set.
            std (float): standard deviation of the training set.
            store_data_to_tmpfs (bool): whether to store the data to tmpfs. We do this before running the training to speed up the training. Make sure you have enough space in tmpfs (RAM).
            n (int): number of CT scans to load. If n is negative, load all the CT scans. For debugging purposes.
        """
        self.num_slices = num_slices
        self.mode = mode
        self.n = n

        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], f"Split {mode} not supported for CT"

        df = pd.read_csv(path_to_df)
        pids = df.loc[df["Imaging ok and thickness < 3 mm"] == 1, "Patient id"].unique()
        # TODO: Fix seed
        np.random.shuffle(pids)
        n_train = int(len(pids) * 0.8)
        n_val = int(len(pids) * 0.1)
        if mode == "train":
            pids = pids[:n_train]
        elif mode == "val":
            pids = pids[n_train : n_train + n_val]
        else:
            pids = pids[n_train + n_val :]
        pids = pids[:n] if n > 0 else pids
        pid_sids = df.loc[
            df["Patient id"].isin(pids) & df["Imaging ok and thickness < 3 mm"] == 1,
            "pid_sid",
        ].values

        if store_data_to_tmpfs:
            self._store_data_to_tmpfs(path_to_data_dir, pid_sids)

        path_to_data_dir = "/dev/shm/ct_data"
        self._paths = []
        for pid in pid_sids:
            path = glob(os.path.join(path_to_data_dir, pid + "*"))
            assert len(path) == 1
            path = path[0]
            self._paths.append(path)
        print(f"{self.mode} set size: {len(pid_sids)}")

        self.transforms = transforms.Compose([ToTensor()])

    def _store_data_to_tmpfs(self, path_to_data_dir, pid_sids):
        """
        Store the data to tmpfs to speed up the training.

        Args:
            path_to_data_dir (string): path to the data directory that contains the CT scans.
            pid_sids (list): list of patient ids.
        """
        print("Storing data to tmpfs...")
        tmpfs_dir = "/dev/shm/ct_data"
        if not os.path.exists(tmpfs_dir):
            os.mkdir(tmpfs_dir)
        for pid_sid in pid_sids:
            path = glob(os.path.join(path_to_data_dir, pid_sid + "*"))
            assert len(path) == 1, f"{pid_sid} not found"
            path = path[0]
            shutil.copy(path, tmpfs_dir)
        print("Done!")

    def __getitem__(self, index):
        """
        Given the CT scan index, return the sequence of slices.

        Args:
            index (int): the CT scan index provided by the pytorch sampler.

        Returns:
            slices (tensor): the slices of sampled from the CT scan. The dimension is `num slices` x `height` x `width`.
        """
        path = self._paths[index]
        scan = load_scan(path)
        if self.mode in ["train", "val"]:
            st_idx = np.random.randint(0, scan.shape[0] - self.num_slices)
            sequence = scan[st_idx : st_idx + self.num_slices]
            sequence, msk = preprocess_sequence(sequence)
            sample = {"sequence": sequence, "mask": msk}
            return self.transforms(sample)

        sequence, msk = [], []
        for i in range(0, scan.shape[0], self.num_slices):
            _seq = scan[i : i + self.num_slices]
            _seq, _msk = preprocess_sequence(_seq)
            sequence.append(_seq)
            msk.append(_msk)
        if sequence[-1].shape[0] != self.num_slices:
            sequence[-1] = np.pad(
                sequence[-1],
                ((0, self.num_slices - len(sequence[-1])), (0, 0), (0, 0)),
                constant_values=sequence[-1].min(),
            )
            msk[-1] = np.pad(
                msk[-1],
                ((0, self.num_slices - len(msk[-1])), (0, 0), (0, 0)),
                constant_values=msk[-1].min(),
            )
        sequence = np.asarray(sequence)
        msk = np.asarray(msk)
        sample = {"sequence": sequence, "mask": msk}
        return self.transforms(sample)

    def __len__(self):
        return len(self._paths)
