"""Inspired by from https://github.com/tanelp/tiny-diffusion."""

import os
import random

import numpy as np
import torch
import h5py

from torch.utils.data import TensorDataset

from .project_path import datadir
from .normalizer import Normalizer


def get_seismic_dataset():

    # ============================================================
    # ORIGINAL CSGM SEISMIC DATASET
    # ============================================================

    # # Define data directory
    # data_path = os.path.join(
    #     datadir("training-data"),
    #     "training-pairs.h5"
    # )

    # # Download the dataset into the data directory if it does not exist
    # if not os.path.isfile(data_path):
    #     os.system(
    #         f"wget 'https://www.dropbox.com/scl/fi/0dmnhlxk4jso10gr3oua9/"
    #         f"training-pairs.h5?rlkey=2sdyqgs79jqoc7vjh1qrcnwx7"
    #         f"&st=nni5cf4p&dl=0' "
    #         f"--no-check-certificate -O {data_path}"
    #     )

    # # Load seismic images and create training and testing data
    # file = h5py.File(data_path, 'r')
    # x = torch.from_numpy(file['dm'][...])
    # y = torch.from_numpy(file['rtm'][...])
    # file.close()


    # ============================================================
    # MODIFIED: USE RESOLVABILITY SEISMIC TRAINING DATA
    # ============================================================

    # Path:
    #
    # seismic-devito/
    # ├── csgm/
    # │   └── csgm/utils/toy_dataset.py
    # │
    # └── resolvability/
    #     └── data/seismic/dataset_train.h5

    data_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "resolvability",
            "data",
            "seismic",
            "dataset_train.h5"
        )
    )

    # Make sure the dataset exists.
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Seismic training dataset not found: {data_path}"
        )

    print("Loading seismic dataset from:")
    print(data_path)

    # ------------------------------------------------------------
    # Load the resolvability seismic training data.
    #
    # broadband_dm: (4000, 256, 256)
    # rtm:          (4000, 256, 256)
    #
    # x = target image
    # y = conditioning image
    # ------------------------------------------------------------

    with h5py.File(data_path, "r") as file:
        x = torch.from_numpy(
            file["broadband_dm"][...]
        ).float()

        y = torch.from_numpy(
            file["rtm"][...]
        ).float()

    print("Loaded data:")
    print("  broadband_dm:", x.shape)
    print("  rtm:         ", y.shape)


    # ============================================================
    # ORIGINAL CSGM PREPROCESSING
    # ============================================================

    # Zero out water layer.
    y[..., :10] = 0.0

    # Normalize target seismic images.
    x_normalizer = Normalizer(x)
    x = x_normalizer.normalize(x)

    # Normalize conditioning seismic images.
    y_normalizer = Normalizer(y)
    y = y_normalizer.normalize(y)

    nsamples = x.shape[0]

    # Randomly shuffle samples.
    perm_idxs = torch.randperm(nsamples)

    x = x[perm_idxs, ...]
    y = y[perm_idxs, ...]


    # ============================================================
    # MODIFIED SHAPE HANDLING
    # ============================================================
    #
    # Resolvability data initially:
    #
    # x = (N, 256, 256)
    # y = (N, 256, 256)
    #
    # CSGM training expects:
    #
    # data = (N, 2, 256, 256, 1)
    #
    # Channel 0 -> target
    # Channel 1 -> conditioning image
    # Last dimension -> feature/channel dimension used by FNO
    #

    x = x.unsqueeze(1).unsqueeze(-1)
    y = y.unsqueeze(1).unsqueeze(-1)

    # x: (N, 1, 256, 256, 1)
    # y: (N, 1, 256, 256, 1)

    data = torch.cat((x, y), dim=1)

    # data: (N, 2, 256, 256, 1)

    print("CSGM data shape:")
    print(" ", data.shape)


    # ============================================================
    # ORIGINAL CSGM 90/10 TRAIN/VALIDATION SPLIT
    # ============================================================

    ntrain = nsamples // 10 * 9

    print("Training samples:  ", ntrain)
    print("Validation samples:", nsamples - ntrain)

    return (
        TensorDataset(data[:ntrain, ...]),
        TensorDataset(data[ntrain:, ...]),
        x_normalizer,
        y_normalizer
    )


def find_replace_closest_number(some_set, k):
    closest_num = min(some_set, key=lambda x: abs(x - k))
    some_set.remove(closest_num)
    some_set.add(k)
    return some_set


def optimal_jittered_sampling(interval, num_samples):
    # Calculate the subinterval size
    subinterval_size = (interval[1] - interval[0]) / num_samples

    samples = []

    for i in range(num_samples):

        # Calculate the center of the current subinterval
        center = interval[0] + (i + 0.5) * subinterval_size

        # Add a random jitter within the subinterval
        jitter = random.uniform(
            -subinterval_size / 2,
            subinterval_size / 2
        )

        sample = center + jitter

        # Ensure the sample is within the interval bounds
        sample = max(
            interval[0],
            min(interval[1], sample)
        )

        samples.append(sample)

    return np.array(samples).astype(np.float32)


def quadratic(n=200,
              s=15,
              x_range=(-3, 3),
              eval_pattern='jitter',
              phase='train',
              device='cpu'):

    """Creat quadratic toy dataset of pairs of coordinates and function values.

    This toy dataset is obtained from:
    https://arxiv.org/pdf/2209.14125.pdf.

    Args:
        n: Number of data points.
        s: Maximum number of points at which functions is evaluated.
        d: Dimension of the input space.
        x_range: Range of the input space.
        eval_pattern: Whether to evaluate the function on the same
        coordinates ('same') or random coordinates with the same size.
        Default is 'same'.

    Returns:
        A list of n data points. Each data point is a tuple of two arrays
        with the first array being the coordinates and the second array
        being the function values.
    """

    noise_dist = torch.distributions.gamma.Gamma(1.0, 2.5)

    a_choices = torch.tensor(
        [-1.0, 1.0],
        device=device
    )

    if eval_pattern == 'same':

        x = np.linspace(
            *x_range,
            s
        )

    elif eval_pattern == 'jitter':

        x = optimal_jittered_sampling(
            x_range,
            s
        )

    if phase != 'train':

        x = set(x)

        for val in [-1.0, 0.0, 0.5]:
            find_replace_closest_number(
                x,
                val
            )

        x = list(x)

        x.sort()

        x = np.array(x)

    x = torch.from_numpy(
        np.array(x).astype(np.float32)
    ).reshape(
        1,
        1,
        -1
    ).repeat(
        n,
        1,
        1
    ).to(device)

    a = a_choices[
        torch.randint(
            0,
            a_choices.size(0),
            (n,),
            device=device
        )
    ].reshape(
        -1,
        1,
        1
    )

    eps = noise_dist.sample(
        (n, 1)
    ).reshape(
        -1,
        1,
        1
    ).to(device)

    y = a * x**2 + eps

    return torch.cat(
        (y, x),
        dim=1
    )


def get_seismic_eval_dataset():
    """
    Load the independent seismic evaluation dataset.

    IMPORTANT:
    Normalization statistics are computed from dataset_train.h5,
    because the CSGM was trained using those statistics.
    The eval dataset is never used to fit the normalizers.
    """

    # Paths to the training and independent evaluation datasets
    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..",
            "resolvability", "data", "seismic"
        )
    )

    train_path = os.path.join(base_dir, "dataset_train.h5")
    eval_path = os.path.join(base_dir, "dataset_eval.h5")

    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    if not os.path.isfile(eval_path):
        raise FileNotFoundError(f"Evaluation dataset not found: {eval_path}")

    print("Loading normalization statistics from:")
    print(train_path)

    # ---------------------------------------------------------
    # Fit normalizers ONLY using the training dataset
    # ---------------------------------------------------------
    with h5py.File(train_path, "r") as file:
        x_train = torch.from_numpy(
            file["broadband_dm"][...]
        ).float()

        y_train = torch.from_numpy(
            file["rtm"][...]
        ).float()

    # Match the preprocessing used during training
    y_train[..., :10] = 0.0

    x_normalizer = Normalizer(x_train)
    y_normalizer = Normalizer(y_train)

    # ---------------------------------------------------------
    # Load independent evaluation dataset
    # ---------------------------------------------------------
    print("Loading independent evaluation dataset from:")
    print(eval_path)

    with h5py.File(eval_path, "r") as file:
        x_eval = torch.from_numpy(
            file["broadband_dm"][...]
        ).float()

        y_eval = torch.from_numpy(
            file["rtm"][...]
        ).float()

    y_eval[..., :10] = 0.0

    print("Loaded evaluation data:")
    print("  broadband_dm:", x_eval.shape)
    print("  rtm:         ", y_eval.shape)

    # Normalize eval data using TRAIN statistics
    x_eval = x_normalizer.normalize(x_eval)
    y_eval = y_normalizer.normalize(y_eval)

    # Shape expected by CSGM:
    # (N, 2, 256, 256, 1)
    x_eval = x_eval.unsqueeze(1).unsqueeze(-1)
    y_eval = y_eval.unsqueeze(1).unsqueeze(-1)

    data_eval = torch.cat((x_eval, y_eval), dim=1)

    print("CSGM evaluation data shape:")
    print(" ", data_eval.shape)

    return TensorDataset(data_eval), x_normalizer, y_normalizer