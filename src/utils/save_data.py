import pickle
from pathlib import Path
import h5py
import numpy as np
import time


def h5_create_file(folder_path: str, filename: str) -> str:
    """
    Create an empty HDF5 file for experiments.

    Args:
        folder_path (Path): Path to folder.
        filename (str): HDF5 file name.

    Returns:
        Path to created file.
    """
    file_path = Path(folder_path) / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(file_path, "w") as f:
        f.attrs["created_at"] = time.asctime()
    return file_path

def h5_load_file(folder_path: str, filename: str) -> dict:
    """
    Load all data from an HDF5 file into a dictionary.

    Args:
        filename (str): Path to HDF5 file.

    Returns:
        data (dict): Dictionary with all groups and datasets.
    """
    file_path = Path(folder_path) / filename
    data = {}
    with h5py.File(file_path, "r") as f:
        for group_name in f.keys():
            grp = f[group_name]
            data[group_name] = {}
            # Load datasets
            for dset_name, dset in grp.items():
                data[group_name][dset_name] = dset[()]
            # Load attributes
            data[group_name]['attrs'] = dict(grp.attrs)
    return data

def append_experiment(folder_path,data_dict, filename="navier_strokes_2d_cylinder_wake.h5",group_name="test", extra_attrs=None):
    """
    Append a new experiment to HDF5 file as a new group.

    Args:
        folder (Path/str): Path to folder.
        filename (str): HDF5 file name.
        data_dict (dict): {name: np.ndarray} with experiment data.
        extra_attrs (dict, optional): Metadata to store in group.

    Returns:
        group_name (str): Name of the created experiment group.
    """
    file_path = Path(folder_path) / filename
    with h5py.File(file_path, "a") as f:
        # automatic group name
        idx = len(f.keys())
        if group_name in f.keys():
            del f[group_name]
            grp = f.create_group(group_name)
            # grp = f[group_name]

        else:
            grp = f.create_group(group_name)

        # store datasets
        for key, value in data_dict.items():
            grp.create_dataset(key, data=value)

        # store metadata
        if extra_attrs:
            for k, v in extra_attrs.items():
                grp.attrs[k] = v

    return group_name

def save_trajg_to_h5( filename, traj, grid, meta):

    """
    Save Burgers trajectory + grid + metadata into HDF5 as a new experiment.
    """
    with h5py.File(filename, "a") as f:
        idx = len(f.keys())
        group_name = f"experiment_{idx:03d}"
        grp = f.create_group(group_name)

        grp.create_dataset("trajectory", data=traj, compression="lzf", chunks=True)
        grp.create_dataset("grid", data=grid, compression="lzf", chunks=True)

        for k, v in meta.items():
            grp.attrs[k] = v

    return group_name