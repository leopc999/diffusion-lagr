from mpi4py import MPI
import h5py
from torch.utils.data import DataLoader, Dataset
import numpy as np


def _open_h5(path):
    """Open an HDF5 file: try the h5py MPI driver first (as the released code does),
    and fall back to the serial driver when it is unavailable.

    A host without an MPI-built h5py can therefore train on a single GPU without
    editing the two call sites in this file by hand.
    """
    try:
        return h5py.File(path, 'r', driver='mpio', comm=MPI.COMM_SELF)
    except Exception as e:  # a missing/unsupported driver raises ValueError/KeyError/OSError
        print(f"[turb_datasets] warning: cannot use the mpio driver ({e!r}); falling back "
              f"to serial h5py. Install an MPI-built h5py for multi-process training.", flush=True)
        return h5py.File(path, 'r')


def load_data(
    *,
    dataset_path,
    dataset_name,
    batch_size,
    class_cond=False,
    deterministic=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param dataset_path: a dataset path.
    :param dataset_name: a dataset name.
    :param batch_size: the batch size of each returned pair.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. Not implemented.
    :param deterministic: if True, yield results in a deterministic order.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    with _open_h5(dataset_path) as f:  # mpio first, serial fallback
        len_dataset = f[dataset_name].len()

    chunk_size = len_dataset // size
    start_idx  = rank * chunk_size

    dataset = TurbDataset(
        dataset_path, dataset_name, class_cond, start_idx, chunk_size,
    )

    shuffle = True if deterministic else False
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=True
    )

    while True:
        yield from loader


class TurbDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        dataset_name,
        class_cond,
        start_idx,
        chunk_size,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.class_cond = class_cond
        self.start_idx  = start_idx
        self.chunk_size = chunk_size

    def __len__(self):
        return self.chunk_size

    def __getitem__(self, idx):
        idx += self.start_idx

        with _open_h5(self.dataset_path) as f:  # mpio first, serial fallback
            data = f[self.dataset_name][idx].astype(np.float32)
            data = np.moveaxis(data, -1, 0)

            out_dict = {}
            if self.class_cond:
                raise NotImplementedError()
                out_dict["y"] = f[self.dataset_name + '_y'][idx]

        return data, out_dict
