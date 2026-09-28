from mpi4py import MPI
import h5py
from torch.utils.data import DataLoader, Dataset
import numpy as np


def _open_h5(path):
    """打开 h5：优先使用 h5py 的 MPI 驱动（原仓库写法），不可用时回退到串行驱动。

    这样在没有安装 MPI 版 h5py 的机器上也能直接跑单卡训练，无需手工改动本文件里的两处调用。
    """
    try:
        return h5py.File(path, 'r', driver='mpio', comm=MPI.COMM_SELF)
    except Exception as e:  # 驱动缺失/不可用时可能抛 ValueError/KeyError/OSError
        print(f"[turb_datasets] 警告: 无法使用 mpio 驱动（{e!r}），已回退到串行 h5py 读取；"
              f"若要多进程训练请安装 MPI 版 h5py。", flush=True)
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

    with _open_h5(dataset_path) as f:  # 优先 mpio，不可用则自动回退串行
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

        with _open_h5(self.dataset_path) as f:  # 优先 mpio，不可用则自动回退串行
            data = f[self.dataset_name][idx].astype(np.float32)
            data = np.moveaxis(data, -1, 0)

            out_dict = {}
            if self.class_cond:
                raise NotImplementedError()
                out_dict["y"] = f[self.dataset_name + '_y'][idx]

        return data, out_dict
