import pdb

from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader, get_worker_info

# from utils import crop_obs
import os
import pathlib
import numpy as np
import glob
import torch


class EnvIterDataset(IterableDataset):
    def __init__(self, data_dir, train_steps, batch_length, seed=0):
        self.data_dir = data_dir
        self.batch_length = batch_length
        self.train_steps = train_steps
        self.seed = seed

    def load_episodes(self, balance=False):
        directory = pathlib.Path(self.data_dir).expanduser()
        worker_info = get_worker_info()
        if worker_info is None:
            # we are in the main process
            seed = 0
        else:
            seed = worker_info.seed

        random = np.random.RandomState((self.seed + seed) % (1 << 32))
        cache = {}
        while True:
            for filename in directory.glob("*.npz"):
                if filename not in cache:
                    cache[filename] = filename

            keys = list(cache.keys())
            indices = random.choice(len(keys), self.train_steps)
            # print(f'indices: {indices}')
            for index in indices:
                filename = cache[keys[index]]

                try:
                    with open(filename, "rb") as f:
                        episode = np.load(f)
                        episode = {k: episode[k] for k in episode.keys()}
                except Exception as e:
                    print(f"Could not load episode: {e}")
                    continue

                if self.batch_length:
                    total = len(next(iter(episode.values())))
                    available = total - self.batch_length
                    if available >= 0:
                        if balance:
                            index = min(random.randint(0, total), available)
                        else:
                            index = int(random.randint(0, available + 1))
                            # index = available
                        episode = {
                            k: v[index : index + self.batch_length]
                            for k, v in episode.items()
                        }
                        episode['pad_mask'] = np.zeros((self.batch_length,), dtype=np.float32)
                    else:
                        episode = self.pad_episode(episode)

                yield episode

    def pad_episode(self, episode):
        total = len(next(iter(episode.values())))
        pad_length = self.batch_length - total
        assert pad_length > 0
        episode = {k: self._prepend(v, pad_length) for k, v in episode.items()}
        episode['pad_mask'] = np.zeros((self.batch_length,), dtype=np.float32)
        episode['pad_mask'][:pad_length] = True
        return episode

    @staticmethod
    def _prepend(x, length):
        shape = list(x.shape)
        shape[0] = length
        prefix = np.zeros(shape, dtype=x.dtype)
        return np.concatenate([prefix, x], axis=0)

    def __iter__(self):
        return self.load_episodes()