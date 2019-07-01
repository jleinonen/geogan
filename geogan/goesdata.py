import json

import netCDF4
import numpy as np


class BatchGenerator(object):
    def __init__(self, data_file, errors_file=None, batch_size=32,
        tile_shape=(128,128), random_seed=None):

        self.data_file = data_file
        self.errors_file = errors_file
        self.batch_size = batch_size
        self.tile_shape = tile_shape
        self.img_shape = tile_shape # for compatibility

        self.ds = netCDF4.Dataset(data_file, 'r')
        self.N = self.ds["image"].shape[0]
        self.n_channels = self.ds["image"].shape[-1]
        self.image_shape = self.ds["image"].shape[1:3]
        self.timestamps = [str(ts) for ts in 
            netCDF4.chartostring(self.ds["timestamp"][:])]

        with open(errors_file) as f:
            self.errors = json.load(f)

        self.prng = np.random.RandomState(seed=random_seed)

    def __del__(self):
        if "ds" in self.__dict__:
            self.ds.close()

    def __iter__(self):
        return self

    def __next__(self):
        batch_shape = (self.batch_size,) + self.tile_shape + \
            (self.n_channels,)
        batch = np.zeros(batch_shape, dtype=np.float32)

        for k in range(self.batch_size):
            tile = self.sample_tile()
            tile = tile.astype(np.float32)
            batch[k,...] = tile

        batch /= 127.5
        batch -= 1

        return batch

    def sample_tile(self):
        tile_errors = True
        while tile_errors:
            k = self.prng.randint(self.N)
            i0 = self.prng.randint(self.image_shape[0]-self.tile_shape[0])
            i1 = i0+self.tile_shape[0]
            j0 = self.prng.randint(self.image_shape[1]-self.tile_shape[1])
            j1 = j0+self.tile_shape[1]
            timestamp = self.timestamps[k]
            
            if timestamp in self.errors:
                rect = (i0,i1,j0,j1)
                tile_errors = any(rects_overlap(rect, r) for r
                    in self.errors[timestamp])
            else:
                tile_errors = False

        tile = np.array(self.ds["image"][k,i0:i1,j0:j1,:], copy=False)

        return tile


def rects_overlap(rect1, rect2):
    (l1, r1, b1, t1) = rect1
    (l2, r2, b2, t2) = rect2
    return (l1 < r2) and (r1 >= l2) and (t1 >= b2) and (b1 < t2)
