import numpy as np


class BatchGenerator(object):
    
    def __init__(self, sequences, decoder, batch_size=32, random_seed=None):

        self.batch_size = batch_size
        self.prng = np.random.RandomState(seed=random_seed)
        self.sequences = sequences
        self.N = self.sequences.shape[0]
        self.next_ind = np.array([], dtype=int)
        self.img_shape = tuple(self.sequences.shape[2:4])
        self.num_frames = self.sequences.shape[1]
        self.decoder = decoder

    def __iter__(self):
        return self

    def __next__(self):
        while len(self.next_ind) < self.batch_size:
            ind = np.arange(self.N, dtype=int)
            self.prng.shuffle(ind)
            self.next_ind = np.concatenate([self.next_ind, ind])

        ind = self.next_ind[:self.batch_size]
        self.next_ind = self.next_ind[self.batch_size:]
        frame_ind = self.prng.randint(self.num_frames, size=self.batch_size)

        X = self.augment_sequence_batch(self.sequences[ind,frame_ind,...])
        X = self.decoder(X)

        return X

    def augment_sequence(self, sequence):
        seq = sequence.copy()

        # mirror
        flips = []
        if bool(self.prng.randint(2)):
            flips.append(0)
        if bool(self.prng.randint(2)):
            flips.append(1)
        if flips:
            seq = np.flip(seq, axis=tuple(flips))

        # rotate
        num_rot = self.prng.randint(4)
        if num_rot > 0:
            seq = np.rot90(seq, k=num_rot, axes=(0,1))

        return seq

    def augment_sequence_batch(self, sequences):
        sequences = sequences.copy()
        for i in range(sequences.shape[0]):
            sequences[i,...] = self.augment_sequence(sequences[i,...])
        return sequences


class RainRateDecoder(object):
    def __init__(self, scaling_fn, value_range=(np.log10(0.1), np.log10(100)),
        below_val=np.nan, normalize=True):

        self.logR = np.log10(np.load(scaling_fn))
        self.logR[0] = np.nan
        #self.x = np.arange(len(self.logR))
        self.value_range = value_range
        self.below_val = below_val
        self.normalize_output = normalize

    def __call__(self, img):
        valid = (img != 0)
        img_dec = np.full(img.shape, np.nan, dtype=np.float32)
        img_dec[valid] = self.logR[img[valid]]
        img_dec[img_dec<self.value_range[0]] = self.below_val
        img_dec.clip(max=self.value_range[1], out=img_dec)
        if self.normalize_output:
            img_dec = self.normalize(img_dec)
        return img_dec

    def normalize(self, img):
        return (img-self.below_val) / \
            (self.value_range[1]-self.below_val) 

    def denormalize(self, img, set_nan=True):
        img = img*(self.value_range[1]-self.below_val) + self.below_val
        img[img < self.value_range[0]] = self.below_val
        if set_nan:
            img[img == self.below_val] = np.nan
        return img
