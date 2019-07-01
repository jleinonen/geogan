import gc

import netCDF4
import numpy as np

import gan
import goesdata
import mchdata
import models
import noise
import plots


def setup_gan(data_file=None, application="mch",
    num_epochs=1, steps_per_epoch=None, training_ratio=1,
    batch_size=32, sample_random=False, scaling_fn="../data/scale_rzc.npy",
    error_fn="../data/goes_errors.json", n_samples=None, random_seed=None,
    lr_disc=0.0001, lr_gen=0.0001):

    if data_file is not None:
        if application == "mch":
            with netCDF4.Dataset(data_file, 'r') as ds:
                if n_samples is None:
                    seq = np.array(ds["sequences"][:], copy=False)
                else:
                    if sample_random:
                        prng = np.random.RandomState(seed=random_seed)
                        ind = prng.choice(ds["sequences"].shape[0], n_samples,
                            replace=False)
                        seq = np.array(ds["sequences"][ind,...], copy=False)
                    else:
                        seq = np.array(ds["sequences"][n_samples[0]:n_samples[1]],
                            copy=False)

            dec = mchdata.RainRateDecoder(scaling_fn, below_val=np.log10(0.025))
            batch_gen = mchdata.BatchGenerator(seq, dec, batch_size=batch_size,
                random_seed=random_seed)
            num_channels = 1
        elif application == "goes":
            batch_gen = goesdata.BatchGenerator(data_file, errors_file=error_fn,
                batch_size=batch_size, random_seed=random_seed)
            num_channels = 3
        else:
            raise ValueError("Unknown application.")

        if steps_per_epoch is None:
            steps_per_epoch = batch_gen.N//batch_gen.batch_size

    (gen_styled, gen, styling, noise_shapes) = models.generator_styled(
        num_channels=num_channels)
    disc = models.discriminator(num_channels=num_channels)
    wgan = gan.WGANGP(gen_styled, disc, lr_disc=lr_disc, lr_gen=lr_gen,
        num_channels=num_channels)

    gc.collect()

    return (wgan, batch_gen, noise_shapes, steps_per_epoch)


def train_gan(wgan, batch_gen, noise_shapes, steps_per_epoch, num_epochs,
    application="mch"):
    img_shape = batch_gen.img_shape
    noise_gen = noise.NoiseGenerator(noise_shapes(img_shape),
        batch_size=batch_gen.batch_size)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1,num_epochs))
        wgan.train(batch_gen, noise_gen, steps_per_epoch, training_ratio=5)
        plots.plot_samples(wgan.gen, batch_gen, noise_gen,
            application=application,
            out_fn="../figures/progress_{}.pdf".format(application))

    return wgan
