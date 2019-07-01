import gc

from matplotlib import pyplot as plt
from matplotlib import colors, gridspec
import numpy as np

import models
import noise
import train


def plot_img_mch(img, logR_range=(np.log10(0.1), np.log10(100))):
    plt.imshow(img[:,:,0], interpolation='nearest',
        norm=colors.Normalize(*logR_range))
    plt.gca().tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)


def plot_img_goes(img):
    plt.imshow(img.clip(0,1), interpolation='nearest')
    plt.gca().tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)


def plot_img_grid(Y, num_rows, num_cols, application="mch",
    out_fn=None):
    plot_img = {
        "mch": plot_img_mch,
        "goes": plot_img_goes,
    }[application]

    figsize = (num_cols, num_rows)
    plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.05, hspace=0.05)

    k = 0
    for i in range(num_rows):
        for j in range(num_cols):
            plt.subplot(gs[i,j])
            plot_img(Y[k,:,:,:])
            k += 1

    if out_fn is not None:
        plt.savefig(out_fn, bbox_inches='tight')
        plt.close()


def plot_samples(gen, batch_gen, noise_gen, out_fn=None, num_rows=2,
    num_cols=9, application="mch"):

    try:
        old_batch_size = noise_gen.batch_size
        batch_gen.batch_size = num_rows*num_cols
        noise_gen.batch_size = num_rows*num_cols
        Y_real = next(batch_gen)
        noise = noise_gen()
    finally:
        batch_gen.batch_size = old_batch_size
        noise_gen.batch_size = old_batch_size

    if application == "mch":
        Y_real = batch_gen.decoder.denormalize(Y_real)
    elif application == "goes":
        Y_real = Y_real*0.5 + 0.5

    Y_pred = gen.predict(noise)
    if application == "mch":
        Y_pred = batch_gen.decoder.denormalize(Y_pred)
    elif application == "goes":
        Y_pred = Y_pred*0.5 + 0.5

    Y = np.concatenate((Y_real,Y_pred))
    plot_img_grid(Y, num_rows*2, num_cols,
        application=application, out_fn=out_fn)


def plot_styles(gen, batch_gen, noise_gen, out_fn=None,
    samples_per_style=9, application="mch"):

    num_styles = len(noise_gen)
    try:
        old_batch_size = noise_gen[0].batch_size
        for ng in noise_gen:
            ng.batch_size = samples_per_style
        noise_single = [ng() for ng in noise_gen]
        noise = []
        for i in range(len(noise_single[0])):
            noise.append(np.concatenate(
                [noise_single[k][i] for k in range(len(noise_gen))]
            ))
    finally:
        for ng in noise_gen:
            ng.batch_size = old_batch_size

    for k0 in range(0,num_styles*samples_per_style,samples_per_style):
        for k in range(k0,k0+samples_per_style):
            noise[0][k,...] = noise[0][k0,...]

    Y_pred = gen.predict(noise)
    if application == "mch":
        Y_pred = batch_gen.decoder.denormalize(Y_pred)
    elif application == "goes":
        Y_pred = Y_pred*0.5 + 0.5

    plot_img_grid(Y_pred, num_styles, samples_per_style,
        application=application, out_fn=out_fn)


def plot_transition(gen, styling, batch_gen, noise_gen, 
    out_fn=None, num_samples=2,
    num_transitions=9, application="mch"):

    try:
        old_batch_size = noise_gen.batch_size
        noise_gen.batch_size = num_samples
        def noise_gen_styled():
            noise = noise_gen()
            noise[0] = styling.predict(noise[0])
            return noise
        noise = noise_gen_styled()
    finally:
        noise_gen.batch_size = old_batch_size

    t = np.linspace(0,1,num_transitions)

    style_noise = []
    for i in range(num_samples):
        for j in range(num_transitions):
            # interpolate from noise[0][i,...] to -noise[0][i,...]
            n = noise[0][i,...]*(1-t[j]) - noise[0][i,...]*t[j]
            style_noise.append(n)
    noise[0] = np.stack(style_noise)
    for i in range(1,len(noise)):
        noise[i] = np.stack(
            [noise[i][0,...]]*(num_samples*num_transitions)
        )

    Y_pred = gen.predict(noise)
    if application == "mch":
        Y_pred = batch_gen.decoder.denormalize(Y_pred)
    elif application == "goes":
        Y_pred = Y_pred*0.5 + 0.5

    plot_img_grid(Y_pred, num_samples, num_transitions,
        application=application, out_fn=out_fn)


def interpolate_noise(n1, n2, num_points=5):
    r1 = np.sqrt((n1**2).sum())
    r2 = np.sqrt((n2**2).sum())

    line = np.linspace(0,1,num_points)
    rad = np.linspace(r1,r2,num_points)
    noise_ip = []

    for (i,t) in enumerate(line):
        n = (1-t)*n1+t*n2
        n *= r[i]/np.sqrt((n**2).sum())
        noise_ip.append(n)

    return np.stack(n)


def plot_all(data_fn, gen_weights_fn, application="mch"):
    num_channels = {
        "mch": 1,
        "goes": 3
    }[application]
    (gen_styled, gen, styling, noise_shapes) = models.generator_styled(
        num_channels=num_channels)
    gen_styled.load_weights(gen_weights_fn)

    (wgan, batch_gen, noise_shapes, steps_per_epoch) = train.setup_gan(
        data_fn, n_samples=128, sample_random=True, application=application,
        random_seed=321459)
    
    noise_gen = noise.NoiseGenerator(noise_shapes(), 
        batch_size=batch_gen.batch_size, random_seed=34)
    plot_samples(gen_styled, batch_gen, noise_gen, 
        out_fn="../figures/{}_samples.pdf".format(application))

    noise_gen_1 = noise.NoiseGenerator(noise_shapes(), 
        batch_size=batch_gen.batch_size, random_seed=221)
    noise_gen_2 = noise.NoiseGenerator(noise_shapes(), 
        batch_size=batch_gen.batch_size, random_seed=70)
    noise_gen_3 = noise.NoiseGenerator(noise_shapes(), 
        batch_size=batch_gen.batch_size, random_seed=39)
    plot_styles(gen_styled, batch_gen, 
        [noise_gen_1, noise_gen_2, noise_gen_3], 
        out_fn="../figures/{}_styles.pdf".format(application))

    noise_gen = noise.NoiseGenerator(noise_shapes(), 
        batch_size=batch_gen.batch_size, random_seed=241)
    plot_transition(gen, styling, batch_gen, noise_gen, 
        out_fn="../figures/{}_transition.pdf".format(application))

    gc.collect()
