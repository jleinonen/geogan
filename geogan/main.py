import argparse

import plots
import train


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help="train or plot")
    parser.add_argument('--data_file', type=str, 
        help="Training data file")
    parser.add_argument('--weights_root', type=str, default="",
        help="Network weights file root")
    parser.add_argument('--application', type=str, default="mch",
        help="Training data file file")
    
    args = parser.parse_args()
    mode = args.mode
    data_fn = args.data_file
    weights_root = args.weights_root
    application = args.application

    if mode == "train":
        (wgan, batch_gen, noise_shapes, steps_per_epoch) = train.setup_gan(
            data_fn, batch_size=64, application=application)
        if weights_root:
            wgan.load(wgan.filenames_from_root(weights_root))
        while True:
            train.train_gan(wgan, batch_gen, noise_shapes, 100, 1, 
                application=application);
            wgan.save(weights_root)
    elif mode == "plot":
        gen_weights_file = weights_root+"-gen_weights.h5"
        plots.plot_all(data_fn, gen_weights_file, application=application)
