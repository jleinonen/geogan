import json
import gc

from keras.models import Model
from keras.layers import Input, Concatenate, Average
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import generic_utils
import numpy as np

from layers import GradientPenalty, RandomWeightedAverage
import utils


class WGANGP(object):

    def __init__(self, gen, disc, load_fn_root=None,
        gradient_penalty_weight=10, lr_disc=0.0001, lr_gen=0.0001,
        avg_seed=None, num_channels=1):

        if load_fn_root is not None:
            load_files = self.filenames_from_root(load_fn_root)
            with open(load_files["gan_params"]) as f:
                params = json.load(f)
            gradient_penalty_weight = params["gradient_penalty_weight"]
            lr_disc = params["lr_disc"]
            lr_gen = params["lr_gen"]

        self.gen = gen
        self.disc = disc
        self.gradient_penalty_weight = gradient_penalty_weight
        self.lr_disc = lr_disc
        self.lr_gen = lr_gen
        self.num_channels = num_channels
        self.build_wgan_gp()

        if load_fn_root is not None:
            self.load(load_files)


    def filenames_from_root(self, root):
        fn = {
            "gen_weights": root+"-gen_weights.h5",
            "disc_weights": root+"-disc_weights.h5",
            "gen_opt_weights": root+"-gen_opt_weights.h5",
            "disc_opt_weights": root+"-disc_opt_weights.h5",
            "gan_params": root+"-gan_params.json"
        }
        return fn


    def load(self, load_files):
        self.gen.load_weights(load_files["gen_weights"])
        self.disc.load_weights(load_files["disc_weights"])
        
        self.disc.trainable = False
        self.gen_trainer._make_train_function()
        utils.load_opt_weights(self.gen_trainer,
            load_files["gen_opt_weights"])
        self.disc.trainable = True
        self.gen.trainable = False
        self.disc_trainer._make_train_function()
        utils.load_opt_weights(self.disc_trainer,
            load_files["disc_opt_weights"])
        self.gen.trainable = True


    def save(self, save_fn_root):
        paths = self.filenames_from_root(save_fn_root)
        self.gen.save_weights(paths["gen_weights"], overwrite=True)
        self.disc.save_weights(paths["disc_weights"], overwrite=True)
        utils.save_opt_weights(self.disc_trainer, paths["disc_opt_weights"])
        utils.save_opt_weights(self.gen_trainer, paths["gen_opt_weights"])
        params = {
            "gradient_penalty_weight": self.gradient_penalty_weight,
            "lr_disc": self.lr_disc,
            "lr_gen": self.lr_gen
        }
        with open(paths["gan_params"], 'w') as f:
            json.dump(params, f)


    def build_wgan_gp(self):

        # find shapes for inputs
        noise_shapes = utils.input_shapes(self.gen, "noise")

        # Create optimizers
        self.opt_disc = Adam(self.lr_disc, beta_1=0.5, beta_2=0.9)
        self.opt_gen = Adam(self.lr_gen, beta_1=0.5, beta_2=0.9)

        # Create generator training network
        self.disc.trainable = False
        noise_in = [Input(shape=s) for s in noise_shapes]
        gen_in = noise_in
        gen_out = self.gen(gen_in)
        gen_out = utils.ensure_list(gen_out)
        disc_in_gen = gen_out
        disc_out_gen = self.disc(disc_in_gen)
        self.gen_trainer = Model(inputs=gen_in, outputs=disc_out_gen)
        self.gen_trainer.compile(loss=wasserstein_loss,
            optimizer=self.opt_gen)
        self.disc.trainable = True

        # Create discriminator training network
        self.gen.trainable = False
        disc_in_real = Input(shape=(None,None,self.num_channels))
        noise_in = [Input(shape=s) for s in noise_shapes]
        disc_in_fake = self.gen(noise_in) 
        disc_in_avg = RandomWeightedAverage()([disc_in_real,disc_in_fake])
        disc_out_real = self.disc(disc_in_real)
        disc_out_fake = self.disc(disc_in_fake)
        disc_out_avg = self.disc(disc_in_avg)
        disc_gp = GradientPenalty()([disc_out_avg, disc_in_avg])

        self.disc_trainer = Model(inputs=[disc_in_real]+noise_in,
            outputs=[disc_out_real,disc_out_fake,disc_gp])
        self.disc_trainer.compile(
            loss=[wasserstein_loss, wasserstein_loss, 'mse'], 
            loss_weights=[1.0, 1.0, self.gradient_penalty_weight],
            optimizer=self.opt_disc
        )
        self.gen.trainable = True


    def train(self, batch_gen, noise_gen, num_gen_batches=1, 
        training_ratio=1, show_progress=True):

        disc_target_real = None
        if show_progress:
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(
                num_gen_batches*batch_gen.batch_size)
            batch_counter = 1

        for k in range(num_gen_batches):
        
            # train discriminator
            disc_loss = None
            disc_loss_n = 0
            for rep in range(training_ratio):
                # generate some real samples
                Y_real = next(batch_gen)
                noise = noise_gen()

                if disc_target_real is None: # on the first iteration
                    # run discriminator once just to find the shapes
                    disc_outputs = self.disc_trainer.predict(
                        [Y_real]+noise)
                    disc_target_real = np.ones(disc_outputs[0].shape,
                        dtype=np.float32)
                    disc_target_fake = -disc_target_real
                    gen_target = disc_target_real
                    gp_target = np.zeros(disc_outputs[2].shape, 
                        dtype=np.float32)
                    disc_target = [disc_target_real, disc_target_fake,
                        gp_target]
                    del disc_outputs

                try:
                    self.gen.trainable = False    
                    dl = self.disc_trainer.train_on_batch(
                        [Y_real]+noise, disc_target)
                finally:
                    self.gen.trainable = True

                if disc_loss is None:
                    disc_loss = np.array(dl)
                else:
                    disc_loss += np.array(dl)
                disc_loss_n += 1

                del Y_real

            disc_loss /= disc_loss_n

            try:
                self.disc.trainable = False
                gen_loss = self.gen_trainer.train_on_batch(
                    noise_gen(), gen_target)
            finally:
                self.disc.trainable = True

            if show_progress:
                losses = []
                for (i,dl) in enumerate(disc_loss):
                    losses.append(("D{}".format(i), dl))
                for (i,gl) in enumerate([gen_loss]):
                    losses.append(("G{}".format(i), gl))
                progbar.add(batch_gen.batch_size, 
                    values=losses)

            gc.collect()


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred, axis=-1)
