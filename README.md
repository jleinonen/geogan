# Style-based GAN for geophysical fields

This is a reference implementation of a style-based generative adversarial network (GAN) designed to generate geophysical fields. This code supports a paper to be submitted to the Climate Informatics 2019 meeting.

You might also be interested in [this paper](https://doi.org/10.1029/2019GL082532) ([code](../../../cloudsat-gan/)) or in [my post on GANs in the atmospheric sciences](https://jleinonen.github.io/2019/06/06/gans-atmos.html).

## Obtaining the data

Download the data from https://doi.org/10.7910/DVN/ZDWWMG and follow the instructions there.

## Obtaining the trained network

Get the trained weights from [this release](../../releases/download/v0.1-data/mch_gan.zip). Unzip the file, preferably into the `models` directory.

## Running the code

For training, you'll want a machine with a GPU and around 32 GB of memory (the training procedure for the radar dataset loads the entire dataset into memory). Running the pre-trained model should work just fine on a CPU.

You may want to work with the code interactively; in this case, just start a Python shell in the `geogan` directory.

If you want the simplest way to run the code, the following two options are available. You may also want to look at what `main.py` does in order to get an idea of how the training flow works.

### Producing plots

You can replicate the plots in the paper by going to the `geogan` directory and using
```
python main.py plot --data_file=<data_file> --weights_root=<weights_root>
```
where `<data_file>` is the path to the training data and `<weights_root>` is the path and prefix of the stored weights. For example, if you unzipped the pre-trained weights to the `models` directory, you should use `--weights_root=../models/mch_gan`.

### Training the model

Run the following to start the training:
```
python main.py train --data_file=<data_file> --weights_root=<weights_root>
```
where the parameters are as above. This will run the training loop until terminated (with e.g. ctrl-C) and save the weights after each 100 batches (overwriting the original weights, so save the weights under a different name if you want to avoid this).
