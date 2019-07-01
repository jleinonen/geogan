# Style-based GAN for geophysical fields

This is a reference implementation of a style-based generative adversarial network (GAN) designed to generate geophysical fields. This code supports a paper to be submitted to the Climate Informatics 2019 meeting.

## Obtaining the data

We are currently discussing the release of the radar training dataset with the provider. Unfortunately, running the training using this dataset will not be possible before the data is release. Information about data access will be added here once it's available.

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
