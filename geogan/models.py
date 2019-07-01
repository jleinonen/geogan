from keras import backend as K
from keras.layers import Input, Concatenate, Add
from keras.layers import LeakyReLU
from keras.layers import Dense, Conv2D, UpSampling2D
from keras.layers import GlobalAveragePooling2D
from keras.models import Model

from layers import BatchStd, PixelwiseInstanceNormalization, LocalizedAdaIN
from layers import SNConv2D
from utils import input_shapes


def stylegan_subblock(channels, scale_ratio=2):
    def block(x, noise, style_weights):
        x = Conv2D(channels, kernel_size=(3,3), padding='same',
            kernel_initializer="glorot_normal")(x)
        x = LeakyReLU(0.2)(x)
        if noise is not None:
            scaled_noise = Conv2D(channels, kernel_size=(1,1),
                padding='same',
                use_bias=False, kernel_initializer="he_normal")(noise)
            x = Add()([x,scaled_noise])
        scale = Conv2D(channels, kernel_size=(1,1), 
            kernel_initializer="lecun_normal", bias_initializer='ones')(style_weights)
        bias = Conv2D(channels, kernel_size=(1,1),
            kernel_initializer="lecun_normal", bias_initializer="zeros")(style_weights)
        x = LocalizedAdaIN(scale_ratio=scale_ratio)([x,scale,bias])
        return x
    return block


def stylegan_block(channels, scale_ratio=2, upscale=True):
    def block(x, noise, style_weights):
        if upscale:
            x = UpSampling2D(interpolation='nearest')(x)

        x = stylegan_subblock(channels, scale_ratio=scale_ratio)(
            x, noise[0], style_weights)
        x = stylegan_subblock(channels, scale_ratio=scale_ratio)(
            x, noise[1], style_weights)

        return x

    return block


def style_network(num_styles=32):
    z = Input(shape=(None,None,num_styles), name="noise_in_0")

    w = Conv2D(num_styles, kernel_size=(3,3), padding='valid',
        kernel_initializer="he_normal")(z)
    w = LeakyReLU(0.2)(w)
    w = PixelwiseInstanceNormalization()(w)
    w = Conv2D(num_styles, kernel_size=(3,3), padding='valid',
        kernel_initializer="he_normal", activation="tanh")(w)

    model = Model(inputs=z, outputs=w)
    return model


def generator(num_channels=1, num_styles=32):
    style = Input(shape=(None,None,num_styles), name="style_in_0")
    initial_noise = Input(shape=(None,None,1), name="noise_in_1")
    additive_noise = [Input(shape=(None,None,1),
        name="noise_in_{}".format(i+2)) for i in range(12)]
    noise_in = [initial_noise] + additive_noise

    x = Conv2D(512, kernel_size=(1,1), use_bias=False,
        kernel_initializer="he_normal")(initial_noise)
    block_channels = [512, 512, 256, 128, 64, 64]
    for (i,channels) in enumerate(block_channels):
        x = stylegan_block(channels, scale_ratio=2**i, upscale=(i!=0))(
            x, additive_noise[i*2:i*2+2], style
        )

    out = Conv2D(num_channels, kernel_size=(1,1), padding='same',
        activation='linear')(x)

    model = Model(inputs=[style]+noise_in, outputs=out)

    def noise_shapes(img_shape=(128,128)):
        style_shape = (img_shape[0]//32, img_shape[1]//32, num_styles)
        initial_shape = (img_shape[0]//32, img_shape[1]//32, 1)
        additive_shape = []
        for (i,ch) in enumerate(block_channels):
            sc = 2**(len(block_channels)-i-1)
            shape = (img_shape[0]//sc, img_shape[1]//sc, 1)
            additive_shape.append(shape)
            additive_shape.append(shape)
        return [style_shape, initial_shape] + additive_shape

    return (model, noise_shapes)


def generator_styled(num_channels=1, num_styles=32):
    (gen, gen_noise_shapes) = generator(
        num_channels=num_channels, num_styles=num_styles)
    gen_noise_specs = input_shapes(gen, "noise")
    gen_noise = [Input(shape=s, name="noise_in_{}".format(i+1)) 
        for (i,s) in enumerate(gen_noise_specs)]
    
    styling = style_network(num_styles=num_styles)
    style_noise = Input(shape=(None,None,num_styles), name="noise_in_0")
    style = styling(style_noise)

    gen_inputs = [style] + gen_noise
    out = gen(gen_inputs)

    def noise_shapes(img_shape=(128,128)):
        style_noise_shape = (
            img_shape[0]//32+4, img_shape[1]//32+4, num_styles
        )
        return [style_noise_shape]+gen_noise_shapes(img_shape)[1:]

    gen_styled = Model(inputs=[style_noise]+gen_noise, outputs=out)

    return (gen_styled, gen, styling, noise_shapes)


def conv_block(channels, conv_size=(3,3), stride=1, 
    activation=True, padding='same', 
    kernel_initializer="he_normal"):

    def block(x):
        x = SNConv2D(channels, conv_size, padding=padding,
            strides=(stride,stride), 
            kernel_initializer=kernel_initializer)(x)
        if activation:
            x = LeakyReLU(0.2)(x)
        return x

    return block


def down_block(channels, conv_size=(3,3), stride=1):

    def block(x):
        x = conv_block(channels, conv_size=conv_size, stride=stride)(x)
        x = conv_block(channels, conv_size=conv_size, stride=1)(x)
        return x

    return block


def discriminator(num_channels=1):
    disc_in = Input(shape=(None,None,num_channels), name="sample_in")
    
    x = down_block(64, conv_size=(5,5))(disc_in)
    batch_std = BatchStd()(x)
    x = Concatenate()([x,batch_std])
    x = down_block(64,stride=2)(x)
    x = down_block(128,stride=2)(x)
    x = down_block(256,stride=2)(x)
    x = down_block(512,stride=2)(x)
    batch_std = BatchStd()(x)
    x = Concatenate()([x,batch_std])
    x = down_block(512,stride=2)(x)
    x = GlobalAveragePooling2D()(x)
    disc_out = Dense(1, activation='linear')(x)

    disc = Model(inputs=disc_in, outputs=disc_out)

    return disc
