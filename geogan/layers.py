from keras.engine import InputSpec
from keras.layers import Layer
from keras.layers.convolutional import _Conv
from keras.layers.merge import _Merge
from keras.legacy import interfaces
from keras import backend as K


class BatchStd(Layer):

    def call(self, batch):
        batch_shape = K.shape(batch)[:-1]
        batch_shape = K.concatenate([batch_shape, (1,)])
        
        batch_std = K.mean(K.std(batch, axis=0, keepdims=True), 
            axis=-1, keepdims=True)

        return K.zeros(batch_shape) + batch_std

    def compute_output_shape(self, input_shapes):
        return input_shapes[:-1] + (1,)


class LocalizedAdaIN(Layer):
    def __init__(self, **kwargs):
        if "axis" in kwargs:
            self.axis = kwargs.pop("axis")
        else:
            channels_last = (K.image_data_format() == 'channels_last')
            self.axis = [-3,-2] if channels_last else [-2,-1]
        self.scale_ratio = kwargs.pop("scale_ratio")
        self.interpolation = kwargs.pop("interpolation", "bilinear")
        self.epsilon = 1e-2
        super(LocalizedAdaIN, self).__init__(**kwargs)

    def call(self, inputs):

        (x, scale, bias) = inputs
        mean = K.mean(x, axis=self.axis, keepdims=True)
        std = K.std(x, axis=self.axis, keepdims=True)+self.epsilon
        scale = K.resize_images(scale, self.scale_ratio, self.scale_ratio,
            K.image_data_format(), interpolation=self.interpolation)
        bias = K.resize_images(bias, self.scale_ratio, self.scale_ratio,
            K.image_data_format(), interpolation=self.interpolation)

        return (x-mean)/std * scale + bias

    def get_config(self):
        config = {
            "axis": self.axis,
            "scale_ratio": self.scale_ratio,
            "epsilon": self.epsilon,
        }
        base_config = super(LocalizedAdaIN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]


class PixelwiseInstanceNormalization(Layer):
    def __init__(self, **kwargs):
        if "axis" in kwargs:
            self.axis = kwargs.pop("axis")
        else:
            channels_last = (K.image_data_format() == 'channels_last')
            self.axis = [-3,-2] if channels_last else [-2,-1]
        self.epsilon = 1e-2
        super(PixelwiseInstanceNormalization, self).__init__(**kwargs)

    def call(self, x):
        mean = K.mean(x, axis=self.axis, keepdims=True)
        std = K.std(x, axis=self.axis, keepdims=True)+self.epsilon

        return (x-mean)/std

    def get_config(self):
        config = {
            "axis": self.axis,
            "epsilon": self.epsilon,
        }
        base_config = super(PixelwiseInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GradientPenalty(Layer):
    def __init__(self, **kwargs):
        super(GradientPenalty, self).__init__(**kwargs)

    def build(self, input_shapes):
        # Create a trainable weight variable for this layer.
        super(GradientPenalty, self).build(input_shapes)  # Be sure to call this somewhere!

    def call(self, inputs):
        target, wrt = inputs
        grad = K.gradients(target, wrt)[0]
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))-1

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)


class RandomWeightedAverage(_Merge):
    def build(self, input_shape):
        super(RandomWeightedAverage, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError('A `RandomWeightedAverage` layer should be '
                             'called on exactly 2 inputs')

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `RandomWeightedAverage` layer should be '
                             'called on exactly 2 inputs')

        (x,y) = inputs
        shape = K.shape(x)
        weights = K.random_uniform(shape[:1],0,1)
        for i in range(len(K.int_shape(x))-1):
            weights = K.expand_dims(weights,-1)
        return x*weights + y*(1-weights)


class SNConv2D(_Conv):
    # From https://github.com/jason71995/Keras-GAN-Library/
    #   blob/master/gan_libs/SNGAN.py
    @interfaces.legacy_conv2d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super(SNConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.input_spec = InputSpec(ndim=4)
        self.Ip = 1
        self.u = self.add_weight(
            name='W_u',
            shape=(1,filters),
            initializer='random_uniform',
            trainable=False
        )

    def call(self, inputs):
        outputs = K.conv2d(
            inputs,
            self.W_bar(),
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


    def get_config(self):
        config = super(SNConv2D, self).get_config()
        config.pop('rank')
        return config

    def W_bar(self):
        # Spectrally Normalized Weight
        W_mat = K.permute_dimensions(self.kernel, (3, 2, 0, 1)) # (h, w, i, o) => (o, i, h, w)
        W_mat = K.reshape(W_mat,[K.shape(W_mat)[0], -1]) # (o, i * h * w)

        if not self.Ip >= 1:
            raise ValueError("The number of power iterations should be positive integer")

        _u = self.u
        _v = None

        for _ in range(self.Ip):
            _v = _l2normalize(K.dot(_u, W_mat))
            _u = _l2normalize(K.dot(_v, K.transpose(W_mat)))

        sigma = K.sum(K.dot(_u,W_mat)*_v)

        K.update(self.u,K.in_train_phase(_u, self.u))
        return self.kernel / sigma
        

def _l2normalize(x):
    return x / K.sqrt(K.sum(K.square(x)) + K.epsilon())