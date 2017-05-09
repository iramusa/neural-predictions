from keras.layers import Input, Dense, Convolution2D, Deconvolution2D, MaxPooling2D,\
    UpSampling2D, Merge, LSTM, Flatten, ZeroPadding2D, Reshape, BatchNormalization


IMAGE_SIZE_X = 28
IMAGE_SIZE_Y = 28
IMAGE_CHANNELS = 1

INPUT_IMAGE_SHAPE = (IMAGE_SIZE_Y, IMAGE_SIZE_X, IMAGE_CHANNELS)

V_SIZE = 64

POSITIONAL_ARGS = 'pos_args'
KEYWORD_ARGS = 'key_args'

ENCODER = {
        'name': 'conv4',
        'input_shape': INPUT_IMAGE_SHAPE,
        'output_shape': (V_SIZE,),
        'layers': [
            {
                'type': Convolution2D,
                POSITIONAL_ARGS: [8, 3, 3],
                KEYWORD_ARGS : {
                    'subsample': (2, 2),
                    'activation': 'relu',
                    'init': 'glorot_normal',
                    'border_mode': 'same'
                }
            },
            {
                'type': BatchNormalization,
                KEYWORD_ARGS: {
                    'mode': 2,
                }
            },
            {
                'type': Convolution2D,
                POSITIONAL_ARGS: [8, 3, 3],
                KEYWORD_ARGS : {
                    'subsample': (2, 2),
                    'activation': 'relu',
                    'init': 'glorot_normal',
                    'border_mode': 'same'
                }
            },
            {
                'type': Flatten,
            },
            {
                'type': BatchNormalization,
                KEYWORD_ARGS: {
                    'mode': 2,
                }
            },
            {
                'type': Dense,
                POSITIONAL_ARGS: [V_SIZE],
                KEYWORD_ARGS: {
                    'activation': 'relu',
                    'init': 'uniform',
                }
            },
        ],
    }

DECODER = {
        'name': 'deconv4',
        'input_shape': (V_SIZE,),
        'output_shape': INPUT_IMAGE_SHAPE,
        'layers': [
            {
                'type': BatchNormalization,
                KEYWORD_ARGS: {
                    'mode': 2,
                }
            },
            {
                'type': Dense,
                POSITIONAL_ARGS: [7*7*8],
                'output_dim': 7*7*8,
                KEYWORD_ARGS: {
                    'init': 'glorot_normal',
                    'activation': 'relu',
                }
            },
            {
                'type': BatchNormalization,
                KEYWORD_ARGS: {
                    'mode': 2,
                }
            },
            {
                'type': Reshape,
                POSITIONAL_ARGS: [(7, 7, 8)],
                'shape': (7, 7, 8),
            },
            {
                'type': UpSampling2D,
                POSITIONAL_ARGS: [(2, 2)]
            },


            {
                'type': Convolution2D,
                POSITIONAL_ARGS: [8, 3, 3],
                KEYWORD_ARGS : {
                    'init': 'glorot_normal',
                    'activation': 'relu',
                    'border_mode': 'same'

                }
            },
            {
                'type': BatchNormalization,
                KEYWORD_ARGS: {
                    'mode': 2,
                }
            },
            {
                'type': UpSampling2D,
                POSITIONAL_ARGS: [(2, 2)]
            },

            {
                'type': Convolution2D,
                POSITIONAL_ARGS: [1, 3, 3],
                KEYWORD_ARGS: {
                    'init': 'glorot_normal',
                    'activation': 'sigmoid',
                    'border_mode': 'same'
                }
            },
            # {
            #     'type': UpSampling2D,
            #     POSITIONAL_ARGS: [(2, 2)]
            # },
            # {
            #     'type': ZeroPadding2D,
            #     KEYWORD_ARGS: {
            #         'padding': (1, 1)
            #     }
            #
            # },
            # {
            #     'type': BatchNormalization,
            #     KEYWORD_ARGS: {
            #         'mode': 2,
            #     }
            # },
            # {
            #     'type': Convolution2D,
            #     POSITIONAL_ARGS: [64, 6, 6],
            #     KEYWORD_ARGS: {
            #         'init': 'glorot_normal',
            #         'activation': 'relu',
            #         'border_mode': 'same'
            #     }
            # },
            #
            # {
            #     'type': UpSampling2D,
            #     POSITIONAL_ARGS: [(2, 2)]
            # },
            # {
            #     'type': ZeroPadding2D,
            #     KEYWORD_ARGS: {
            #         'padding': (3, 2)
            #     }
            #
            # },
            # {
            #     'type': BatchNormalization,
            #     KEYWORD_ARGS: {
            #         'mode': 2,
            #     }
            # },
            # {
            #     'type': Convolution2D,
            #     POSITIONAL_ARGS: [1, 3, 3],
            #     KEYWORD_ARGS: {
            #         'init': 'glorot_normal',
            #         'activation': 'tanh',
            #         'border_mode': 'same'
            #     }
            # },
        ],
    }

SCREEN_DISCRIMINATOR = {
        'name': 'scr_disc',
        'input_shape': (V_SIZE,),
        'output_shape': (1,),
        'layers': [
            {
                'type': BatchNormalization,
                KEYWORD_ARGS: {
                    'mode': 2,
                }
            },
            {
                'type': Dense,
                POSITIONAL_ARGS: [1],
                KEYWORD_ARGS: {
                    'init': 'glorot_normal',
                    'activation': 'sigmoid',
                }
            },
        ],
    }

SHALLOW_ENCODER = {
        'name': 'sh_enc',
        'input_shape': INPUT_IMAGE_SHAPE,
        'output_shape': (1568,),
        'layers': [
            {
                'type': Convolution2D,
                POSITIONAL_ARGS: [8, 3, 3],
                KEYWORD_ARGS : {
                    'subsample': (2, 2),
                    'activation': 'relu',
                    'init': 'glorot_normal',
                    'border_mode': 'same'
                }
            },
            {
                'type': Flatten,
            },
        ],
    }


DEFAULT_STRUCTURE = {
    'encoder': ENCODER,
    'decoder': DECODER,
    'screen_discriminator': SCREEN_DISCRIMINATOR,
    'shallow_encoder': SHALLOW_ENCODER,
}
