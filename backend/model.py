import tensorflow as tf
import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Reshape, Dropout, BatchNormalization, LSTM, GRU, LayerNormalization, Activation, GlobalMaxPooling1D, LeakyReLU, ReLU, SpatialDropout1D
from keras.layers import Input, Concatenate, Add, UpSampling1D, MultiHeadAttention, AveragePooling1D, GlobalAveragePooling1D, Attention, Lambda
from keras.layers import Conv2D, MaxPooling2D

from keras.initializers import HeNormal

from keras.metrics import Precision, Recall, Accuracy, AUC, BinaryAccuracy
from keras import Model

from gradient_accumulator import GradientAccumulateModel, GradientAccumulateOptimizer

from keras import layers, models

from keras.optimizers import Adam

from keras.regularizers import l2

from keras.losses import binary_crossentropy

# image preprocessing params
PATCH_SIZE = 16
NUM_CHANNELS = 3


# training params
DROPOUT = 0.1
LEARNING_RATE = 0.001


def transformer_encoder(inputs, head_size, num_heads, dff, dropout=0):
    # Attention and Normalization
    
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

     # Feed-Forward Network
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(dff, activation='gelu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)

    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    dff,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, dff, dropout)

    max_pool = layers.GlobalMaxPooling1D(data_format="channels_last")(x)
    avg_pool = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    x = layers.Concatenate()([max_pool, avg_pool])

    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    
    outputs = layers.Dense(2, activation="sigmoid")(x)

    return keras.Model(inputs, outputs)


def transformer_model():
    """
    Function returns a compiled model.

    """
    model = build_model(
        (None, PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS),
        head_size=128,
        num_heads=4,
        dff=128,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=DROPOUT,
        # dropout=DROPOUT,
    )

    model.compile(
        
        # optimizer=grad_acum_optimizer,
        optimizer=Adam(learning_rate=LEARNING_RATE),
        
        # loss=weighted_binary_crossentropy(weights), # loss for classification
        # loss=weighted_mse(weights), #  loss for regression
        metrics=[
            # OutputRange(total_data_percentage=0.9, aggregation_count=200, output_class=0, name='or_0'),
            # AUC(), 
            # BinaryAccuracy(), 
            # Recall(), # proportion of actual positives that were correctly predicted
            # FPR(0.52),
            # FPR(0.54),
            # Precision(0.51),
            # Precision(0.55),
            # F1Score(),
            # TODO output range, like 90% of outputs should be e.g. between 0.4 and 0.6
        ])
    # model.summary()
    return model

