
"""
Authors: Oshani Jayawardane, Buddika Weerasinghe

This file contains the functions for the U-Net model and Autoencoder Model. The following functions are included:

- build_autoencoder(input_shape, num_filters=16)
- build_unet(input_shape, num_filters=16, num_classes=1)

"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Input, concatenate, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2 

def conv_block(input_layer, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input_layer)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.1)(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


# Encoder will be the same for Autoencoder and U-net
def encoder_block(input_layer, num_filters):
    x = conv_block(input_layer, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p 


# Decoder block for autoencoder (no skip connections)
def decoder_block(input_layer, num_filters):
    # x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_layer)
    x = UpSampling2D((2, 2))(input_layer)
    x = Conv2D(num_filters, 3, activation="relu", padding="same", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5))(x)
    x = conv_block(x, num_filters)
    return x


# skip features gets input from encoder for concatenation
def decoder_block_for_unet(input_layer, skip_features, num_filters):
    # x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_layer)
    x = UpSampling2D((2, 2))(input_layer)
    x = Conv2D(num_filters, 3, activation="relu", padding="same", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5))(x)
    x = concatenate([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_autoencoder(input_shape, num_filters=16):
    input_layer = Input(input_shape)
    
    # Encoder
    _, p1 = encoder_block(input_layer, num_filters) 
    _, p2 = encoder_block(p1, num_filters*2)
    _, p3 = encoder_block(p2, num_filters*4*2)
    _, p4 = encoder_block(p3, num_filters*8*2)
    
    # Bridge
    b = conv_block(p4, num_filters*16*2) 
    
    # Decoder
    d1 = decoder_block(b, num_filters*8*2)
    d2 = decoder_block(d1, num_filters*4*2)
    d3 = decoder_block(d2, num_filters*2)
    d4 = decoder_block(d3, num_filters)
    
    output_layer = Conv2D(1, 3, activation="sigmoid", padding="same")(d4)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer, name="Autoencoder")
    
    return autoencoder


def build_unet(input_shape, num_filters, num_classes):
    input_layer = Input(input_shape)
    
    s1, p1 = encoder_block(input_layer, num_filters)
    s2, p2 = encoder_block(p1, num_filters*2)
    s3, p3 = encoder_block(p2, num_filters*4*2)
    s4, p4 = encoder_block(p3, num_filters*8*2)

    b1 = conv_block(p4, num_filters*16*2)

    d1 = decoder_block_for_unet(b1, s4, num_filters*8*2)
    d2 = decoder_block_for_unet(d1, s3, num_filters*4*2)
    d3 = decoder_block_for_unet(d2, s2, num_filters*2)
    d4 = decoder_block_for_unet(d3, s1, num_filters)
    
    # Output
    if num_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    
    output_layer = Conv2D(num_classes, (1, 1), activation=activation)(d4)

    model = Model(inputs=[input_layer], outputs=[output_layer], name="U-Net")
    return model
