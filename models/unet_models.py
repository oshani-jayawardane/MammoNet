
"""
Authors: Oshani Jayawardane, Buddika Weerasinghe

This file contains the functions for the U-Net model and Autoencoder Model. The following functions are included:

- unet_model
- res_unet_model
- TL_vgg_unet_model

The encoder part of the U-Net model is replaced with the VGG16 model. The decoder part is the same as the original U-Net model.
the encoder layers are frozen (excluding the last maxpooling layer in VGG16 - taken as bridge layer in UNet) and the decoder layers are trainable.

"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dropout, Input, MaxPooling2D, Conv2DTranspose, concatenate, UpSampling2D, Add, multiply
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2 
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50



def conv_block(x, num_filters, filter_size, dropout, batch_norm=False, reg=1e-6):
    conv = Conv2D(num_filters, (filter_size, filter_size), padding="same", kernel_regularizer=l1_l2(l1=reg, l2=reg))(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation("relu")(conv)

    conv = Conv2D(num_filters, (filter_size, filter_size), padding="same", kernel_regularizer=l1_l2(l1=reg, l2=reg))(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation("relu")(conv)
    
    if dropout > 0:
        conv = Dropout(dropout)(conv)

    return conv


def res_conv_block(x, num_filters, filter_size, dropout, batch_norm=False, reg=1e-6):
    '''
    Residual convolutional layer.
    Two variants....
    Either put activation function before the addition with shortcut
    or after the addition (which would be as proposed in the original resNet).
    
    1. conv - BN - Activation - conv - BN - Activation 
                                          - shortcut  - BN - shortcut+BN
                                          
    2. conv - BN - Activation - conv - BN   
                                     - shortcut  - BN - shortcut+BN - Activation                                     
    
    Check fig 4 in https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf
    '''

    # Main path
    conv = Conv2D(num_filters, (filter_size, filter_size), padding="same", kernel_regularizer=l1_l2(l1=reg, l2=reg))(x)
    if batch_norm:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation("relu")(conv)
    
    conv = Conv2D(num_filters, (filter_size, filter_size), padding="same", kernel_regularizer=l1_l2(l1=reg, l2=reg))(conv)
    if batch_norm:
        conv = BatchNormalization(axis=3)(conv)
    #conv = layers.Activation('relu')(conv)    #Activation before addition with shortcut

    if dropout > 0:
        conv = Dropout(dropout)(conv)
    
    # Shortcut path
    shortcut = Conv2D(num_filters, (1, 1), padding="same")(x)
    if batch_norm:
        shortcut = BatchNormalization(axis=3)(shortcut)

    # Add shortcut to the main path
    res_path = Add()([conv, shortcut])
    res_path = Activation("relu")(res_path) #Activation after addition with shortcut (Original residual block)

    return res_path



def unet_model(num_classes=1, input_size=(256, 256, 3), num_filters=64, filter_size=3, dropout=0.1, batch_norm=False):
    input_layer = Input(input_size)

    # Encoder
    c1 = conv_block(input_layer, num_filters, filter_size, dropout, batch_norm, reg=1e-6)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, num_filters*2, filter_size, dropout, batch_norm, reg=1e-6)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, num_filters*4, filter_size, dropout, batch_norm, reg=1e-5)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = conv_block(p3, num_filters*8, filter_size, dropout, batch_norm, reg=1e-5)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bridge
    bridge = conv_block(p4, num_filters*16, filter_size, dropout, batch_norm, reg=1e-4)

    # Decoder
    u6 = Conv2DTranspose(num_filters*8, (2, 2), strides=(2, 2), padding='same')(bridge)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, num_filters*8, filter_size, dropout, batch_norm, reg=1e-5)

    u7 = Conv2DTranspose(num_filters*4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, num_filters*4, filter_size, dropout, batch_norm, reg=1e-5)

    u8 = Conv2DTranspose(num_filters*2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, num_filters*2, filter_size, dropout, batch_norm, reg=1e-6)

    u9 = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = conv_block(u9, num_filters, filter_size, dropout, batch_norm, reg=1e-6)
    
    # Output
    if num_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    
    output_layer = Conv2D(num_classes, (1, 1), activation=activation)(c9)

    model = Model(inputs=[input_layer], outputs=[output_layer], name="U-Net")
    
    return model


# x = UpSampling2D((2, 2))(input_layer)
# x = Conv2D(num_filters, 3, activation="relu", padding="same", kernel_regularizer=l1_l2(l1=1e-6, l2=1e-6))(x)



def res_unet_model(num_classes=1, input_size=(256, 256, 3), num_filters=64, filter_size=3, dropout=0.1, batch_norm=False):
    input_layer = Input(input_size)

    # Encoder
    c1 = res_conv_block(input_layer, num_filters, filter_size, dropout, batch_norm, reg=1e-6)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = res_conv_block(p1, num_filters*2, filter_size, dropout, batch_norm, reg=1e-6)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = res_conv_block(p2, num_filters*4, filter_size, dropout, batch_norm, reg=1e-5)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = res_conv_block(p3, num_filters*8, filter_size, dropout, batch_norm, reg=1e-5)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bridge
    bridge = res_conv_block(p4, num_filters*16, filter_size, dropout, batch_norm, reg=1e-4)

    # Decoder
    u6 = Conv2DTranspose(num_filters*8, (2, 2), strides=(2, 2), padding='same')(bridge)
    u6 = concatenate([u6, c4])
    c6 = res_conv_block(u6, num_filters*8, filter_size, dropout, batch_norm, reg=1e-5)

    u7 = Conv2DTranspose(num_filters*4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = res_conv_block(u7, num_filters*4, filter_size, dropout, batch_norm, reg=1e-5)

    u8 = Conv2DTranspose(num_filters*2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = res_conv_block(u8, num_filters*2, filter_size, dropout, batch_norm, reg=1e-6)

    u9 = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = res_conv_block(u9, num_filters, filter_size, dropout, batch_norm, reg=1e-6)
    
    # Output
    if num_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    
    output_layer = Conv2D(num_classes, (1, 1), activation=activation)(c9)

    res_unet_model = Model(inputs=[input_layer], outputs=[output_layer], name="Residual-U-Net")
    
    return res_unet_model



# Unet pre-trained model with VGG16 (weights: imagenet) 

def TL_vgg_unet_model(num_classes=1, num_filters=64, input_shape=(256, 256, 3), dropout=0, batch_norm=False):
    """
    @input_shape : tuple, (height, width, channels)
    @num_classes : int, number of classes
    return: model
    """

    input_shape = input_shape
    base_VGG = VGG16(include_top = False, weights = "imagenet", input_shape = input_shape)

    # freezing all layers in VGG16 
    for layer in base_VGG.layers: 
        layer.trainable = False

    # the bridge (exclude the last maxpooling layer in VGG16) 
    bridge = base_VGG.get_layer("block5_conv3").output
    # print(bridge.shape)

    # Decoder
    u6 = Conv2DTranspose(num_filters*8, (2, 2), strides=(2, 2), padding='same')(bridge)
    u6 = concatenate([u6, base_VGG.get_layer("block4_conv3").output], axis=3)
    c6 = conv_block(u6, num_filters*8, 3, dropout, batch_norm, reg=1e-4)
    
    u7 = Conv2DTranspose(num_filters*4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, base_VGG.get_layer("block3_conv3").output], axis=3)
    c7 = conv_block(u7, num_filters*4, 3, dropout, batch_norm, reg=1e-5)

    u8 = Conv2DTranspose(num_filters*2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, base_VGG.get_layer("block2_conv2").output], axis=3)
    c8 = conv_block(u8, num_filters*2, 3, dropout, batch_norm, reg=1e-5)

    u9 = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, base_VGG.get_layer("block1_conv2").output], axis=3)
    c9 = conv_block(u9, num_filters, 3, dropout, batch_norm, reg=1e-6)
    
    if num_classes==1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    output_layer = Conv2D(num_classes, (1, 1), activation=activation)(c9)

    vgg_model = Model(inputs=[base_VGG.input], outputs=[output_layer], name="TL-VGG-UNet")

    return vgg_model



# Res-UNet pre-trained model with ResNet50 (weights: imagenet) 

def TL_resnet_unet_model(num_classes=1, num_filters=64, input_shape=(256, 256, 3), dropout=0, batch_norm=False):
    base_ResNet = ResNet50(include_top=False, weights="imagenet", input_tensor=Input(shape=input_shape))

    # Freezing the layers
    for layer in base_ResNet.layers:
        layer.trainable = False

    # Accessing ResNet50 layers for skip connections
    layer_names = ['conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out', 'conv1_relu']  # ResNet50 layers
    layers = [base_ResNet.get_layer(name).output for name in layer_names]

    # Bridge
    bridge = base_ResNet.output

    # Decoder
    u6 = Conv2DTranspose(num_filters*8, (2, 2), strides=(2, 2), padding='same')(bridge)
    u6 = concatenate([u6, layers[0]], axis=3)
    c6 = res_conv_block(u6, num_filters*8, 3, dropout, batch_norm, reg=1e-4)

    u7 = Conv2DTranspose(num_filters*4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, layers[1]], axis=3)
    c7 = res_conv_block(u7, num_filters*4, 3, dropout, batch_norm, reg=1e-5)

    u8 = Conv2DTranspose(num_filters*2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, layers[2]], axis=3)
    c8 = res_conv_block(u8, num_filters*2, 3, dropout, batch_norm, reg=1e-5)

    u9 = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, layers[3]], axis=3)
    c9 = res_conv_block(u9, num_filters, 3, dropout, batch_norm, reg=1e-6)
    
    # Output layer
    if num_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    output_layer = Conv2D(num_classes, (1, 1), activation=activation)(c9)

    resnet_model = Model(inputs=base_ResNet.input, outputs=output_layer, name="TL-ResNet-UNet")

    return resnet_model



################################################################################################
# Attention Mechanism
################################################################################################

def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def attention_block(skip_conn, gating, inter_shape):
    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(skip_conn)
    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(gating)

    concat_xg = concatenate([theta_x, phi_g], axis=3)
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = tf.shape(sigmoid_xg)
    upsample_psi = tf.image.resize(sigmoid_xg, (shape_sigmoid[1]*2, shape_sigmoid[2]*2))
    
    y = multiply([upsample_psi, skip_conn])
    result = Conv2D(inter_shape, (1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn



def attention_unet_model(num_classes=1, input_size=(256, 256, 3), num_filters=64, filter_size=3, dropout=0.1, batch_norm=False):
    inputs = Input(input_size)

    # Encoder
    c1 = conv_block(inputs, num_filters, filter_size, dropout, batch_norm, reg=1e-6)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, num_filters*2, filter_size, dropout, batch_norm, reg=1e-6)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, num_filters*4, filter_size, dropout, batch_norm, reg=1e-5)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = conv_block(p3, num_filters*8, filter_size, dropout, batch_norm, reg=1e-5)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bridge
    bridge = conv_block(p4, num_filters*16, filter_size, dropout, batch_norm, reg=1e-4)

    # Decoder with attention
    g1 = gating_signal(bridge, num_filters*8, batch_norm)
    a1 = attention_block(c4, g1, num_filters*8)
    u6 = Conv2DTranspose(num_filters*8, (2, 2), strides=(2, 2), padding='same')(bridge)
    u6 = concatenate([u6, a1])
    c6 = conv_block(u6, num_filters*8, filter_size, dropout, batch_norm, reg=1e-5)

    g2 = gating_signal(c6, num_filters*4, batch_norm)
    a2 = attention_block(c3, g2, num_filters*4)
    u7 = Conv2DTranspose(num_filters*4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, a2])
    c7 = conv_block(u7, num_filters*4, filter_size, dropout, batch_norm, reg=1e-5)

    g3 = gating_signal(c7, num_filters*2, batch_norm)
    a3 = attention_block(c2, g3, num_filters*2)
    u8 = Conv2DTranspose(num_filters*2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, a3])
    c8 = conv_block(u8, num_filters*2, filter_size, dropout, batch_norm, reg=1e-6)

    g4 = gating_signal(c8, num_filters, batch_norm)
    a4 = attention_block(c1, g4, num_filters)
    u9 = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, a4])
    c9 = conv_block(u9, num_filters, filter_size, dropout, batch_norm, reg=1e-6)

    # Output layer
    output_layer = None
    if num_classes == 1:
        output_layer = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)
    else:
        output_layer = Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[output_layer], name="Attention-U-Net")

    return model


def attention_res_unet_model(num_classes=1, input_size=(256, 256, 3), num_filters=64, filter_size=3, dropout=0.1, batch_norm=False):
    input_layer = Input(input_size)

    # Encoder
    c1 = res_conv_block(input_layer, num_filters, filter_size, dropout, batch_norm, reg=1e-6)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = res_conv_block(p1, num_filters*2, filter_size, dropout, batch_norm, reg=1e-6)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = res_conv_block(p2, num_filters*4, filter_size, dropout, batch_norm, reg=1e-5)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = res_conv_block(p3, num_filters*8, filter_size, dropout, batch_norm, reg=1e-5)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bridge
    bridge = res_conv_block(p4, num_filters*16, filter_size, dropout, batch_norm, reg=1e-4)
    
    # Decoder with attention
    g1 = gating_signal(bridge, num_filters*8, batch_norm)
    a1 = attention_block(c4, g1, num_filters*8)
    u6 = Conv2DTranspose(num_filters*8, (2, 2), strides=(2, 2), padding='same')(bridge)
    u6 = concatenate([u6, a1])
    c6 = res_conv_block(u6, num_filters*8, filter_size, dropout, batch_norm, reg=1e-5)

    g2 = gating_signal(c6, num_filters*4, batch_norm)
    a2 = attention_block(c3, g2, num_filters*4)
    u7 = Conv2DTranspose(num_filters*4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, a2])
    c7 = res_conv_block(u7, num_filters*4, filter_size, dropout, batch_norm, reg=1e-5)

    g3 = gating_signal(c7, num_filters*2, batch_norm)
    a3 = attention_block(c2, g3, num_filters*2)
    u8 = Conv2DTranspose(num_filters*2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, a3])
    c8 = res_conv_block(u8, num_filters*2, filter_size, dropout, batch_norm, reg=1e-6)

    g4 = gating_signal(c8, num_filters, batch_norm)
    a4 = attention_block(c1, g4, num_filters)
    u9 = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, a4])
    c9 = res_conv_block(u9, num_filters, filter_size, dropout, batch_norm, reg=1e-6)

    # Output layer
    output_layer = None
    if num_classes == 1:
        output_layer = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)
    else:
        output_layer = Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[input_layer], outputs=[output_layer], name="Attention-U-Net")

    return model