from tensorflow.keras.backend import set_image_data_format
from tensorflow.keras.layers import Conv3D, concatenate, Conv3DTranspose, Input,  add
from tensorflow.keras.models import Model
import tensorflow as tf
from config import n_filters
from models.helpers import conv_block, fusion_encoder_block
from tensorflow.keras.applications import EfficientNetB0
try:
    import efficientnet_3D.keras as efn
except:
    import efficientnet_3D.tfkeras as efn


def decode_block(inputs, skip, num_filters, project_excite):
    #x = Conv3DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs) #orig
    #x = Conv3DTranspose(n_filters[3], 2, strides=(2, 2, 2), padding='same')(s4) # from unet
    x = Conv3DTranspose(num_filters, 2, strides=(2, 2, 2), padding="same")(inputs) # modified
    expected_shape = tf.constant([3, 32, 32, 32, 1])
    actual_shape = tf.shape(x)[1:]
    print("Expected shape:", expected_shape)
    print("Actual shape:", actual_shape)
    # x = concatenate()([x, skip]) #orig
    x = concatenate([x, skip], axis=-1) #mod
    x = conv_block(x, num_filters, project_excite) #project_excite is also peblock
    return x

def eunet(multi_modal, early_fusion, project_excite, inputs_bmode, inputs_pd, cascade=False):
    iweights = 'imagenet'
    multi_stage_fusion = multi_modal and not early_fusion

    # Input
    if multi_modal and early_fusion:
        #conv, pool, pool_b = fusion_encoder_block(n_filters[0] // 2, multi_modal, project_excite, inputs_bmode, inputs_pd)
        #conv1, pool1, pool1_b = fusion_encoder_block(n_filters[0], multi_stage_fusion, project_excite, conv, None)
        pass
    else:
        encoder = efn.EfficientNetB3(input_tensor=inputs_bmode, weights=None, pooling=None)

    #encoder = efn.EfficientNetB0(input_tensor=inputs_bmode, weights=None, pooling=None)
    try:
        s1 = encoder.get_layer("input_1").output #64
    except:
        s1 = encoder.get_layer("input_2").output #64
    s2 = encoder.get_layer("block2a_expand_activation").output #32
    s3 = encoder.get_layer("block3a_expand_activation").output #16
    s4 = encoder.get_layer("block4a_expand_activation").output #8
    s5 = encoder.get_layer("block6a_expand_activation").output #4
    s6 = encoder.get_layer("top_activation").output #2

    dilated_layers = []
    mode = "cascade"
    depth = 4
    if mode == 'cascade':  
        for i in range(depth):
            s4 = Conv3D(n_filters[4], 3, activation='relu', padding='same', dilation_rate=2**i)(s4)
            dilated_layers.append(s4)
        s4 = add(dilated_layers)
    elif mode == 'parallel':  
        for i in range(depth):
            dilated_layers.append(
                Conv3D(n_filters[4], 3, activation='relu', padding='same', dilation_rate=2**i)(s4)
            )
        s4 = add(dilated_layers)

    d1 = decode_block(s5, s4, n_filters[3], project_excite)
    d2 = decode_block(d1, s3, n_filters[2], project_excite)
    d3 = decode_block(d2, s2, n_filters[1], project_excite)
    d4 = decode_block(d3, s1, n_filters[0], project_excite)
    if cascade:
        return d4
    outputs = Conv3D(1, 1, activation="sigmoid", padding="same")(d4)
    if multi_modal:
        model = Model(inputs=[inputs_bmode, inputs_pd], outputs=[outputs], name="EfficientNetB0_UNET")
    else:
        model = Model(inputs_bmode, outputs, name="EfficientNetB0_UNET")
    return model