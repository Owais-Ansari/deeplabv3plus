import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

from tensorflow.keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Dropout,Convolution2D,GlobalAveragePooling2D, Input,Multiply,GlobalMaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose,concatenate,Reshape,Permute,Activation,UpSampling2D,ZeroPadding2D,Lambda, Dense,DepthwiseConv2D,Concatenate

#from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow.keras.layers import AveragePooling2D, MaxPool2D
from tensorflow.keras import initializers
from tensorflow.keras.regularizers import l2,l1
from tensorflow.keras import constraints
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers import LeakyReLU,PReLU,ReLU
from tensorflow.keras.layers import  Add
import tensorflow.keras.backend as K
from tensorflow import keras

weight_decay = 0.0005



def convolution_block(x, filters, size, strides=(1,1), padding='same', dilation_rate=(1,1),activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding,dilation_rate=dilation_rate,kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def eassp(x, channel = 256 , OS=16):
    #branching for Atrous Spatial Pyramid Pooling
    # Image Feature branch
    if OS == 8:
        atrous_rates = (3, 6, 9)
    else:
        atrous_rates = (6, 12, 18)

    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(channel, (1, 1), padding='same',use_bias=False)(b4)
    b4 = BatchNormalization(epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
      
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    # print(size_before)
    b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
                                                    method='bilinear', align_corners=True))(b4)
    # simple 1x1
    b0 = Conv2D(channel, (1, 1), padding='same', use_bias=False)(x)
    b0 = BatchNormalization( epsilon=1e-5)(b0)
    b0 = Activation('relu')(b0)
    
    
    
    b1 = convolution_block(x, channel//4, (1,1), strides=(1,1), padding='same', dilation_rate =(atrous_rates[0],atrous_rates[0]),activation=True)
    b1 = convolution_block(b1, channel//4, (3,3), strides=(1,1), padding='same', dilation_rate =(atrous_rates[0],atrous_rates[0]),activation=True)
    b1 = convolution_block(b1, channel//4, (3,3), strides=(1,1), padding='same', dilation_rate =(atrous_rates[0],atrous_rates[0]),activation=True)   
    b1 = Conv2D(channel,(1, 1), padding='same', use_bias=False)(b1)
    b1 = BatchNormalization(epsilon=1e-5)(b1)
    b1 = Activation('relu')(b1)
        
    # rate = 12 (24)
    b2 = convolution_block(x, channel//4, (1,1), strides=(1,1), padding='same', dilation_rate =(atrous_rates[1],atrous_rates[1]),activation=True)
    b2 = convolution_block(b2, channel//4, (3,3), strides=(1,1), padding='same', dilation_rate =(atrous_rates[1],atrous_rates[1]),activation=True)
    b2 = convolution_block(b2, channel//4, (3,3), strides=(1,1), padding='same', dilation_rate =(atrous_rates[1],atrous_rates[1]),activation=True)   
    b2 = Conv2D(channel,(1, 1), padding='same', use_bias=False)(b2)
    b2 = BatchNormalization(epsilon=1e-5)(b2)
    b2 = Activation('relu')(b2)
        
    # rate = 18 (36)
    b3 = convolution_block(x, channel//4, (1,1), strides=(1,1), padding='same', dilation_rate =(atrous_rates[2],atrous_rates[2]),activation=True)
    b3 = convolution_block(b3, channel//4, (3,3), strides=(1,1), padding='same', dilation_rate =(atrous_rates[2],atrous_rates[2]),activation=True)
    b3 = convolution_block(b3, channel//4, (3,3), strides=(1,1), padding='same', dilation_rate =(atrous_rates[2],atrous_rates[2]),activation=True)   
    b3 = Conv2D(channel,(1, 1), padding='same', use_bias=False)(b3)
    b3 = BatchNormalization(epsilon=1e-5)(b3)
    b3 = Activation('relu')(b3)
           
    x = Concatenate()([b4, b0, b1, b2, b3])
    x = Conv2D(channel, (1, 1), padding='same',use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    return x


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding,kernel_regularizer=l2(weight_decay), use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x

def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', kernel_regularizer=l2(weight_decay), use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', 
                      kernel_regularizer=l2(weight_decay),
                      use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
                    
                    
                    
                    
                    
                    
def Deeplabv3Plus(weights=None, input_tensor=None, input_shape=(512, 512, 3), classes=2, backbone='efficientnetb0',
              OS=16, alpha=1., activation='softmax'):
    """ Instantiates the Deeplabv3+ architecture
    # Arguments
        weights: one of 'noisy-student' (pre-trained on noisy-student),
            'imagenet' (pre-trained on imagenet) or None (random initialization)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: shape of input image. format HxWxC
            
        classes: number of desired classes. .
            If number of classes not aligned with the weights used, last layer is initialized randomly
        backbone: backbone to use. one of {'efficientnetb0','efficientnetb1','efficientnetb2','efficientnetb3','efficientnetb4',
                         'efficientnetb5','efficientnetb6','efficientnetb7'}
        activation: optional activation to add to the top of the network.
            One of 'softmax', 'sigmoid' or None
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
            Used only for xception backbone.
        
    # Returns
        A Keras model instance.
    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`
    """
    
    if not (backbone in {'efficientnetb0','efficientnetb1','efficientnetb2','efficientnetb3','efficientnetb4',
                         'efficientnetb5','efficientnetb6','efficientnetb7'}):
        raise ValueError('The `backbone` argument should be in '
                         '`efficientnet`  class ')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor
        
    import efficientnet.keras as efn
  #  import efficientnet.tfkeras as efn
    
    if backbone == 'efficientnetb0': 
        encoder = efn.EfficientNetB0(input_shape=input_shape,include_top=False, weights='noisy-student')#'noisy-student')
        
    elif backbone == 'efficientnetb1':
        encoder = efn.EfficientNetB1(input_shape=input_shape,include_top=False, weights='noisy-student')#'imagenet')#
        
    elif backbone == 'efficientnetb2':
        encoder = efn.EfficientNetB2(input_shape=input_shape,include_top=False, weights='noisy-student')#'imagenet')
        
    elif backbone == 'efficientnetb3':
        encoder = efn.EfficientNetB3(input_shape=input_shape,include_top=False, weights='noisy-student')
       
    elif backbone == 'efficientnetb4':
        encoder = efn.EfficientNetB4(input_shape=input_shape,include_top=False, weights='noisy-student')
        
    elif backbone == 'efficientnetb5':
        encoder = efn.EfficientNetB5(input_shape=input_shape,include_top=False, weights='noisy-student')
        
    elif backbone == 'efficientnetb6':
        encoder = efn.EfficientNetB6(input_shape=input_shape,include_top=False, weights='noisy-student')
        
    elif backbone == 'efficientnetb7':
        encoder = efn.EfficientNetB7(input_shape=input_shape,include_top=False, weights='noisy-student')
        
    #encoder.summary()
    input_im = encoder.input
    x = encoder.get_layer('block6a_expand_activation').output
    #x_bottom = encoder.get_layer('top_activation').output
    skip1 =  encoder.get_layer('block3a_expand_activation').output
    #skip2 =  encoder.get_layer('block3a_expand_activation').output
    
    
    if OS == 8:
        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)
    else:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)


      
    #branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
      
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
                                                    method='bilinear', align_corners=True))(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)
    
    
    
    b1 = SepConv_BN(x, 64, 'aspp1',
                        rate=atrous_rates[0], kernel_size=1, depth_activation=True, epsilon=1e-5)
    b1 = SepConv_BN(b1, 64, 'aspp1_1',
                        rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    b1 = SepConv_BN(b1, 64, 'aspp1_2',
                         rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
#          
    b1 = Conv2D(256,(1, 1), padding='same', use_bias=False, name='easpp1_1')(b1)
    b1 = BatchNormalization(name='easpp1_1_BN', epsilon=1e-5)(b1)
    b1 = Activation('relu', name='easpp1_1_activation')(b1)
        
        # rate = 12 (24)
    b2 = SepConv_BN(x, 64, 'assp2',
                        rate=atrous_rates[1], kernel_size=1,depth_activation=True, epsilon=1e-5)
    b2 = SepConv_BN(b2, 64, 'assp2_1',
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    b2 = SepConv_BN(b2, 64, 'assp2_2',
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
#          
    b2 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='eassp2_1')(b2)
    b2 = BatchNormalization(name='eassp2_1_BN', epsilon=1e-5)(b2)
    b2 = Activation('relu', name='eassp2_1_activation')(b2)
        
        # rate = 18 (36)
    b3 = SepConv_BN(x,  64, 'assp3',
                        rate=atrous_rates[2],kernel_size=1, depth_activation=True, epsilon=1e-5)
    b3 = SepConv_BN(b3, 64, 'assp3_1',
                        rate=atrous_rates[2],depth_activation=True, epsilon=1e-5)
    b3 = SepConv_BN(b3, 64, 'assp3_2',
                        rate=atrous_rates[2],depth_activation=True, epsilon=1e-5)
#           
    b3 = Conv2D(256, (1, 1), padding='same', use_bias=False,name='eassp3_1')(b3)
    b3 = BatchNormalization(name='eassp3_1_BN', epsilon=1e-5)(b3)
    b3 = Activation('relu', name='eassp3_1_activation')(b3)

            
    x = Concatenate()([b4, b0, b1, b2, b3])
    x = Dropout(0.1)(x)
    
    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
   
    
# Decoder
    skip_size = tf.keras.backend.int_shape(skip1)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                        skip_size[1:3],
                                                        method='bilinear', align_corners=True))(x)

    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                           use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)
    x = Concatenate()([x, dec_skip1])
    x = Dropout(0.1)(x)


    x = SepConv_BN(x, 64, 'decoder_conv0',kernel_size=1,
                       depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 64, 'decoder_conv0_1',kernel_size=3,
                       depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 64, 'decoder_conv0_2',kernel_size=3,
                       depth_activation=True, epsilon=1e-5)
    
    x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='decoder_conv0_2_1')(x)
    x = BatchNormalization(name='decoder_conv0_2_1_BN', epsilon=1e-5)(x)
    x = Activation('relu', name='decoder_conv0_2_1_activation')(x)
    
    x = SepConv_BN(x, 64, 'decoder_conv1',kernel_size=1,
                       depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 64, 'decoder_conv1_1',kernel_size=3,
                       depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 64, 'decoder_conv1_2',kernel_size=3,
                       depth_activation=True, epsilon=1e-5)
    
    x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='decoder_conv1_2_1')(x)
    x = BatchNormalization(name='decoder_conv1_2_1_BN', epsilon=1e-5)(x)
    x = Activation('relu', name='decoder_conv1_2_1_activation')(x)
    x = Dropout(0.1)(x)

    
    x = Conv2D(classes, (1, 1), padding='same', name='last_layer_befor_acivation')(x)
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                    size_before3[1:3],
                                                    method='bilinear', align_corners=True))(x)                                              
    
    #reshape = Reshape((input_shape[0]*input_shape[1],classes), input_shape = input_shape)(x)
    #x = Permute((1,2))(reshape)
    
    if activation in {'softmax', 'sigmoid'}:
        x = Activation(activation,dtype='float32')(x)

    model = Model(inputs=[input_im],outputs=[x], name='deeplabv3plus')
    if weights is not None:
        model.load_weights(weights)
    return model
   
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# model = Deeplabv3Plus(weights =None,input_shape=(512, 512, 3), classes=2, backbone='efficientnetb0',OS=16, activation='softmax')
# print(model.summary())                    
