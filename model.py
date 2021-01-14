import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers


#V4 Crop
class firstResnetDis(layers.Layer):
    def __init__(self, k, strides=1, ksize = 3):
        super(firstResnetDis, self).__init__()
        # self.leakeyRelu_1 = layers.LeakyReLU()
        self.conv2D_1 = tfa.layers.SpectralNormalization(layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False))
        self.leakeyRelu_2 = layers.LeakyReLU()
        self.conv2D_2 = tfa.layers.SpectralNormalization(layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False))
        self.avg2D = layers.AveragePooling2D()

        self.conv_shortcut = tfa.layers.SpectralNormalization(layers.Conv2D(k,kernel_size=1,strides=1, padding='valid', use_bias=False))
        self.avg2D_shortcut = layers.AveragePooling2D()
        #self.bn_0 = layers.BatchNormalization()

        
        
    def call(self, inputs):

        # x = self.leakeyRelu_1(inputs)
        x = self.conv2D_1(inputs)
        x = self.leakeyRelu_2(x)
        x = self.conv2D_2(x)
        x = self.avg2D(x)
    
        shortcut = self.conv_shortcut(inputs)
        shortcut = self.avg2D_shortcut(shortcut)
        outputs = layers.add([x,shortcut])

        return outputs

class resnetDis(layers.Layer):
    def __init__(self, k, strides=1, ksize = 3):
        super(resnetDis, self).__init__()
        self.leakeyRelu_1 = layers.LeakyReLU()
        self.conv2D_1 = tfa.layers.SpectralNormalization(layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False))
        self.leakeyRelu_2 = layers.LeakyReLU()
        self.conv2D_2 = tfa.layers.SpectralNormalization(layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False))
        self.avg2D = layers.AveragePooling2D()

        self.conv_shortcut = tfa.layers.SpectralNormalization(layers.Conv2D(k,kernel_size=1,strides=1, padding='valid', use_bias=False))
        self.avg2D_shortcut = layers.AveragePooling2D()
        #self.bn_0 = layers.BatchNormalization()

        
        
    def call(self, inputs):

        x = self.leakeyRelu_1(inputs)
        x = self.conv2D_1(x)
        x = self.leakeyRelu_2(x)
        x = self.conv2D_2(x)
        x = self.avg2D(x)
    
        shortcut = self.conv_shortcut(inputs)
        shortcut = self.avg2D_shortcut(shortcut)
        outputs = layers.add([x,shortcut])

        return outputs


class firstResnetEN(layers.Layer):
    def __init__(self, k, strides=1, ksize = 3):
        super(firstResnetEN, self).__init__()
        # self.leakeyRelu_1 = layers.LeakyReLU()
        self.conv2D_1 = layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False)
        self.leakeyRelu_2 = layers.LeakyReLU()
        self.conv2D_2 = layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False)
        self.avg2D = layers.AveragePooling2D()

        self.conv_shortcut = layers.Conv2D(k,kernel_size=1,strides=1, padding='valid', use_bias=False)
        self.avg2D_shortcut = layers.AveragePooling2D()
        #self.bn_0 = layers.BatchNormalization()

        
        
    def call(self, inputs):

        # x = self.leakeyRelu_1(inputs)
        x = self.conv2D_1(inputs)
        x = self.leakeyRelu_2(x)
        x = self.conv2D_2(x)
        x = self.avg2D(x)
    
        shortcut = self.conv_shortcut(inputs)
        shortcut = self.avg2D_shortcut(shortcut)
        outputs = layers.add([x,shortcut])

        return outputs

class resnetEN(layers.Layer):
    def __init__(self, k, strides=1, ksize = 3):
        super(resnetEN, self).__init__()
        self.leakeyRelu_1 = layers.LeakyReLU()
        self.conv2D_1 = layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False)
        self.leakeyRelu_2 = layers.LeakyReLU()
        self.conv2D_2 = layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False)
        self.avg2D = layers.AveragePooling2D()

        self.conv_shortcut = layers.Conv2D(k,kernel_size=1,strides=1, padding='valid', use_bias=False)
        self.avg2D_shortcut = layers.AveragePooling2D()
        #self.bn_0 = layers.BatchNormalization()

        
        
    def call(self, inputs):

        x = self.leakeyRelu_1(inputs)
        x = self.conv2D_1(x)
        x = self.leakeyRelu_2(x)
        x = self.conv2D_2(x)
        x = self.avg2D(x)
    
        shortcut = self.conv_shortcut(inputs)
        shortcut = self.avg2D_shortcut(shortcut)
        outputs = layers.add([x,shortcut])

        return outputs

class resnetDE(layers.Layer):
    def __init__(self, k, strides=1, ksize = 3):
        super(resnetDE, self).__init__()
        self.bn_1 = layers.BatchNormalization()
        self.leakeyRelu_1 = layers.LeakyReLU()
        self.ups2D = layers.UpSampling2D()
        self.conv2D_1 = layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False)

        self.bn_2 = layers.BatchNormalization()
        self.leakeyRelu_2 = layers.LeakyReLU()
        self.conv2D_2 = layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False)

        self.ups2D_shortcut = layers.UpSampling2D()
        self.conv_shortcut = layers.Conv2D(k,kernel_size=1,strides=1, padding='valid', use_bias=False)

    def call(self, inputs):

        x = self.bn_1(inputs)
        x = self.leakeyRelu_1(x)
        x = self.ups2D(x)
        x = self.conv2D_1(x)

        x = self.bn_2(x)
        x = self.leakeyRelu_2(x)
        x = self.conv2D_2(x)
    
        shortcut = self.ups2D_shortcut(inputs)
        shortcut = self.conv_shortcut(shortcut)
        outputs = layers.add([x,shortcut])

        return outputs

class AE(tf.keras.Model):
    def __init__(self, k):
        super(AE, self).__init__()

        #256x256 -> 128x128
        self.enBlock1 = firstResnetEN(k)
        #128x128 -> 64x64
        self.enBlock2 = resnetEN(2*k)
        #64x64 -> 32x32
        self.enBlock3 = resnetEN(4*k)
        #32x32 -> 16x16
        self.enBlock4 = resnetEN(8*k)
        #16x16 -> 8x8
        self.enBlock5 = resnetEN(8*k)               
        # #8x8 -> 4x4
        self.enBlock6 = resnetEN(16*k)
        self.enBlockLeakyReLU = layers.LeakyReLU() 

        self.averagepooling = layers.AveragePooling2D(pool_size = (4, 4)) # -> 1x1x512

        self.pooling_LeakyRelu = layers.LeakyReLU()
        
        self.encoderOutputlayer = layers.Dense(16*k) # -> 1024x1
        self.encoderOutputlayerLeakyReLU = layers.LeakyReLU()

        self.decoderInputlayer = layers.Dense(4*4*16*k) 
        self.decoderInputlayerLeakyReLU = layers.LeakyReLU()

        self.reshapeLayer = layers.Reshape((4, 4, 16*k))
        # #4x4 -> 8x8
        self.deBlock1 = resnetDE(16*k)                
        # #8x8 -> 16x16
        self.deBlock2 = resnetDE(8*k)    
        #16x16 -> 32x32
        self.deBlock3 = resnetDE(8*k)    
        #32x32 -> 64x64
        self.deBlock4 = resnetDE(4*k)    
        #64x64 -> 128x128
        self.deBlock5 = resnetDE(2*k)    
        #128x128 -> 256x256
        self.deBlock6 = resnetDE(k)    

        self.deBatchNormalization = layers.BatchNormalization()
        self.deLeakyReLU = layers.LeakyReLU()
        #256x256xk -> 256x256x3
        self.deConv2D = layers.Conv2D(3, kernel_size= 3, strides=1, padding = 'same', use_bias = False)
        self.tanh = layers.Activation('tanh')

    
    def call(self, input_tensor):

        x = self.enBlock1(input_tensor)
        x = self.enBlock2(x)
        x = self.enBlock3(x)
        x = self.enBlock4(x)
        x = self.enBlock5(x)
        x = self.enBlock6(x)
        x = self.enBlockLeakyReLU(x)
        # x = self.averagepooling(x)
        # x = self.pooling_LeakyRelu(x)
        x = layers.Flatten()(x)
        x = self.encoderOutputlayer(x)
        x = self.encoderOutputlayerLeakyReLU(x)
        x = self.decoderInputlayer(x)
        x = self.decoderInputlayerLeakyReLU(x)
        x = self.reshapeLayer(x)

        x = self.deBlock1(x)
        x = self.deBlock2(x)
        x = self.deBlock3(x)
        x = self.deBlock4(x)
        x = self.deBlock5(x)
        x = self.deBlock6(x)

        x = self.deBatchNormalization(x)
        x = self.deLeakyReLU(x)
        x = self.deConv2D(x)
        x = self.tanh(x)

        return x
class discriminatorOnCode(tf.keras.Model):
    def __init__(self, ch = 64):
        super(discriminator, self).__init__()
        self.D1 = layers.Dense(ch*8)
        self.D2 = layers.Dense(ch*16)
        self.D3 = layers.Dense(ch*32)
        self.D4 = layers.Dense(ch*1)
        self.out = layers.Activation("sigmoid")
    def call(self, inputs):
        d = self.D1(inputs)
        d = self.D2(d)
        d = self.D3(d)
        d = self.D4(d)
        d = self.out(d)
        return d
class discriminator(tf.keras.Model):
    def __init__(self, ch = 64):
        super(discriminator, self).__init__()
        self.res0 = firstResnetDis(ch, ksize=3)
        self.res1 = resnetDis(ch*2, ksize=3)
        self.res2 = resnetDis(ch*4, ksize=3)
        self.res3 = resnetDis(ch*8, ksize=3)
        self.res4 = resnetDis(ch*8, ksize=3)
        # self.res5 = resnetDis(ch*16, ksize=3)
        # self.averagepooling = layers.AveragePooling2D(pool_size = (4, 4))
        self.fullyConnect = tfa.layers.SpectralNormalization(layers.Dense(1))
        # self.conv4 = tfa.layers.SpectralNormalization(layers.Conv2D(1, kernel_size=4, strides=1, padding='valid', use_bias=False))

    def call(self, inputs):

        d = self.res0(inputs)
        d = self.res1(d)
        d = self.res2(d)
        d = self.res3(d)
        d = self.res4(d)
        # d = self.res5(d)
        d = layers.ReLU()(d)
        # d = self.averagepooling(d)
        d = self.fullyConnect(d)
        d = tf.math.sigmoid(d)
        return d
# V4
# class firstResnetDis(layers.Layer):
#     def __init__(self, k, strides=1, ksize = 3):
#         super(firstResnetDis, self).__init__()
#         # self.leakeyRelu_1 = layers.LeakyReLU()
#         self.conv2D_1 = tfa.layers.SpectralNormalization(layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False))
#         self.leakeyRelu_2 = layers.LeakyReLU()
#         self.conv2D_2 = tfa.layers.SpectralNormalization(layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False))
#         self.avg2D = layers.AveragePooling2D()

#         self.conv_shortcut = tfa.layers.SpectralNormalization(layers.Conv2D(k,kernel_size=1,strides=1, padding='valid', use_bias=False))
#         self.avg2D_shortcut = layers.AveragePooling2D()
#         #self.bn_0 = layers.BatchNormalization()

        
        
#     def call(self, inputs):

#         # x = self.leakeyRelu_1(inputs)
#         x = self.conv2D_1(inputs)
#         x = self.leakeyRelu_2(x)
#         x = self.conv2D_2(x)
#         x = self.avg2D(x)
    
#         shortcut = self.conv_shortcut(inputs)
#         shortcut = self.avg2D_shortcut(shortcut)
#         outputs = layers.add([x,shortcut])

#         return outputs

# class resnetDis(layers.Layer):
#     def __init__(self, k, strides=1, ksize = 3):
#         super(resnetDis, self).__init__()
#         self.leakeyRelu_1 = layers.LeakyReLU()
#         self.conv2D_1 = tfa.layers.SpectralNormalization(layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False))
#         self.leakeyRelu_2 = layers.LeakyReLU()
#         self.conv2D_2 = tfa.layers.SpectralNormalization(layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False))
#         self.avg2D = layers.AveragePooling2D()

#         self.conv_shortcut = tfa.layers.SpectralNormalization(layers.Conv2D(k,kernel_size=1,strides=1, padding='valid', use_bias=False))
#         self.avg2D_shortcut = layers.AveragePooling2D()
#         #self.bn_0 = layers.BatchNormalization()

        
        
#     def call(self, inputs):

#         x = self.leakeyRelu_1(inputs)
#         x = self.conv2D_1(x)
#         x = self.leakeyRelu_2(x)
#         x = self.conv2D_2(x)
#         x = self.avg2D(x)
    
#         shortcut = self.conv_shortcut(inputs)
#         shortcut = self.avg2D_shortcut(shortcut)
#         outputs = layers.add([x,shortcut])

#         return outputs


# class firstResnetEN(layers.Layer):
#     def __init__(self, k, strides=1, ksize = 3):
#         super(firstResnetEN, self).__init__()
#         # self.leakeyRelu_1 = layers.LeakyReLU()
#         self.conv2D_1 = layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False)
#         self.leakeyRelu_2 = layers.LeakyReLU()
#         self.conv2D_2 = layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False)
#         self.avg2D = layers.AveragePooling2D()

#         self.conv_shortcut = layers.Conv2D(k,kernel_size=1,strides=1, padding='valid', use_bias=False)
#         self.avg2D_shortcut = layers.AveragePooling2D()
#         #self.bn_0 = layers.BatchNormalization()

        
        
#     def call(self, inputs):

#         # x = self.leakeyRelu_1(inputs)
#         x = self.conv2D_1(inputs)
#         x = self.leakeyRelu_2(x)
#         x = self.conv2D_2(x)
#         x = self.avg2D(x)
    
#         shortcut = self.conv_shortcut(inputs)
#         shortcut = self.avg2D_shortcut(shortcut)
#         outputs = layers.add([x,shortcut])

#         return outputs

# class resnetEN(layers.Layer):
#     def __init__(self, k, strides=1, ksize = 3):
#         super(resnetEN, self).__init__()
#         self.leakeyRelu_1 = layers.LeakyReLU()
#         self.conv2D_1 = layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False)
#         self.leakeyRelu_2 = layers.LeakyReLU()
#         self.conv2D_2 = layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False)
#         self.avg2D = layers.AveragePooling2D()

#         self.conv_shortcut = layers.Conv2D(k,kernel_size=1,strides=1, padding='valid', use_bias=False)
#         self.avg2D_shortcut = layers.AveragePooling2D()
#         #self.bn_0 = layers.BatchNormalization()

        
        
#     def call(self, inputs):

#         x = self.leakeyRelu_1(inputs)
#         x = self.conv2D_1(x)
#         x = self.leakeyRelu_2(x)
#         x = self.conv2D_2(x)
#         x = self.avg2D(x)
    
#         shortcut = self.conv_shortcut(inputs)
#         shortcut = self.avg2D_shortcut(shortcut)
#         outputs = layers.add([x,shortcut])

#         return outputs

# class resnetDE(layers.Layer):
#     def __init__(self, k, strides=1, ksize = 3):
#         super(resnetDE, self).__init__()
#         self.bn_1 = layers.BatchNormalization()
#         self.leakeyRelu_1 = layers.LeakyReLU()
#         self.ups2D = layers.UpSampling2D()
#         self.conv2D_1 = layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False)

#         self.bn_2 = layers.BatchNormalization()
#         self.leakeyRelu_2 = layers.LeakyReLU()
#         self.conv2D_2 = layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False)

#         self.ups2D_shortcut = layers.UpSampling2D()
#         self.conv_shortcut = layers.Conv2D(k,kernel_size=1,strides=1, padding='valid', use_bias=False)

#     def call(self, inputs):

#         x = self.bn_1(inputs)
#         x = self.leakeyRelu_1(x)
#         x = self.ups2D(x)
#         x = self.conv2D_1(x)

#         x = self.bn_2(x)
#         x = self.leakeyRelu_2(x)
#         x = self.conv2D_2(x)
    
#         shortcut = self.ups2D_shortcut(inputs)
#         shortcut = self.conv_shortcut(shortcut)
#         outputs = layers.add([x,shortcut])

#         return outputs

# class AE(tf.keras.Model):
#     def __init__(self, k):
#         super(AE, self).__init__()

#         #256x256 -> 128x128
#         self.enBlock1 = firstResnetEN(k)
#         #128x128 -> 64x64
#         self.enBlock2 = resnetEN(2*k)
#         #64x64 -> 32x32
#         self.enBlock3 = resnetEN(4*k)
#         #32x32 -> 16x16
#         self.enBlock4 = resnetEN(8*k)
#         #16x16 -> 8x8
#         self.enBlock5 = resnetEN(8*k)               
#         # #8x8 -> 4x4
#         self.enBlock6 = resnetEN(16*k)
#         # self.enBlockLeakyReLU = layers.LeakyReLU() 

#         # self.averagepooling = layers.AveragePooling2D(pool_size = (4, 4)) # -> 1x1x512  

#         # self.pooling_LeakyRelu = layers.LeakyReLU()
        
#         # self.encoderOutputlayer = layers.Dense(32*k) # -> 512x1
#         # self.encoderOutputlayerLeakyReLU = layers.LeakyReLU()

#         # self.reshapeLayer = layers.Reshape((4, 4, 2*k))
#         #4x4 -> 8x8
#         self.deBlock1 = resnetDE(16*k)                
#         # #8x8 -> 16x16
#         self.deBlock2 = resnetDE(8*k)    
#         #16x16 -> 32x32
#         self.deBlock3 = resnetDE(8*k)    
#         #32x32 -> 64x64
#         self.deBlock4 = resnetDE(4*k)    
#         #64x64 -> 128x128
#         self.deBlock5 = resnetDE(2*k)    
#         #128x128 -> 256x256
#         self.deBlock6 = resnetDE(k)    

#         self.deBatchNormalization = layers.BatchNormalization()
#         self.deLeakyReLU = layers.LeakyReLU()
#         #256x256xk -> 256x256x3
#         self.deConv2D = layers.Conv2D(3, kernel_size= 3, strides=1, padding = 'same', use_bias = False)
#         self.tanh = layers.Activation('tanh')

    
#     def call(self, input_tensor):

#         x = self.enBlock1(input_tensor)
#         x = self.enBlock2(x)
#         x = self.enBlock3(x)
#         x = self.enBlock4(x)
#         x = self.enBlock5(x)
#         x = self.enBlock6(x)
#         x = self.enBlockLeakyReLU(x)
#         # x = self.averagepooling(x)
#         # x = self.pooling_LeakyRelu(x)
#         # x = layers.Flatten()(x)
#         # x = self.encoderOutputlayer(x)
#         # x = self.encoderOutputlayerLeakyReLU(x)
#         # x = self.reshapeLayer(x)

#         x = self.deBlock1(x)
#         x = self.deBlock2(x)
#         x = self.deBlock3(x)
#         x = self.deBlock4(x)
#         x = self.deBlock5(x)
#         x = self.deBlock6(x)

#         x = self.deBatchNormalization(x)
#         x = self.deLeakyReLU(x)
#         x = self.deConv2D(x)
#         x = self.tanh(x)

#         return x
# class discriminator(tf.keras.Model):
#     def __init__(self, ch = 64):
#         super(discriminator, self).__init__()
#         self.res0 = firstResnetDis(ch, ksize=3)
#         self.res1 = resnetDis(ch*2, ksize=3)
#         self.res2 = resnetDis(ch*4, ksize=3)
#         self.res3 = resnetDis(ch*8, ksize=3)
#         self.res4 = resnetDis(ch*8, ksize=3)
#         # self.res5 = resnetDis(ch*16, ksize=3)
#         # self.averagepooling = layers.AveragePooling2D(pool_size = (4, 4))
#         self.fullyConnect = tfa.layers.SpectralNormalization(layers.Dense(1))
#         # self.conv4 = tfa.layers.SpectralNormalization(layers.Conv2D(1, kernel_size=4, strides=1, padding='valid', use_bias=False))

#     def call(self, inputs):

#         d = self.res0(inputs)
#         d = self.res1(d)
#         d = self.res2(d)
#         d = self.res3(d)
#         d = self.res4(d)
#         # d = self.res5(d)
#         d = layers.ReLU()(d)
#         # d = self.averagepooling(d)
#         d = self.fullyConnect(d)
#         d = tf.math.sigmoid(d)
#         return d
# #
# class firstResnetEN(layers.Layer):
#         def __init__(self, k, strides=1, ksize = 3):
#             super(firstResnetEN, self).__init__()
#             # self.leakeyRelu_1 = layers.LeakyReLU()
#             self.conv2D_1 = layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False)
#             self.leakeyRelu_2 = layers.LeakyReLU()
#             self.conv2D_2 = layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False)
#             self.avg2D = layers.AveragePooling2D()

#             self.conv_shortcut = layers.Conv2D(k,kernel_size=1,strides=1, padding='valid', use_bias=False)
#             self.avg2D_shortcut = layers.AveragePooling2D()
#             #self.bn_0 = layers.BatchNormalization()

            
            
#         def call(self, inputs):

#             # x = self.leakeyRelu_1(inputs)
#             x = self.conv2D_1(inputs)
#             x = self.leakeyRelu_2(x)
#             x = self.conv2D_2(x)
#             x = self.avg2D(x)
        
#             shortcut = self.conv_shortcut(inputs)
#             shortcut = self.avg2D_shortcut(shortcut)
#             outputs = layers.add([x,shortcut])

#             return outputs

# class resnetEN(layers.Layer):
#     def __init__(self, k, strides=1, ksize = 3):
#         super(resnetEN, self).__init__()
#         self.leakeyRelu_1 = layers.LeakyReLU()
#         self.conv2D_1 = layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False)
#         self.leakeyRelu_2 = layers.LeakyReLU()
#         self.conv2D_2 = layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False)
#         self.avg2D = layers.AveragePooling2D()

#         self.conv_shortcut = layers.Conv2D(k,kernel_size=1,strides=1, padding='valid', use_bias=False)
#         self.avg2D_shortcut = layers.AveragePooling2D()
#         #self.bn_0 = layers.BatchNormalization()

        
        
#     def call(self, inputs):

#         x = self.leakeyRelu_1(inputs)
#         x = self.conv2D_1(x)
#         x = self.leakeyRelu_2(x)
#         x = self.conv2D_2(x)
#         x = self.avg2D(x)
    
#         shortcut = self.conv_shortcut(inputs)
#         shortcut = self.avg2D_shortcut(shortcut)
#         outputs = layers.add([x,shortcut])

#         return outputs

# class resnetDE(layers.Layer):
#     def __init__(self, k, strides=1, ksize = 3):
#         super(resnetDE, self).__init__()
#         self.bn_1 = layers.BatchNormalization()
#         self.leakeyRelu_1 = layers.LeakyReLU()
#         self.ups2D = layers.UpSampling2D()
#         self.conv2D_1 = layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False)

#         self.bn_2 = layers.BatchNormalization()
#         self.leakeyRelu_2 = layers.LeakyReLU()
#         self.conv2D_2 = layers.Conv2D(k, kernel_size= 3, strides=1, padding = 'same', use_bias = False)

#         self.ups2D_shortcut = layers.UpSampling2D()
#         self.conv_shortcut = layers.Conv2D(k,kernel_size=1,strides=1, padding='valid', use_bias=False)

#     def call(self, inputs):

#         x = self.bn_1(inputs)
#         x = self.leakeyRelu_1(x)
#         x = self.ups2D(x)
#         x = self.conv2D_1(x)

#         x = self.bn_2(x)
#         x = self.leakeyRelu_2(x)
#         x = self.conv2D_2(x)
    
#         shortcut = self.ups2D_shortcut(inputs)
#         shortcut = self.conv_shortcut(shortcut)
#         outputs = layers.add([x,shortcut])

#         return outputs

# class AE(tf.keras.Model):
#     def __init__(self, k):
#         super(AE, self).__init__()

#         #256x256 -> 128x128
#         self.enBlock1 = firstResnetEN(k)
#         #128x128 -> 64x64
#         self.enBlock2 = resnetEN(2*k)
#         #64x64 -> 32x32
#         self.enBlock3 = resnetEN(4*k)
#         #32x32 -> 16x16
#         self.enBlock4 = resnetEN(8*k)
#         #16x16 -> 8x8
#         self.enBlock5 = resnetEN(8*k)               
#         #8x8 -> 4x4
#         self.enBlock6 = resnetEN(16*k)

#         self.enBlockLeakyReLU = layers.LeakyReLU() # -> 4x4x512
        
#         self.encoderOutputlayer = layers.Dense(4*4*16*k) # -> 8192x1
#         self.encoderOutputlayerLeakyReLU = layers.LeakyReLU()

#         self.reshapeLayer = layers.Reshape((4, 4, 16*k))
#         #4x4 -> 8x8
#         self.deBlock1 = resnetDE(16*k)                
#         #8x8 -> 16x16
#         self.deBlock2 = resnetDE(8*k)    
#         #16x16 -> 32x32
#         self.deBlock3 = resnetDE(8*k)    
#         #32x32 -> 64x64
#         self.deBlock4 = resnetDE(4*k)    
#         #64x64 -> 128x128
#         self.deBlock5 = resnetDE(2*k)    
#         #128x128 -> 256x256
#         self.deBlock6 = resnetDE(k)    

#         self.deBatchNormalization = layers.BatchNormalization()
#         self.deLeakyReLU = layers.LeakyReLU()
#         #256x256xk -> 256x256x3
#         self.deConv2D = layers.Conv2D(3, kernel_size= 3, strides=1, padding = 'same', use_bias = False)
#         self.tanh = layers.Activation('tanh')

    
#     def call(self, input_tensor):

#         x = self.enBlock1(input_tensor)
#         x = self.enBlock2(x)
#         x = self.enBlock3(x)
#         x = self.enBlock4(x)
#         x = self.enBlock5(x)
#         x = self.enBlock6(x)
#         x = self.enBlockLeakyReLU(x)
#         x = layers.Flatten()(x)
#         x = self.encoderOutputlayer(x)
#         x = self.encoderOutputlayerLeakyReLU(x)
#         x = self.reshapeLayer(x)

#         x = self.deBlock1(x)
#         x = self.deBlock2(x)
#         x = self.deBlock3(x)
#         x = self.deBlock4(x)
#         x = self.deBlock5(x)
#         x = self.deBlock6(x)

#         x = self.deBatchNormalization(x)
#         x = self.deLeakyReLU(x)
#         x = self.deConv2D(x)
#         x = self.tanh(x)

#         return x