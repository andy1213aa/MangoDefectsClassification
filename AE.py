import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
def main():
    batchsize = 16
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
            
            self.encoderOutputlayer = layers.Dense(32*k) # -> 1024x1
            self.encoderOutputlayerLeakyReLU = layers.LeakyReLU()

            self.reshapeLayer = layers.Reshape((4, 4, 2*k))
            #4x4 -> 8x8
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
            x = self.averagepooling(x)
            x = self.pooling_LeakyRelu(x)
            x = layers.Flatten()(x)
            x = self.encoderOutputlayer(x)
            x = self.encoderOutputlayerLeakyReLU(x)
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
            d = layers.Activation('tanh')(d)
            return d

    def generateData(tfRecordDir):
        def _parse_function(example_proto):
            features = tf.io.parse_single_example(
                example_proto,
                features={
                    "ID": tf.io.FixedLenFeature([], tf.float32),
                    "width": tf.io.FixedLenFeature([], tf.float32),
                    "height": tf.io.FixedLenFeature([], tf.float32),
                    'data_raw': tf.io.FixedLenFeature([], tf.string)
                }
            )
            # ID = features['ID']
            width = features['width']
            height = features['height']
            data = features['data_raw']
            data = tf.io.decode_raw(data, tf.uint8)
            data = tf.cast(data, tf.float32)
            data = tf.reshape(data, [width, height, 3])
            data = tf.image.resize(data, [256, 256], method='bicubic')
            data = data/127.5 -1

            return data

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data = tf.data.TFRecordDataset(tfRecordDir)
        data = data.map(_parse_function, num_parallel_calls=AUTOTUNE)

        training_batch = data.batch(16, drop_remainder = True)
        training_batch = training_batch.prefetch(buffer_size = AUTOTUNE)
        return training_batch
 

    def combineImages(images, col=4, row=4):
        images = (images+1)/2
        images = images.numpy()
        b, h, w, _ = images.shape
        imagesCombine = np.zeros(shape=(h*col, w*row, 3))
        for y in range(col):
            for x in range(row):
                imagesCombine[y*h:(y+1)*h, x*w:(x+1)*w] = images[x+y*row]
        return imagesCombine
   
   
    # 指定TFrecords路径，得到training iterator。
    # train_tfrecords = r'E:\NTNU2-1\imageRecognition\finalProject\trainingData.tfrecords'
    def AELossFunction(imgReal, imgFake, logitFake):
        MSE = tf.reduce_mean(keras.losses.MSE(imgReal, imgFake))
        # L2_loss= (1/batchsize)*tf.norm(tensor= (imgReal-imgFake), ord='euclidean')
        feature_loss = tf.reduce_mean(keras.losses.MSE(VGG19Subnet(tf.image.resize(imgReal, [224, 224], method='bicubic')), VGG19Subnet(tf.image.resize(imgFake, [224, 224], method='bicubic'))))
        # feature_loss = F_coefficient * tf.norm(tensor= VGG19Subnet(tf.image.resize(imgReal, [224, 224], method='bicubic')) - VGG19Subnet(tf.image.resize(imgFake, [224, 224], method='bicubic')), ord='euclidean')
        adversarial_loss = -tf.reduce_mean(logitFake)
        return MSE, feature_loss, adversarial_loss
    def DISLossFunction(logitReal, logitFake):
        
        realLogitLoss = -tf.reduce_mean(logitReal)
        fakeLogitLoss = tf.reduce_mean(logitFake)
        return realLogitLoss, fakeLogitLoss

    @tf.function()
    def trainModelDis(train_data):
        print('DIS SIDE EFFECT')
        with tf.GradientTape() as t:
            imgFake = modelAE(train_data,  training = False)
            logitReal = modelDIS(train_data)
            logitFake = modelDIS(imgFake)
            disRealLoss, disFakeLoss = DISLossFunction(logitReal, logitFake)
            totlaLoss = disRealLoss+disFakeLoss
        DISGrade = t.gradient(totlaLoss, modelDIS.trainable_variables)
        modelDIS_optimizer.apply_gradients(zip(DISGrade, modelDIS.trainable_variables))
        return disRealLoss, disFakeLoss


    @tf.function()
    def trainModelAE(train_data):
        print('AE SIDE EFFECT')    
        with tf.GradientTape() as tape:
            imgFake = modelAE(train_data,  training = True)
            logitFake = modelDIS(imgFake, training = False)
            MSE_loss, F_loss, adv_loss = AELossFunction(train_data, imgFake, logitFake)
            totalLoss = F_loss + MSE_loss + adv_loss 
        AEGrade = tape.gradient(totalLoss, modelAE.trainable_variables)
        modelAE_optimizer.apply_gradients(zip(AEGrade, modelAE.trainable_variables))
        return  MSE_loss, F_loss, adv_loss

    version = "V4"
    modelAE = AE(32)
    modelAE_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1 = 0.5, beta_2=0.99)
    
    if version == "V4":
        modelDIS = discriminator(32)
        modelDIS_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1 = 0.5, beta_2=0.99)  
    elif version == "V5":
        modelDIS = discriminatorOnCode(64)
        modelDIS_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1 = 0.5, beta_2=0.99)  

    VGG19 = tf.keras.applications.VGG19()
    VGG19Subnet = keras.models.Model(inputs = VGG19.layers[0].input, outputs = VGG19.layers[2].output)
    del VGG19
 
    # tfRecordDir = '/home/csun001/Nyx/NyxDataSet/Nyx_tfrecords/NyxDataSet256_256.tfrecords'    
    tfRecordDir = '/home/csun001/finalProject/cropTrainingData.tfrecords'    
    # each data batch
    train_data = generateData(tfRecordDir)

    summary_writer = tf.summary.create_file_writer('/home/csun001/finalProject/log/' + version)
    epochs = 15
    for epoch in range(epochs):
        for _, batch in enumerate(train_data):
            pixelwiseLoss, perceptualLoss, adversarialLoss = trainModelAE(batch)
            disRealLoss, disFakeLoss = trainModelDis(batch)
            with summary_writer.as_default():
                tf.summary.scalar('pixelwiseLoss', pixelwiseLoss, modelAE_optimizer.iterations)
                tf.summary.scalar('perceptualLoss', perceptualLoss, modelAE_optimizer.iterations)
                tf.summary.scalar('adversarialLoss', adversarialLoss, modelAE_optimizer.iterations)
                tf.summary.scalar('disRealLoss', disRealLoss, modelAE_optimizer.iterations)
                tf.summary.scalar('disFakeLoss', disFakeLoss, modelAE_optimizer.iterations)
                # tf.summary.scalar('VGGLoss', VGGLoss, modelAE_optimizer.iterations)
            if modelAE_optimizer.iterations % 100 == 0:
                fakeImage = modelAE(batch, training=False)
                rawImage = combineImages(batch)
                fakeImage = combineImages(fakeImage)
                with summary_writer.as_default():
                    tf.summary.image('rawImage', [rawImage], step=modelAE_optimizer.iterations)
                    tf.summary.image('fakeImage', [fakeImage], step=modelAE_optimizer.iterations)
                modelAE.save_weights('/home/csun001/finalProject/log/V4/modelAE/trained_ckpt')
                modelAE.save_weights('/home/csun001/finalProject/log/V4/modelDIS/trained_ckpt')
        print(f'Epoch: {epoch:4} pixelwiseLoss:{pixelwiseLoss:4.2f} perceptualLoss:{perceptualLoss:4.2f} adversarialLoss{adversarialLoss:4.5f} disLoss{disRealLoss+disFakeLoss:4.5f}')

            
main()