load = -1
ss = 0.02 #Style loss multiplier

from PIL import Image
from math import floor, log2
import numpy as np
import time
from functools import partial
from random import random
import os

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K

#Lambdas
def AdaIN(x):
    #Normalize x[0]
    mean = K.mean(x[0], axis = [1, 2], keepdims = True)
    std = K.std(x[0], axis = [1, 2], keepdims = True) + 1e-7
    y = (x[0] - mean) / std

    #Reshape gamma and beta
    pool_shape = [-1, 1, 1, y.shape[-1]]
    g = K.reshape(x[1], pool_shape)
    b = K.reshape(x[2], pool_shape)

    #Multiply by x[1] (GAMMA) and add x[2] (BETA)
    return y * g + b

from tensorflow.keras.applications.vgg19 import VGG19

def saveModel(model, name, num):
    json = model.to_json()
    with open("Models/"+name+".json", "w") as json_file:
        json_file.write(json)

    model.save_weights("Models/"+name+"_"+str(num)+".h5")

def loadModel(name, num):

    file = open("Models/"+name+".json", 'r')
    json = file.read()
    file.close()

    mod = model_from_json(json)
    mod.load_weights("Models/"+name+"_"+str(num)+".h5")

    return mod




#Get encoder and representations
vgg = VGG19(include_top=False, weights='imagenet')

from datagen import dataGenerator, printProgressBar

for i in range(len(vgg.layers)):
    if vgg.layers[i].name == 'block4_conv1':
        encoder = Model(vgg.input, vgg.layers[i].output)
    if vgg.layers[i].name == 'block1_conv1':
        rep1 = Model(vgg.input, vgg.layers[i].output)
    if vgg.layers[i].name == 'block2_conv1':
        rep2 = Model(vgg.input, vgg.layers[i].output)
    if vgg.layers[i].name == 'block3_conv1':
        rep3 = Model(vgg.input, vgg.layers[i].output)
    if vgg.layers[i].name == 'block4_conv1':
        rep4 = Model(vgg.input, vgg.layers[i].output)

style_encoder = model_from_json(encoder.to_json())
style_encoder.set_weights(encoder.get_weights())


#Build decoder and style encoder
if load < 0:
    decoder = Sequential()
    decoder.add(Conv2D(256, 3, activation='relu', padding='same', input_shape = [None, None, 512]))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(256, 3, activation='relu', padding='same'))
    decoder.add(Conv2D(256, 3, activation='relu', padding='same'))
    decoder.add(Conv2D(256, 3, activation='relu', padding='same'))
    decoder.add(Conv2D(128, 3, activation='relu', padding='same'))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(128, 3, activation='relu', padding='same'))
    decoder.add(Conv2D(64, 3, activation='relu', padding='same'))
    decoder.add(UpSampling2D())
    decoder.add(Conv2D(64, 3, activation='relu', padding='same'))
    decoder.add(Conv2D(3, 3, activation='sigmoid', padding='same'))

    inp = Input([None, None, 3])
    x = style_encoder(inp)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation = 'relu')(x)
    m = Dense(512)(x)
    s = Dense(512)(x)

    sencoder = Model(inp, [m, s])
else:
    decoder = loadModel("dec", load)
    sencoder = loadModel("sen", load)



#Loss functions
def content_loss(y_true, y_pred, y_rep, sample_weight = None):
    #Normalize prediction
    mean = K.mean(y_pred, axis = [1, 2], keepdims = True)
    std = K.std(y_pred, axis = [1, 2], keepdims = True) + 1e-7
    yp = (y_pred - mean) / std

    #Normalize representation
    mean = K.mean(y_rep, axis = [1, 2], keepdims = True)
    std = K.std(y_rep, axis = [1, 2], keepdims = True) + 1e-7
    yr = (y_rep - mean) / std

    #Find difference in normalized representations
    return K.mean(K.square(yp - yr))

def style_loss(y_true, y_pred, y_rep, weight = 1, sample_weight = None):
    mean_loss = K.mean(y_pred, axis = [1, 2]) - K.mean(y_rep, axis = [1, 2])
    mean_loss = K.square(mean_loss)
    std_loss = K.std(y_pred, axis = [1, 2]) - K.std(y_rep, axis = [1, 2])
    std_loss = K.square(std_loss)
    return K.sum(K.mean(mean_loss + std_loss, axis = 0)) * weight


#Training model
content_image = Input([None, None, 3])
style_image = Input([None, None, 3])

content_rep = encoder(content_image)
[smean, sstd] = sencoder(style_image)
full_rep = Lambda(AdaIN)([content_rep, sstd, smean])

decoded_image = decoder(full_rep)

#Loss things

s_loss1t = rep1(style_image)
c_loss1t = rep1(content_image)
loss1p = rep1(decoded_image)
s_loss1 = partial(style_loss, y_rep = s_loss1t, weight = 100)
c_loss1 = partial(content_loss, y_rep = c_loss1t)

s_loss2t = rep2(style_image)
c_loss2t = rep2(content_image)
loss2p = rep2(decoded_image)
s_loss2 = partial(style_loss, y_rep = s_loss2t, weight = 5)
c_loss2 = partial(content_loss, y_rep = c_loss2t)

s_loss3t = rep3(style_image)
c_loss3t = rep3(content_image)
loss3p = rep3(decoded_image)
s_loss3 = partial(style_loss, y_rep = s_loss3t, weight = 0.8)
c_loss3 = partial(content_loss, y_rep = c_loss3t)

s_loss4t = rep4(style_image)
c_loss4t = rep4(content_image)
loss4p = rep4(decoded_image)
s_loss4 = partial(style_loss, y_rep = s_loss4t, weight = 0.015)
c_loss4 = partial(content_loss, y_rep = c_loss4t)

#Model building

model = Model([content_image, style_image], [loss1p, loss2p, loss3p, loss4p, loss1p, loss2p, loss3p, loss4p])
eval_model = Model([content_image, style_image], decoded_image)

eval_model.summary()

encoder.trainable = False
for layer in encoder.layers:
    layer.trainable = False

rep1.trainable = False
for layer in rep1.layers:
    layer.trainable = False

rep2.trainable = False
for layer in rep2.layers:
    layer.trainable = False

rep3.trainable = False
for layer in rep3.layers:
    layer.trainable = False

rep4.trainable = False
for layer in rep4.layers:
    layer.trainable = False

#Compilation
model.compile(optimizer = Adam(lr = 0.0001),
                loss = [c_loss1, c_loss2, c_loss3, c_loss4, s_loss1, s_loss2, s_loss3, s_loss4],
                loss_weights = [1, 1, 1, 1, ss, ss, ss, ss])

#Datasets
coco = dataGenerator('coco', 256)
wikiart = dataGenerator('wikiart', 256)

#Evaluation function
def evaluate(num = 0):
    content_images = coco.get_batch(8)
    style_images = wikiart.get_batch(8)

    out_images = eval_model.predict([content_images, style_images], batch_size = 2)

    r = []
    for i in range(8):
        r.append(np.concatenate([content_images[i], style_images[i], out_images[i]], axis = 1))

    c = np.concatenate(r, axis = 0)

    x = Image.fromarray(np.uint8(c * 255))
    x.save("Results/e" + str(num) + ".png")

#Training loop
batch_size = 8
for iteration in range(100000):
    print(iteration, end = '\r')
    dummy = np.ones([batch_size, 1])
    content_images = coco.get_batch(batch_size)
    style_images = wikiart.get_batch(batch_size)
    loss = model.train_on_batch([content_images, style_images], [dummy] * 8)

    if iteration % 10 == 0:
        print(loss)
        evaluate(floor(iteration / 100))
        saveModel(decoder, "dec", floor(iteration / 1000))
        saveModel(sencoder, "sen", floor(iteration / 1000))
