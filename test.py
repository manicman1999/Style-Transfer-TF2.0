import numpy as np
from PIL import Image
import random

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg19 import VGG19

model_num = 90

vgg = VGG19(include_top=False, weights='imagenet')

from datagen import dataGenerator, printProgressBar

for i in range(len(vgg.layers)):
    if vgg.layers[i].name == 'block4_conv1':
        encoder = Model(vgg.input, vgg.layers[i].output)

def loadModel(name, num):

    file = open("Models/"+name+".json", 'r')
    json = file.read()
    file.close()

    mod = model_from_json(json)
    mod.load_weights("Models/"+name+"_"+str(num)+".h5")

    return mod

decoder = loadModel('dec', model_num)
sencoder = loadModel('sen', model_num)

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

c_image = Input([None, None, 3])
s_image = Input([None, None, 3])

c_encoding = encoder(c_image)
[smean, sstd] = sencoder(s_image)

full_representation = AdaIN([c_encoding, sstd, smean])

decoded_image = decoder(full_representation)

model = Model([c_image, s_image], decoded_image)


resize_content = True
resize_style = True
n = 0

while True:

    print("Options (or press enter to continue):")
    opt = input()

    if opt.split(' ')[0] == "rc":
        resize_content = not resize_content
    elif opt.split(' ')[0] == "rs":
        resize_style = not resize_style

    print("Content image location: ", end = '')
    content = input()

    try:
        im1 = Image.open(content).convert('RGB')
        if resize_content:
            im1 = im1.resize((256, 256), Image.BICUBIC)
        im1 = np.array(im1).astype('float32') / 255.0
        im1 = np.array([im1])
    except:
        print("Not found!")
        continue

    print("Style image location: ", end = '')
    style = input()

    try:
        im2 = Image.open(style).convert('RGB')
        if resize_style:
            im2 = im2.resize((256, 256), Image.BICUBIC)
        im2 = np.array(im2).astype('float32') / 255.0
        im2 = np.array([im2])
    except:
        print("Not found!")
        continue


    output = model.predict([im1, im2])

    output = Image.fromarray(np.uint8(output[0] * 255))
    output.save("Results/output"+str(n)+".png")

    print("Saved as output"+str(n)+".png in /Results/")
    n = n + 1
