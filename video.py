vidname = 'vid.mp4'
style = 's1.jpg'


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


resize_content = False
resize_style = True
n = 0

im2 = Image.open(style).convert('RGB')
if resize_style:
    im2 = im2.resize((256, 256), Image.BICUBIC)
im2 = np.array(im2).astype('float32') / 255.0


import cv2
vidcap = cv2.VideoCapture(vidname)
success,image = vidcap.read()
count = 0
size = (image.shape[1], image.shape[0])

images = []
style = []

while success:
    images.append(np.float32(image) / 255.0)
    style.append(im2)
    success,image = vidcap.read()
    print('Read a new frame: ', count, end = '\r')
    count += 1

print()

images = np.array(images)
style = np.array(style)

output = model.predict([images, style], batch_size = 16, verbose = 1)
output = np.uint8(output * 255)

out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 30, size)

for i in range(output.shape[0]):
    #print(output[i].shape)
    out.write(output[i])
    print('Write frame: ', i, end = '\r')

out.release()
