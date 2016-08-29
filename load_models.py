import sys
sys.path.append("./keras")
from keras.models import Sequential
#from keras.models import Graph
from keras.layers.embeddings import Embedding
from keras.layers.core import *
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.layers.recurrent import *


def create_crcn():
    model = Sequential()
    model.add(BLSTM(
        300, 300, return_sequences=True,init='he_normal',
        is_entity=True, regularize=True))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Embedding(300, 512,init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Embedding(512, 4096,init='he_normal'))
    model.add(Dropout(0.7))
    return model

def create_rcn():
    model = Sequential()
    model.add(BLSTM(
        300, 300, return_sequences=True,init='he_normal',
        is_entity=False, regularize=True))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Embedding(300, 512,init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Embedding(512, 4096,init='he_normal'))
    model.add(Dropout(0.7))
    return model

