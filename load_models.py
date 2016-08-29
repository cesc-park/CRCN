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


def create_crcn_original(is_entity=False):
    model = Sequential()
    model.add(BRNN(
        300, 300, return_sequences=True,
        activation='relu',init='he_normal',
        is_entity=is_entity))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Embedding(300, 512,init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Embedding(512, 4096,init='he_normal'))
    model.add(Dropout(0.7))
    return model

def create_crcn_bn(is_entity=False):
    model = Sequential()
    model.add(BRNN(
        300, 300, return_sequences=True,
        activation='relu',init='he_normal',
        is_entity=is_entity))
    model.add(BatchNormalization(300))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Embedding(300, 512,init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Embedding(512, 4096,init='he_normal'))
    model.add(Dropout(0.7))
    return model

def create_crcn_reg(is_entity=False):
    model = Sequential()
    model.add(BRNN(
        300, 300, return_sequences=True,
        activation='relu',init='he_normal',
        is_entity=is_entity, regularize=True))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Embedding(300, 512,init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Embedding(512, 4096,init='he_normal'))
    model.add(Dropout(0.7))
    return model

def create_crcn_dense(is_entity=False):
    model = Sequential()
    model.add(BRNN(
        300, 300, return_sequences=True,
        activation='relu',init='he_normal',
        is_entity=is_entity))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(CRCN_Dense(
        300, 512,init='he_normal',activation='relu', is_entity=is_entity))
    model.add(Dropout(0.5))
    model.add(CRCN_Dense(
        512, 4096,init='he_normal',activation='relu', is_entity=is_entity))
    model.add(Dropout(0.7))
    return model

def create_crcn_dense_reg(is_entity=False):
    model = Sequential()
    model.add(BRNN(
        300, 300, return_sequences=True,
        activation='relu',init='he_normal',
        is_entity=is_entity, regularize=True))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(CRCN_Dense(
        300, 512,init='he_normal',activation='relu',
        is_entity=is_entity, regularize=True))
    model.add(Dropout(0.5))
    model.add(CRCN_Dense(
        512, 4096,init='he_normal',activation='relu',
        is_entity=is_entity, regularize=True))
    model.add(Dropout(0.7))
    return model

def create_crcn_lstm(is_entity=False):
    model = Sequential()
    model.add(LSTM(
        300, 300, return_sequences=True,init='he_normal',is_entity=is_entity))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Embedding(300, 512,init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Embedding(512, 4096,init='he_normal'))
    model.add(Dropout(0.7))
    return model

def create_crcn_blstm(is_entity=False):
    model = Sequential()
    model.add(BLSTM(
        300, 300, return_sequences=True,init='he_normal',is_entity=is_entity))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Embedding(300, 512,init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Embedding(512, 4096,init='he_normal'))
    model.add(Dropout(0.7))
    return model

def create_crcn_blstm_reg(is_entity=False):
    model = Sequential()
    model.add(BLSTM(
        300, 300, return_sequences=True,init='he_normal',
        is_entity=is_entity, regularize=True))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Embedding(300, 512,init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Embedding(512, 4096,init='he_normal'))
    model.add(Dropout(0.7))
    return model

def create_crcn_blstm_dense(is_entity=False):
    model = Sequential()
    model.add(BLSTM(
        300, 300, return_sequences=True,init='he_normal',is_entity=is_entity))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(CRCN_Dense(
        300, 512,init='he_normal',activation='relu', is_entity=is_entity))
    model.add(Dropout(0.5))
    model.add(CRCN_Dense(
        512, 4096,init='he_normal',activation='relu', is_entity=is_entity))
    model.add(Dropout(0.7))
    return model

def create_crcn_blstm_dense_reg(is_entity=False):
    model = Sequential()
    model.add(BLSTM(
        300, 300, return_sequences=True,init='he_normal',
        is_entity=is_entity,regularize=True))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(CRCN_Dense(
        300, 512,init='he_normal',activation='relu',
        is_entity=is_entity,regularize=True))
    model.add(Dropout(0.5))
    model.add(CRCN_Dense(
        512, 4096,init='he_normal',activation='relu',
        is_entity=is_entity,regularize=True))
    model.add(Dropout(0.7))
    return model

def create_crcn_blstm_dense_bn(is_entity=False):
    model = Sequential()
    model.add(BLSTM(
        300, 300, return_sequences=True,init='he_normal', activation='tanh',
        inner_activation='hard_sigmoid', is_entity=is_entity))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(CRCN_Dense(
        300, 512,init='he_normal',activation='relu', is_entity=is_entity))
    model.add(Dropout(0.5))
    model.add(CRCN_Dense(
        512, 4096,init='he_normal',activation='relu', is_entity=is_entity))
    model.add(Dropout(0.7))
    return model

