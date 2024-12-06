import os 
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization, Lambda, Conv2D, LeakyReLU, ReLU, Dropout, Flatten, Dense, Activation, Reshape, Conv2DTranspose, Input, Concatenate, Embedding, multiply
import networkx as nx
from node2vec import Node2Vec
from tensorflow.keras.utils import plot_model
import datetime

def custom_activation(output):
    logexpsum = K.sum(K.exp(output), axis=-1, keepdims=True)
    result= logexpsum/ (logexpsum+ 1.0)
    return result

def define_CGAN_Discriminator_OneHot(n_classes, opt, in_shape=(1, 256, 1)):
    inp = Input(shape=in_shape)
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(n_classes, np.prod(in_shape))(label))
    label_embedding = Reshape(in_shape)(label_embedding)
    merged = Concatenate()([inp, label_embedding])
    fe = Conv2D(filters=64, kernel_size=(1, 1), strides=1)(merged)
    fe = LeakyReLU()(fe)
    fe = Dropout(0.3)(fe)
    fe = Conv2D(filters=64, kernel_size=(1, 2), strides=1)(fe)
    fe = LeakyReLU()(fe)
    fe = Dropout(0.3)(fe)
    fe = Conv2D(filters=64, kernel_size=(1, 3), strides=1)(fe)
    fe = LeakyReLU()(fe)
    fe = Dropout(0.3)(fe)
    fe = Conv2D(filters=64, kernel_size=(1, 4), strides=1)(fe)
    fe = LeakyReLU()(fe)
    fe = Dropout(0.3)(fe)
    fe = Conv2D(filters=64, kernel_size=(1, 5), strides=1)(fe)
    fe = LeakyReLU()(fe)
    fe = Dropout(0.3)(fe)

    fe = Flatten()(fe)
    fe = Dense(1)(fe)
    
    d_out_layer = Lambda(custom_activation)(fe)
    d_model = Model([inp, label], d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    plot_model(d_model, to_file='CGAN_D.png', show_shapes=True, show_layer_names=True)
    return d_model

def define_CGAN_Generator_OneHot(n_classes, latent_dim=124):
    in_lat = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(n_classes, latent_dim)(label))
    merged = Concatenate()([in_lat, label_embedding])

    gen = Dense(1 * 248 * 32)(merged)
    gen = ReLU()(gen)
    gen = Reshape((1, 248, 32))(gen)
    gen = Conv2DTranspose(filters = 32, kernel_size=(1,3), strides=1)(gen)
    gen = ReLU()(gen)
    gen = Conv2DTranspose(filters = 32, kernel_size=(1,3), strides=1)(gen)
    gen = ReLU()(gen)
    gen = Conv2DTranspose(filters = 32, kernel_size=(1,3), strides=1)(gen)
    gen = ReLU()(gen)
    out_layer = Conv2DTranspose(filters = 1, kernel_size= (1, 3), strides=1, activation='tanh')(gen)
    g_model = Model([in_lat, label], out_layer)  # 두 입력을 모두 포함
    plot_model(g_model, to_file='CGAN_G.png', show_shapes=True, show_layer_names=True)
    return g_model

def define_CGAN_OneHot(g_model, d_model, opt):
    d_model.trainable = False
    g_input = g_model.input
    g_output = g_model.output
    gan_output = d_model([g_output, g_input[1]])
    model = Model(g_input, gan_output)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model