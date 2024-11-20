import os 
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization, Lambda, Conv1D, Conv2D, LeakyReLU, ReLU, Dropout, Flatten, Dense, Activation, Reshape, Conv2DTranspose, Input, Concatenate, Embedding, multiply
from models import *

def c_define_discriminator(n_classes, opt, in_shape=(1, 256, 1)):
    inp = Input(shape=in_shape)
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(n_classes, np.prod(in_shape))(label))
    label_embedding = Reshape(in_shape)(label_embedding)
    merged = Concatenate()([inp, label_embedding])
    fe = Conv2D(filters=32, kernel_size=(1, 9), strides=2)(merged)
    fe = LeakyReLU()(fe)
    fe = Conv2D(filters=32, kernel_size=(1, 8))(fe)
    fe = LeakyReLU()(fe)
    fe = Conv2D(filters=32, kernel_size=(1, 8))(fe)
    fe = LeakyReLU()(fe)
    fe = Flatten()(fe)
    fe = Dropout(0.4)(fe)
    fe = Dense(1)(fe)
    
    d_out_layer = Lambda(custom_activation)(fe)
    d_model = Model([inp, label], d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return d_model

def c_define_generator(n_classes, latent_dim=100):
    # 잠재 공간 벡터 입력
    in_lat = Input(shape=(latent_dim,))
    # 레이블 입력
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(n_classes, latent_dim)(label))
    merged = Concatenate()([in_lat, label_embedding])

    n_nodes = 32 * 1 * 244
    gen = Dense(n_nodes)(merged)
    gen = ReLU()(gen)
    gen = Reshape((1, 244, 32))(gen)

    gen = Conv2DTranspose(filters = 32, kernel_size=(1,5), strides=1)(gen)
    gen = ReLU()(gen)   
    gen = Conv2DTranspose(filters = 32, kernel_size=(1,5), strides=1)(gen)
    gen = ReLU()(gen)
    gen = Conv2DTranspose(filters = 32, kernel_size=(1,5), strides=1)(gen)
    gen = ReLU()(gen)
    out_layer = Conv2DTranspose(filters = 1, kernel_size= (1, 5), strides=1, activation='tanh', padding='same')(gen)
    g_model = Model([in_lat, label], out_layer)  # 두 입력을 모두 포함
    return g_model

# def c_define_generator(n_classes, latent_dim=256):
#     # 잠재 공간 벡터 입력
#     in_lat = Input(shape=(latent_dim,))
#     # 레이블 입력
#     label = Input(shape=(1,), dtype='int32')
#     # 레이블 임베딩
#     label_embedding = Flatten()(Embedding(n_classes, latent_dim)(label))
#     # 잠재 공간 벡터와 레이블 임베딩 병합
#     merged = Concatenate()([in_lat, label_embedding])

#     n_nodes = 32 * 1 * 256
#     gen = Dense(n_nodes)(merged)
#     # gen = BatchNormalization()(gen)
#     gen = ReLU()(gen)
#     gen = Reshape((1, 256, 32))(gen)

#     gen = Conv2D(filters = 32, kernel_size=(1,20), strides=1)(gen)
#     gen = ReLU()(gen)   
#     gen = Conv2D(filters = 32, kernel_size=(1,40), strides=1)(gen)
#     gen = ReLU()(gen)   
#     gen = Conv2DTranspose(filters = 32, kernel_size=(1,40), strides=1)(gen)
#     gen = ReLU()(gen)
#     gen = Conv2DTranspose(filters = 32, kernel_size=(1,20), strides=1)(gen)
#     gen = ReLU()(gen)
#     out_layer = Conv2DTranspose(filters = 1, kernel_size= (1, 5), strides=1, activation='tanh', padding='same')(gen)
#     g_model = Model([in_lat, label], out_layer)  # 두 입력을 모두 포함
#     return g_model

def c_define_GAN(g_model, d_model, opt):
    d_model.trainable = False
    g_input = g_model.input
    g_output = g_model.output
    # 생성기의 출력과 레이블을 판별기의 입력으로 사용
    gan_output = d_model([g_output, g_input[1]])
    model = Model(g_input, gan_output)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def c_define_trans_discriminator(n_classes, opt, in_shape=(1, 256, 1)):
    inp = Input(shape=in_shape)
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(n_classes, np.prod(in_shape))(label))
    label_embedding = Reshape(in_shape)(label_embedding)
    merged = Concatenate()([inp, label_embedding])
    
    fe = Conv2D(filters=32, kernel_size=(1, 9), strides=2)(merged)
    # fe = BatchNormalization()(fe)
    fe = LeakyReLU()(fe)
    fe = Conv2D(filters=32, kernel_size=(1, 8))(fe)
    # fe = BatchNormalization()(fe)
    fe = LeakyReLU()(fe)
    fe = Conv2D(filters=32, kernel_size=(1, 8))(fe)
    # fe = BatchNormalization()(fe)
    fe = LeakyReLU()(fe)
    fe = Flatten()(fe)
    fe = Dropout(0.4)(fe)
    fe = Dense(1)(fe)
    
    # Discriminator D
    d_out_layer = Lambda(custom_activation)(fe)
    d_model = Model([inp, label], d_out_layer)
    
    # Freeze all layers except the last two
    for layer in d_model.layers[:-2]:
        layer.trainable = False
        # layer.trainable = True
    for layer in d_model.layers[-2:]:
        layer.trainable = True
    
    d_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return d_model

def c_define_trans_generator(n_classes, latent_dim=100):
    # 잠재 공간 벡터 입력
    in_lat = Input(shape=(latent_dim,))
    # 레이블 입력
    label = Input(shape=(1,), dtype='int32')
    # 레이블 임베딩
    label_embedding = Flatten()(Embedding(n_classes, latent_dim)(label))
    # 잠재 공간 벡터와 레이블 임베딩 병합
    merged = Concatenate()([in_lat, label_embedding])

    n_nodes = 32 * 1 * 244
    gen = Dense(n_nodes)(merged)
    # gen = BatchNormalization()(gen)
    gen = ReLU()(gen)
    gen = Reshape((1, 244, 32))(gen)

    # # 추가 레이어
    # gen = Conv2DTranspose(filters = 32, kernel_size=(1,5), strides=1)(gen)
    # gen = BatchNormalization()(gen)
    # gen = ReLU()(gen)
    # # 추가 레이어

    gen = Conv2DTranspose(filters = 32, kernel_size=(1,5), strides=1)(gen)
    # gen = BatchNormalization()(gen)
    gen = ReLU()(gen)   
    gen = Conv2DTranspose(filters = 32, kernel_size=(1,5), strides=1)(gen)
    # gen = BatchNormalization()(gen)
    gen = ReLU()(gen)
    gen = Conv2DTranspose(filters = 32, kernel_size=(1,5), strides=1)(gen)
    # gen = BatchNormalization()(gen)
    gen = ReLU()(gen)
    out_layer = Conv2DTranspose(filters = 1, kernel_size= (1, 5), strides=1, activation='tanh', padding='same')(gen)
    g_model = Model([in_lat, label], out_layer)  # 두 입력을 모두 포함
    
    # Freeze all layers except the last two
    for layer in g_model.layers[:-2]:
        layer.trainable = False
        # layer.trainable = True
    for layer in g_model.layers[-2:]:
        layer.trainable = True
    
    return g_model