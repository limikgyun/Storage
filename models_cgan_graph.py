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

def generate_labels(n):
    return list(range(n))

def labels_to_graph_embedding(labels, coordinate_X, coordinate_Y, dimensions=124, walk_length=30, num_walks=200, workers=10):
    # Create a 3x3 grid graph
    G = nx.grid_2d_graph(coordinate_X, coordinate_Y)
    mapping = {node: label for node, label in zip(G.nodes(), labels)}
    G = nx.relabel_nodes(G, mapping)
    # Generate walks and train node2vec model
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    # Create embeddings for each label
    embeddings = {label: model.wv[str(label)] for label in G.nodes()}
    embeddings_array = np.array([embeddings[label] for label in labels])
    return embeddings_array

def custom_activation(output):
    logexpsum = K.sum(K.exp(output), axis=-1, keepdims=True)
    result= logexpsum/ (logexpsum+ 1.0)
    return result

def define_CGAN_Discriminator_Graph(n_classes, optimizer, graph_embedding, in_shape=(1, 256, 1)):
    inp = Input(shape=in_shape)
    label = Input(shape=(1,), dtype='int32')

    # 그래프 임베딩을 Keras 레이어로 변환
    embedding_layer = Embedding(input_dim=n_classes, output_dim=graph_embedding.shape[1], weights=[graph_embedding], trainable=True)
    

    fe = Conv2DTranspose(filters=1, kernel_size=(1, 1), strides=1)(noisy_label_embedding)
    fe = LeakyReLU()(fe)
    fe = Dropout(0.3)(fe)
    
    label_embedding = Flatten()(embedding_layer(label))
    label_embedding = Reshape((1, 1, graph_embedding.shape[1]))(label_embedding)  # Reshape to match Conv2D input shape
    label_embedding = Conv2D(filters=1, kernel_size=(1, 1), strides=1)(label_embedding)  # Conv2D 레이어를 사용하여 차원 맞추기
    label_embedding = Reshape(in_shape)(label_embedding)  # Reshape to match inp shape
    noisy_label_embedding = Lambda(lambda x: x[0] + x[1])([label_embedding, inp])
    
    
    label_embedding = Flatten()(embedding_layer(label))
    label_embedding = Dense(np.prod(in_shape))(label_embedding)  # 차원을 맞추기 위해 Dense 레이어 추가
    label_embedding = Reshape(in_shape)(label_embedding)  # Reshape to match inp shape
    noisy_label_embedding = Lambda(lambda x: x[0] + x[1])([label_embedding, inp])
    # label_embedding = Flatten()(Embedding(n_classes, np.prod(in_shape))(label))
    # label_embedding = Reshape(in_shape)(label_embedding)
    # merged = Concatenate()([inp, label_embedding])
    fe = Conv2D(filters=64, kernel_size=(1, 1), strides=1)(noisy_label_embedding)
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
    d_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    plot_model(d_model, to_file='CGAN_D.png', show_shapes=True, show_layer_names=True)
    return d_model

def define_CGAN_Generator_Graph(n_classes, graph_embedding, latent_dim=124):
     # 잠재 공간 벡터 입력
    in_lat = Input(shape=(latent_dim,))
    # 레이블 입력
    label = Input(shape=(1,), dtype='int32')
    
    # 그래프 임베딩을 Keras 레이어로 변환
    embedding_layer = Embedding(input_dim=n_classes, output_dim=graph_embedding.shape[1], weights=[graph_embedding], trainable=True)
    label_embedding = Flatten()(embedding_layer(label))
    # 특정 값에 노이즈를 더한 입력값 생성
    noisy_label_embedding = Lambda(lambda x: x[0] + x[1])([label_embedding, in_lat])
    
    gen = Dense(1 * 248 * 32)(noisy_label_embedding)
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

def define_CGAN_Graph(g_model, d_model, opt):
    d_model.trainable = False
    g_input = g_model.input
    g_output = g_model.output
    gan_output = d_model([g_output, g_input[1]])
    model = Model(g_input, gan_output)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model