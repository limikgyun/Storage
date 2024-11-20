import time
import pickle
import numpy as np
import tensorflow as tf
import csv
import os

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from numpy.random import seed
from models import *
from models_cgan import *
from utils import *
from datetime import datetime
from matplotlib import pyplot as plt
from 전처리 import *

def fit_GAN(run, g_model, d_model, c_model, gan_model, n_samples, n_classes, X_sup, y_sup, dataset, n_epochs, n_batch, latent_dim = 100):
    tst_history = []
    X_tra, y_tra, X_tst, y_tst = dataset
    bat_per_epo = int(X_tra.shape[0] / n_batch)# calculate the number of batches per training epoch
    n_steps = bat_per_epo * n_epochs# calculate the number of training iterations
    half_batch = int(n_batch / 2)# calculate the size of half a batch of samples

    with open('/home/mnetlig/CGAN_Zero_Tensor/Storage/시각화/CSV_d_model_predictions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # write the header
        writer.writerow(['Step', 'Real Data Accuracy', 'Fake Data Accuracy'])

        # fit the model
        for i in range(n_steps):
            # update discriminator (c)
            [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], half_batch)
            c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
            # update discriminator (d)
            [X_real, _], y_real = generate_real_samples((X_tra, y_tra), half_batch)
            d_loss1, d_acc1 = d_model.train_on_batch(X_real, y_real)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss2, d_acc2 = d_model.train_on_batch(X_fake, y_fake)
            # update generator (g)
            X_gan, y_gan = generate_latent_points(latent_dim, n_batch), np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d/%d/%d, c[%.3f,%.0f], d[%.3f,%.0f,%.3f,%.0f], g[%.3f]' % (run+1, i+1, n_steps, c_loss, c_acc*100, d_loss1, d_acc1*100, d_loss2, d_acc2*100, g_loss))
            # write discriminator accuracies to CSV
            writer.writerow([i+1, d_acc1, d_acc2])
            # test after a epoch
            if (i+1) % (bat_per_epo * 1) == 0:
                _, _acc = c_model.evaluate(X_tst, y_tst, verbose=0)
                print('Test Accuracy: %.3f' % (_acc * 100))
                tst_history.append(_acc)

                # acc_log에 tst_acc 추가
                epoch = (i+1) // bat_per_epo
                # # 특정 레이블에 대해 생성된 데이터를 가시화
                # n_CSI = 20  # 20개의 CSI를 생성!
                # z_inputs = generate_latent_points(latent_dim, n_CSI)
                # generated_samples = g_model.predict(z_inputs)
                # generated_samples = generated_samples.reshape(-1, 256)
                # generated_samples = single_minmaxscale(generated_samples, scale_range=(0, 1))
                # plt.figure()
                # for x_g in generated_samples:
                #     plt.rcParams['pdf.fonttype'] = 42
                #     plt.rcParams['ps.fonttype'] = 42        
                #     plt.title('Generated Data (fake)', fontsize=20)
                #     plt.xlabel('CSI Index', fontsize=18)
                #     plt.ylabel('CSI Amplitude (Normalized)', fontsize=18)
                #     plt.axis([0, 256, 0, 1])
                #     plt.grid(True)
                #     plt.plot(x_g)
                # plt.savefig('/home/mnetlig/CGAN_Zero_Tensor/Storage/시각화/Base-GAN-에폭%d.png' % ((i+1)//bat_per_epo), dpi=400)
                # plt.close()
    return tst_history

def fit_CGAN(run, g_model, d_model, gan_model, n_samples, n_classes, X_sup, y_sup, dataset, n_epochs, n_batch, latent_dim=100):
    # 임의의 경로 설정
    start_time = time.time()  # 시작 시간 기록
    current_time = datetime.now().strftime('%y%m%d-%H%M')
    folder_path = '/home/mnetlig/CGAN_Zero_Tensor/Storage/시각화/{}'.format(current_time)
    file_name_csv = 'C간-log.csv'
    file_path_csv = os.path.join(folder_path, file_name_csv)
    file_name_img = 'C간-에폭{}-레이블{}.png'
    file_path_img = os.path.join(folder_path, file_name_img)
    os.makedirs(folder_path, exist_ok=True)
    
    tst_history = []
    X_tra, y_tra, X_tst, y_tst = dataset
    bat_per_epo = int(X_tra.shape[0] / n_batch)
    n_steps = bat_per_epo * n_epochs
    half_batch = int(n_batch / 2)

    with open(file_path_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Step', 'Real Accuracy', 'Fake Accuracy', 'D Loss Real', 'D Loss Fake', 'G Loss'])
        for i in range(n_steps):
            # update discriminator (d)
            [X_real, labels_real], y_real = generate_real_samples((X_tra, y_tra), half_batch)
            # print(X_real.shape)
            d_loss1 = d_model.train_on_batch([X_real, labels_real], y_real)
            real_acc = np.mean(d_model.predict([X_real, labels_real]))

            [X_fake, labels_fake], y_fake = c_generate_fake_samples(g_model, latent_dim, half_batch, n_classes)
            d_loss2 = d_model.train_on_batch([X_fake, labels_fake], y_fake)
            fake_acc = np.mean(d_model.predict([X_fake, labels_fake]))

            # update generator (g)
            z_input, labels_input = generate_latent_points(latent_dim, n_batch), np.random.randint(0, n_classes, n_batch)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            # summarize loss on this batch
            print('>%d/%d/%d, d[%.3f,%.3f], g[%.3f], Real:[%.3f], Fake:[%.3f]' % (run+1, i+1, n_steps, d_loss1[0], d_loss2[0], g_loss, real_acc, fake_acc))
            # write the accuracies to the CSV file
            writer.writerow([i+1, real_acc, fake_acc, d_loss1[0], d_loss2[0], g_loss])

            # test after a epoch
            if (i+1) % (bat_per_epo * 1) == 0:
                _, _acc = d_model.evaluate([X_tst, y_tst], np.ones((X_tst.shape[0], 1)), verbose=0)
                tst_history.append(_acc)
                epoch = (i+1) // bat_per_epo
                print('%d에폭 // %d/%d 스텝 // d의 Test Accuracy: %.3f퍼센트' % (epoch, i+1,n_steps,_acc * 100))
                visualize_generated_data(g_model, latent_dim, 10, [0, 1, 2], epoch, file_path_img)
    end_time = time.time()  # 종료 시간 기록
    elapsed_time = end_time - start_time  # 소요된 시간 계산
    print('총 소요 시간 : ', int(elapsed_time),'초', '\n에포크별 소요 시간 : ', int(elapsed_time / n_epochs),'초')  # 소요된 시간 출력
    return tst_history

def run_base(pickle_file_path):  # pickle_file을 매개변수로 받도록 수정
    data_file_name_with_ext = os.path.basename(pickle_file_path)
    data_name, _ = os.path.splitext(data_file_name_with_ext)
    # experiment setup
    n_classes = 3
    n_samples = [3000]
    run_times = 1
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    n_epochs = 100
    n_batch = 128

    # load dataset
    dataset = data_preproc(np.asarray(pickle.load(open(pickle_file_path, 'rb'))))
    X_tra, y_tra, X_tst, y_tst = dataset
    for j in range(len(n_samples)):
        history = []
        # select supervised dataset
        X_sup, y_sup = select_supervised_samples(X_tra, y_tra, n_samples[j], n_classes)        
        for i in range(run_times):
            print('{}회차 실시 중/총 {}회 중'.format(i+1, run_times))
            # change seed for each run
            seed(run_times)
            # define a semi-GAN model
            d_model, c_model = define_discriminator(n_classes, optimizer)
            g_model = define_generator()
            gan_model = define_GAN(g_model, d_model, optimizer)

            # train the semi-GAN model
            tst_acc = fit_GAN(i, g_model, d_model, c_model, gan_model, n_samples[j], n_classes, X_sup, y_sup, dataset, n_epochs, n_batch)
            history.append(max(tst_acc))

        # save models:
        g_model.save('/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-{}-g-{}샘플-{}에폭.h5'.format(data_name, n_samples[j], n_epochs))
        d_model.save('/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-{}-d-{}샘플-{}에폭.h5'.format(data_name, n_samples[j], n_epochs))
        c_model.save('/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-{}-c-{}샘플-{}에폭.h5'.format(data_name, n_samples[j], n_epochs))

def run_cgan(pickle_file_path):# 전이학습의 base가 될 cGAN 모델 학습
    data_file_name_with_ext = os.path.basename(pickle_file_path)
    data_name, _ = os.path.splitext(data_file_name_with_ext)
    # experiment setup
    n_classes = 9
    n_samples = [3000]
    run_times = 1
    # optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    optimizer_gan = Adam(learning_rate=0.0002, beta_1=0.5)
    n_epochs = 100
    n_batch = 128

    # load dataset
    dataset = data_preproc(np.asarray(pickle.load(open(pickle_file_path,'rb'))))
    X_tra, y_tra, X_tst, y_tst = dataset
    for j in range(len(n_samples)):
        history = []
        # select supervised dataset
        X_sup, y_sup = select_supervised_samples(X_tra, y_tra, n_samples[j], n_classes)        
        for i in range(run_times):
            print('{}회차 실시 중/총 {}회 중'.format(i+1, run_times))
            # change seed for each run
            seed(run_times)
            # define a cGAN model
            d_model = c_define_discriminator(n_classes, optimizer)
            g_model = c_define_generator(n_classes)
            gan_model = c_define_GAN(g_model, d_model, optimizer_gan)

            # train the cGAN model
            tst_acc = fit_CGAN(i, g_model, d_model, gan_model, n_samples[j], n_classes, X_sup, y_sup, dataset, n_epochs, n_batch)

            history.append(max(tst_acc))
        
        # best = max(history)
        if len(history) >= 1:
            best = sum(history[-1:]) / 1
        else:
            best = sum(history) / len(history)  # history의 길이가 2보다 작을 경우 전체 평균
        # d_model.summary()
        # g_model.summary()
    g_model.save('/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base-C간-{}-g-{}샘플-{}에폭.h5'.format(data_name, n_samples[j], n_epochs))
    d_model.save('/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base-C간-{}-d-{}샘플-{}에폭.h5'.format(data_name, n_samples[j], n_epochs))

def run_trans_cgan(pickle_file_path, d_path, g_path):# 전이학습의 base가 될 cGAN 모델 학습
    data_file_name_with_ext = os.path.basename(pickle_file_path)
    data_name, _ = os.path.splitext(data_file_name_with_ext)
    # experiment setup
    n_classes = 9
    n_samples = [3000]
    run_times = 1
    # optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    optimizer_gan = Adam(learning_rate=0.0002, beta_1=0.5)
    n_epochs = 100
    n_batch = 128

    # load dataset
    dataset = data_preproc(np.asarray(pickle.load(open(pickle_file_path,'rb'))))
    dataset = data_preproc(np.asarray(pickle.load(open(pickle_file_path,'rb'))))
    X_tra, y_tra, X_tst, y_tst = dataset
    for j in range(len(n_samples)):
        history = []
        # select supervised dataset
        X_sup, y_sup = removed_select_supervised_samples(X_tra, y_tra, n_samples[j], n_classes, [0,2]) # 레이블 0,2에 대해 전이학습
        for i in range(run_times):
            print('{}회차 실시 중 / 총 {}회 중'.format(i+1, run_times))
            # change seed for each run
            seed(run_times)
            
            # Load pre-trained cGAN models
            to_d_model = load_model(d_path)
            to_g_model = load_model(g_path)

            # Define new models for transfer learning
            d_model = c_define_trans_discriminator(n_classes, optimizer)
            g_model = c_define_trans_generator(n_classes)

            # Set weights from pre-trained models
            d_model.set_weights(to_d_model.get_weights())
            g_model.set_weights(to_g_model.get_weights())
            gan_model = c_define_GAN(g_model, d_model, optimizer_gan)

            # Train the cGAN model
            tst_acc = fit_CGAN(i, g_model, d_model, gan_model, n_samples[j], n_classes, X_sup, y_sup, dataset, n_epochs, n_batch)

        #     history.append(max(tst_acc))
        
        # # Calculate the best accuracy
        # if len(history) >= 1:
        #     best = sum(history[-1:]) / 1
        # else:
        #     best = sum(history) / len(history)

        # Save models
        g_model.save('/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Trans-C간-{}-g-{}샘플-{}에폭.h5'.format(data_name, n_samples[j], n_epochs))
        d_model.save('/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Trans-C간-{}-d-{}샘플-{}에폭.h5'.format(data_name, n_samples[j], n_epochs))

def run_trans_base(n_epochs, pickle_file_path, c_path, d_path, g_path):# base 모델(G/C/D GAN)을 불러와서 Transfer Learning / 데이터에 몇몇 레이블을 빼고, removed_select_supervised_samples 함수를 사용 해서 전이학습 실시
    data_file_name_with_ext = os.path.basename(pickle_file_path)
    data_name, _ = os.path.splitext(data_file_name_with_ext)
    #experiment setup
    n_classes = 3
    # n_samples = [16] # init value / define the number of labeled samples here
    n_samples = [1000]
    run_times = 1 # define the number of runs to traing under this setting
    # optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    # n_epochs = 80
    # n_batch = 128   # init value
    n_batch = 128

    #load dataset
    dataset = data_preproc(np.asarray(pickle.load(open(pickle_file_path,'rb'))))
    # dataset = data_preproc(np.asarray(pickle.load(open('/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Merged b (Real[0,2]+Fake[1])_csi.pickle','rb'))))
    # dataset = data_preproc(np.asarray(pickle.load(open('/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Generated_b.pickle','rb'))))
    
    # # 특정 레이블로만 학습하는 데이터셋
    # selected_label = 1 # 0부터 5까지의 레이블 중 선택해서 removed_select_supervised_samples 기능에서 사용하게됨
    # dataset = data_preproc(np.asarray(pickle.load(open(f'/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Only_label_{selected_label}.pickle','rb'))))
    
    X_tra, y_tra, X_tst, y_tst = dataset
    for j in range(len(n_samples)):
        history = []
        # select supervised dataset
        
        # # 특정 레이블로 필터된 샘플을 선택
        # X_sup, y_sup = removed_select_supervised_samples(X_tra, y_tra, n_samples[j], n_classes, selected_label)
        
        # cGAN으로 생성된 데이터를 추가한 버전
        X_sup, y_sup = select_supervised_samples(X_tra, y_tra, n_samples[j], n_classes)

        # # cGAN으로 생성한 데이터만 쓴 버전
        # X_sup, y_sup = removed_select_supervised_samples(X_tra, y_tra, n_samples[j], n_classes)
        
        for i in range(run_times):
            print('{}회차 실시 중 / 총 {}회 중'.format(i+1, run_times))
            # change seed for each run
            seed(run_times)
            
            to_c_model = load_model(c_path)
            to_d_model = load_model(d_path)
            to_g_model = load_model(g_path)

            d_model, c_model = define_trans_discriminator(n_classes, optimizer)
            g_model = define_trans_generator()
            # d_model, c_model = define_trans_discriminator_test(n_classes, optimizer)
            # g_model = define_trans_generator_test()

            # 가중치 불러오기
            c_model.set_weights(to_c_model.get_weights())
            d_model.set_weights(to_d_model.get_weights())
            g_model.set_weights(to_g_model.get_weights())

            gan_model = define_GAN(g_model, d_model, optimizer)

            # train the semi-GAN model
            tst_acc = fit_GAN(i ,g_model, d_model, c_model, gan_model, n_samples[j], n_classes, X_sup, y_sup, dataset, n_epochs, n_batch)

            history.append(max(tst_acc))
        best = max(history)

        # save models:
        # g_model.save('/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Trans-Base간-{}-g-{}샘플-{}에폭-{}.h5'.format(data_name, n_samples[j], n_epochs, int(best*100)))
        # d_model.save('/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Trans-Base간-{}-d-{}샘플-{}에폭-{}.h5'.format(data_name, n_samples[j], n_epochs, int(best*100)))
        c_model.save('/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Trans-Base간-{}-c-{}샘플-{}에폭.h5'.format(data_name, n_samples[j], n_epochs))

if __name__ == '__main__':    
    # 241004 새 실험을 위한 절차
    # # for x
    pickle_files = [
    '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Processed_aa_000.pickle',
    # '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Processed_x++_00.pickle'
    ]
    for pickle_file in pickle_files:
    #     run_base(pickle_file)
        run_cgan(pickle_file)

    # run_trans_cgan(9, '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Removed_[1]_x++_00.pickle',
    #                '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/C간-Processed_x_00-d-3000샘플-100에폭.h5'
    #                '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/C간-Processed_x_00-g-3000샘플-100에폭.h5')        
    # run_trans_base(90, '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Generated_[1]_x++_00.pickle',
    #                     '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-Processed_x_00-c-3000샘플-100에폭.h5',
    #                     '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-Processed_x_00-d-3000샘플-100에폭.h5',
    #                     '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-Processed_x_00-g-3000샘플-100에폭.h5')
    # run_trans_cgan(9, '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Removed_[1]_x_00.pickle',
    #                '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/C간-Processed_x++_00-d-3000샘플-100에폭.h5',
    #                '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/C간-Processed_x++_00-g-3000샘플-100에폭.h5')        
    # run_trans_base(90, '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Generated_[1]_x_00.pickle',
    #                     '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-Processed_x++_00-c-3000샘플-100에폭.h5',
    #                     '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-Processed_x++_00-d-3000샘플-100에폭.h5',
    #                     '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-Processed_x++_00-g-3000샘플-100에폭.h5')
    # for z
    # pickle_files = [
    # '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Processed_z_00.pickle',
    # '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Processed_z++_00.pickle'
    # ]
    # for pickle_file in pickle_files:
    #     run_base(pickle_file)
    # for pickle_file in pickle_files:
    #     run_cgan(pickle_file)
    # run_trans_cgan(9,
    #                '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Removed_[1]_z++_00.pickle',
    #                '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/C간-Processed_z_00-d-3000샘플-100에폭.h5'
    #                '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/C간-Processed_z_00-g-3000샘플-100에폭.h5')        
    # generate_csi_by_CGAN('/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Removed_[1]_z++_00.pickle',
    #                      '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Trans-C간-Removed_[1]_z++_00-g-1000샘플-9에폭.h5')
    # run_trans_base(90, '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Generated_[1]_z++_00.pickle',
    #                     '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-Processed_z_00-c-3000샘플-100에폭.h5',
    #                     '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-Processed_z_00-d-3000샘플-100에폭.h5',
    #                     '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-Processed_z_00-g-3000샘플-100에폭.h5')
    # run_trans_cgan(5, 
    #                '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Removed_[1]_z_00.pickle',
    #                '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/C간-Processed_z++_00-d-3000샘플-100에폭.h5',
    #                '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/C간-Processed_z++_00-g-3000샘플-100에폭.h5')
    # generate_csi_by_CGAN('/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Removed_[1]_z_00.pickle',
    #                      '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Trans-C간-Removed_[1]_z_00-g-1000샘플-5에폭.h5')      
    # generate_csi_by_CGAN('/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Removed_[1]_z_00.pickle',
    #                      '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Trans-C간-Removed_[1]_z_00-g-1000샘플-9에폭.h5')        
    # run_trans_base(7, '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Generated_[1]_z_00.pickle',
    #                     '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-Processed_z++_00-c-3000샘플-100에폭.h5',
    #                     '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-Processed_z++_00-d-3000샘플-100에폭.h5',
    #                     '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-Processed_z++_00-g-3000샘플-100에폭.h5')













    # #1 제안사항 학습 순서
    # pickle_files = [
    # '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Processed_x_00.pickle',
    # '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Processed_x+_00.pickle',
    # '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Processed_x++_00.pickle'
    # ]
    # for pickle_file in pickle_files:
    #     run_base(pickle_file)
    #     run_cgan(pickle_file)
    
    ## run_trans_cgan(9)은  # 7~9 에포크가 적당 (분류정확도가 피크를 찍기 직전)
    
    # for n_epoch in [1,3]:
    #     run_trans_cgan(n_epoch,
    #     '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Removed_c_0[0,2].pickle',
    #     '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/C간-b-240923-2228-d-3000샘플-200에폭-100.h5',
    #     '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/C간-b-240923-2228-g-3000샘플-200에폭-100.h5')

#     run_trans_base(90,
# '/home/mnetlig/Desktop/CSI-SemiGAN-master//Merged b2c 7 (Real[0,2]+Fake[1])_csi.pickle',
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-b-240924-0021-c-3000샘플-200에폭-100.h5',
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-b-240924-0021-d-3000샘플-200에폭-100.h5',
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-b-240924-0021-g-3000샘플-200에폭-100.h5')
    
#     run_trans_base(90,
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Merged b2c 9 (Real[0,2]+Fake[1])_csi.pickle',
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-b-240924-0021-c-3000샘플-200에폭-100.h5',
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-b-240924-0021-d-3000샘플-200에폭-100.h5',
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-b-240924-0021-g-3000샘플-200에폭-100.h5')

#     run_trans_base(53,#83,0.9279999732971191 / #65 / #81 /#80
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Merged a2c 8 (Real[0,2]+Fake[1])_csi.pickle',
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-a-240924-0008-c-3000샘플-200에폭-100.h5',
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-a-240924-0008-d-3000샘플-200에폭-100.h5',
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-a-240924-0008-g-3000샘플-200에폭-100.h5')


    # run_trans_base(80)    # 80 에포크가 적당 (분류정확도가 피크를 찍은 직후)
#     run_trans_base(80,
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Merged b2a (Real[0,2]+Fake[1])_csi.pickle',
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-b-240924-0021-c-3000샘플-200에폭-100.h5',
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-b-240924-0021-d-3000샘플-200에폭-100.h5',
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-b-240924-0021-g-3000샘플-200에폭-100.h5')

#     run_trans_base(80,
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Merged a2c (Real[0,2]+Fake[1])_csi.pickle',
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-a-240924-0008-c-3000샘플-200에폭-100.h5',
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-a-240924-0008-d-3000샘플-200에폭-100.h5',
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-a-240924-0008-g-3000샘플-200에폭-100.h5')
    
#     run_trans_base(80,
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Merged b2c (Real[0,2]+Fake[1])_csi.pickle',
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-b-240924-0021-c-3000샘플-200에폭-100.h5',
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-b-240924-0021-d-3000샘플-200에폭-100.h5',
# '/home/mnetlig/CGAN_Zero_Tensor/Storage/모델/Base간-b-240924-0021-g-3000샘플-200에폭-100.h5')
    






    # #2 제안사항 학습 순서
    # run_base('/home/mnetlig/CGAN_Zero_Tensor/Storage/데이터/Processed_b_0.pickle')
    # for i in range(1):
    #     # for label in [1, 3, 5]:
    #     #     run_trans_base(label)
    #     run_w_transfer_case2()
    #     print(f"=====================================================\n{i+1}회 완료\n=====================================================")