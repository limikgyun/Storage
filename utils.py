import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler 
from matplotlib import pyplot as plt

def single_minmaxscale(data, scale_range):
    def minmaxscale(data, scale_range):
        scaler = MinMaxScaler(scale_range)
        scaler.fit(data)
        normalized = scaler.transform(data)
        return normalized
    X = []
    for i in data:
        X.append(minmaxscale(i.reshape(-1,1), scale_range))
    return np.asarray(X)     

def data_preproc(dataset, scale_range = (-1, 1)):
    X_tra, y_tra, X_tst, y_tst = dataset
    X_tra = single_minmaxscale(X_tra, scale_range)
    X_tst = single_minmaxscale(X_tst, scale_range)

    X_tra = X_tra.astype('float32')
    X_tra = X_tra.reshape(-1,1,256,1)
    X_tst = X_tst.astype('float32')
    X_tst = X_tst.reshape(-1,1,256,1)
    print('Finished preprocessing.')
    return (X_tra, y_tra, X_tst, y_tst)

def generate_data_by_label(generator, latent_dim, n_samples, label_value):
    latent_vectors = np.random.randn(n_samples, latent_dim)
    labels = np.full((n_samples, 1), label_value)
    generated_data = generator.predict([latent_vectors, labels])
    return generated_data

def visualize_generated_data(g_model, latent_dim, n_samples, labels, epoch, save_path):
    for label in labels:
        generated_samples = generate_data_by_label(g_model, latent_dim, n_samples, label)
        generated_samples = generated_samples.reshape(-1, 256)
        generated_samples = single_minmaxscale(generated_samples, scale_range=(0, 1))
        plt.figure()
        for x_g in generated_samples:
            plt.rcParams['pdf.fonttype'] = 42
            plt.rcParams['ps.fonttype'] = 42        
            plt.title('Location $p_{%d}$ (fake)' % label, fontsize=20)
            plt.xlabel('CSI Index', fontsize=18)
            plt.ylabel('CSI Amplitude (Normalized)', fontsize=18)
            plt.axis([0, 256, 0, 1])
            plt.grid(True)
            plt.plot(x_g)
        plt.savefig(save_path.format(epoch, label), dpi=400)
        plt.close()

# def log_accuracy_to_csv(epoch, tst_acc):
#     with open(f'/home/mnetlig/CGAN_Zero_Tensor/Storage/시각화/CSV_acc_log.csv', mode='a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([epoch, tst_acc])

def select_supervised_samples(X, Y, n_samples, n_classes):
    X_list, Y_list = list(), list()
    n_per_class = int(n_samples / n_classes)
    for i in range(n_classes):
        X_with_class = X[Y == i]
        if len(X_with_class) == 0:
            raise ValueError(f"No samples found for class {i}")
        if n_per_class <= 0:
            raise ValueError("Number of samples per class must be greater than 0")
        ix = np.random.randint(0, len(X_with_class), n_per_class)
        [X_list.append(X_with_class[j]) for j in ix]
        [Y_list.append(i) for j in ix]
    
    return np.asarray(X_list), np.asarray(Y_list)

def removed_select_supervised_samples(X, Y, n_samples, n_classes, selected_label):
    X_list, Y_list = list(), list()
    n_per_class = int(n_samples / n_classes)

    for i in range(n_classes):
        # if i % 2 == 0:
        #     continue  # 레이블이 0 혹은 짝수인 경우 건너뜀
        if i not in selected_label:
            continue  # 레이블이 selected_label이 아닌 경우 건너뜀

        X_with_class = X[Y == i]
        
        if len(X_with_class) == 0:
            raise ValueError(f"No samples found for class {i}")
        
        if n_per_class <= 0:
            raise ValueError("Number of samples per class must be greater than 0")
        
        ix = np.random.randint(0, len(X_with_class), n_per_class)
        [X_list.append(X_with_class[j]) for j in ix]
        [Y_list.append(i) for j in ix]
    
    return np.asarray(X_list), np.asarray(Y_list)

def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    ix = np.random.randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    Y=np.ones((n_samples, 1))
    return [X, labels], Y 

def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input   # [16,100]

def generate_fake_samples(generator, latent_dim, n_samples):
    z_input = generate_latent_points(latent_dim, n_samples)
    images = generator.predict(z_input)
    y = np.zeros((n_samples, 1))
    return images, y

def c_generate_fake_samples(generator, latent_dim, n_samples, n_classes):
    z_input = generate_latent_points(latent_dim, n_samples)
    labels = np.random.randint(0, n_classes, n_samples)
    images = generator.predict([z_input, labels])
    y = np.zeros((n_samples, 1))
    return [images, labels], y